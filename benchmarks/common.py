import ctypes
import gc
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from affine import Affine
from centerline.geometry import Centerline
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numba import cuda
from numba import njit
from numpy.typing import NDArray
from pygeoops import centerline as pygeoops_centerline
from rasterio.features import rasterize
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely import Geometry
from shapely import LineString
from shapely import MultiLineString
from shapely import MultiPolygon
from shapely import Polygon
from shapely import STRtree
from skimage.morphology import skeletonize
from tqdm import tqdm

from crestline import crestline

matplotlib.use('Agg')

Shape: TypeAlias = Polygon | MultiPolygon
Extractor: TypeAlias = Callable[[Shape, NDArray[np.bool_], Affine], Geometry]
MethodRegistry: TypeAlias = dict[str, tuple[Extractor, Extractor]]
Result: TypeAlias = dict[str, Any]

CUDA_METHOD = 'crestline (CUDA)'

RESULT_COLUMNS = [
    ('mpx', 'mpx', 1, 2),
    ('f1_5m', 'f5', 1, 4),
    ('f1_2m', 'f2', 1, 4),
    ('offset_m', 'offset', 1, 3),
    ('dead_ends', 'dead', 1, 1),
    ('true_dead_ends', 'true_dead', 1, 1),
    ('loops', 'loops', 1, 1),
    ('holes', 'holes', 1, 1),
    ('runtime_ms', 'time', 1000, 2),
]

ENDPOINT_SNAP_M = 0.5
"""Grid size in metres that line endpoints are snapped to before they are matched into junctions."""

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SURFACE_COLOR, LAND_COLOR, HAIRLINE_COLOR, AXIS_COLOR = '#fcfcfb', '#e4e2db', '#e1e0d9', '#c3c2b7'
INK_COLOR, SECONDARY_INK_COLOR, MUTED_COLOR = '#0b0b0b', '#52514e', '#898781'
ACCENT_COLOR, DEAD_END_COLOR, NEUTRAL_COLOR = '#2a78d6', '#eb6834', '#aeaca5'

logger = logging.getLogger('benchmarks')


@dataclass
class Scene:
    fields: dict[str, Any]
    label: str
    polygon: Shape
    mask: NDArray[np.bool_]
    transform: Affine
    centerlines: list[LineString]
    true_dead_ends: int
    holes: int


def configure_logging(log_level: str) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(log_level.upper())


def release_memory() -> None:
    gc.collect()
    with suppress(OSError, AttributeError):
        ctypes.CDLL('libc.so.6').malloc_trim(0)


def noisy_mask(
    polygon: Shape,
    seed: int,
    padding: int = 40,
    coarse_noise: float = 2.5,
    fine_noise: float = 0.8,
) -> tuple[NDArray[np.bool_], Affine]:
    random_generator = np.random.default_rng(seed + 1000)
    min_x, min_y, max_x, max_y = polygon.bounds
    raster_shape = (int(np.ceil(max_y - min_y)) + 2 * padding, int(np.ceil(max_x - min_x)) + 2 * padding)
    transform = from_origin(min_x - padding, max_y + padding, 1, 1)

    clean_mask = rasterize([(polygon, 1)], out_shape=raster_shape, transform=transform).astype(bool)
    signed_distance = distance_transform_edt(clean_mask) - distance_transform_edt(~clean_mask)

    coarse_field = gaussian_filter(random_generator.normal(size=raster_shape), 4.0)
    fine_field = gaussian_filter(random_generator.normal(size=raster_shape), 1.0)
    coarse_offset = coarse_field * coarse_noise / coarse_field.std()
    fine_offset = fine_field * fine_noise / fine_field.std()

    return signed_distance + coarse_offset + fine_offset > 0.5, transform


def vectorize(mask: NDArray[np.bool_], transform: Affine) -> Shape:
    shape_records = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    polygons = [shapely.geometry.shape(geometry) for geometry, _value in shape_records]
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


def as_lines(geometry: Geometry | None) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []

    lines = []
    for part in shapely.get_parts(geometry):
        if part.geom_type == 'LineString':
            lines.append(part)
        elif part.geom_type in ('MultiLineString', 'GeometryCollection'):
            lines.extend(as_lines(part))

    return [line for line in lines if line.length > 0]


def sample(lines: list[LineString], step: float = 0.5) -> NDArray[np.float64]:
    sampled = []
    for line in lines:
        point_count = max(int(np.ceil(line.length / step)), 1) + 1
        distances = np.linspace(0, line.length, point_count)
        sampled.append(shapely.get_coordinates(shapely.line_interpolate_point(line, distances)))

    return np.concatenate(sampled) if sampled else np.zeros((0, 2))


def end_keys(lines: list[LineString]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    noded_segments = shapely.get_parts(shapely.node(MultiLineString(lines)))
    endpoints = [coord for segment in noded_segments for coord in (segment.coords[0], segment.coords[-1])]
    snapped = np.round(np.array(endpoints) / ENDPOINT_SNAP_M).astype(np.int64)
    return np.unique(snapped, axis=0, return_counts=True)


def dead_ends(lines: list[LineString]) -> int:
    if not lines:
        return 0

    _, occurrences = end_keys(lines)
    return int((occurrences == 1).sum())


def end_points(lines: list[LineString]) -> NDArray[np.float64]:
    if not lines:
        return np.zeros((0, 2))

    grid_keys, occurrences = end_keys(lines)
    return grid_keys[occurrences == 1] * ENDPOINT_SNAP_M


def loops(lines: list[LineString]) -> int:
    if not lines:
        return 0

    noded_segments = shapely.get_parts(shapely.node(MultiLineString(lines)))
    return len(shapely.get_parts(shapely.polygonize(noded_segments)))


def holes(polygon: Shape, min_area: float = 50.0) -> int:
    return sum(1 for part in shapely.get_parts(polygon) for ring in part.interiors if Polygon(ring).area >= min_area)


def true_dead_ends(lines: list[LineString], half_widths: list[float] | NDArray[np.float64]) -> int:
    if not lines:
        return 0

    line_array = np.asarray(lines, dtype=object)
    half_width_array = np.asarray(half_widths, dtype=np.float64)
    endpoints = np.concatenate([shapely.get_point(line_array, 0), shapely.get_point(line_array, -1)])
    owner = np.tile(np.arange(len(line_array)), 2)

    reach = half_width_array.max() + 0.5 * half_width_array[owner] + 1.0
    endpoint_index, line_index = STRtree(line_array).query(endpoints, predicate='dwithin', distance=reach)
    is_other = line_index != owner[endpoint_index]
    endpoint_index, line_index = endpoint_index[is_other], line_index[is_other]

    gap = shapely.distance(endpoints[endpoint_index], line_array[line_index])
    is_joined = gap <= half_width_array[line_index] + 0.5 * half_width_array[owner[endpoint_index]]

    is_dead_end = ~shapely.is_closed(line_array)[owner]
    is_dead_end[endpoint_index[is_joined]] = False
    return int(is_dead_end.sum())


def score(predicted_lines: list[LineString], true_lines: list[LineString]) -> dict[str, float]:
    true_points, predicted_points = sample(true_lines), sample(predicted_lines)
    if len(predicted_points) == 0:
        return {'f5': 0.0, 'f2': 0.0, 'offset': float('nan')}

    predicted_to_truth = cKDTree(true_points).query(predicted_points)[0]
    truth_to_predicted = cKDTree(predicted_points).query(true_points)[0]

    metrics = {}
    for tolerance in (2, 5):
        precision = (predicted_to_truth <= tolerance).mean()
        recall = (truth_to_predicted <= tolerance).mean()
        metrics[f'f{tolerance}'] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    metrics['offset'] = float(predicted_to_truth[predicted_to_truth <= 5].mean())
    return metrics


@njit(cache=True)
def _drop_spurs(
    node_count: int, edges: NDArray[np.int64], edge_lengths: NDArray[np.float64], max_spur_length: float
) -> NDArray[np.bool_]:
    shifted_degree = np.zeros(node_count + 1, np.int64)

    for edge_index in range(len(edges)):
        shifted_degree[edges[edge_index, 0] + 1] += 1
        shifted_degree[edges[edge_index, 1] + 1] += 1

    adjacency_start = np.cumsum(shifted_degree)
    adjacency_node = np.empty(adjacency_start[-1], np.int64)
    adjacency_edge = np.empty(adjacency_start[-1], np.int64)
    insert_position = adjacency_start[:-1].copy()

    for edge_index in range(len(edges)):
        node_a, node_b = edges[edge_index, 0], edges[edge_index, 1]
        for node, neighbor in ((node_a, node_b), (node_b, node_a)):
            adjacency_node[insert_position[node]], adjacency_edge[insert_position[node]] = neighbor, edge_index
            insert_position[node] += 1

    degree = shifted_degree[1:].copy()
    edge_alive = np.ones(len(edges), np.bool_)
    chain_edges = np.empty(len(edges), np.int64)

    changed = True
    while changed:
        changed = False
        for start_node in range(node_count):
            if degree[start_node] != 1:
                continue

            current_node, previous_edge, chain_length, chain_edge_count = start_node, -1, 0.0, 0
            while True:
                next_node, next_edge = -1, -1
                for slot in range(adjacency_start[current_node], adjacency_start[current_node + 1]):
                    if edge_alive[adjacency_edge[slot]] and adjacency_edge[slot] != previous_edge:
                        next_node, next_edge = adjacency_node[slot], adjacency_edge[slot]
                        break

                if next_node < 0:
                    break

                chain_edges[chain_edge_count] = next_edge
                chain_edge_count += 1
                chain_length += edge_lengths[next_edge]
                current_node, previous_edge = next_node, next_edge

                if degree[current_node] != 2 or chain_length >= max_spur_length:
                    break

            if chain_length < max_spur_length and degree[current_node] != 2:
                for chain_position in range(chain_edge_count):
                    removed_edge = chain_edges[chain_position]
                    edge_alive[removed_edge] = False
                    degree[edges[removed_edge, 0]] -= 1
                    degree[edges[removed_edge, 1]] -= 1

                changed = True

    return edge_alive


def skeleton(mask: NDArray[np.bool_], transform: Affine, prune_factor: float = 2.0) -> LineString | MultiLineString:
    skeleton_raster = np.pad(skeletonize(mask), 1)
    pixel_rows, pixel_cols = np.nonzero(skeleton_raster)

    if len(pixel_rows) < 2:
        return MultiLineString()

    pixel_ids = np.full(skeleton_raster.shape, -1, np.int64)
    pixel_ids[pixel_rows, pixel_cols] = np.arange(len(pixel_rows))

    edge_blocks = []
    for row_step, col_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        is_linked = skeleton_raster[pixel_rows + row_step, pixel_cols + col_step]

        if row_step and col_step:
            column_neighbor = skeleton_raster[pixel_rows, pixel_cols + col_step]
            row_neighbor = skeleton_raster[pixel_rows + row_step, pixel_cols]
            is_linked &= ~column_neighbor & ~row_neighbor

        linked_rows, linked_cols = pixel_rows[is_linked], pixel_cols[is_linked]
        source_ids = pixel_ids[linked_rows, linked_cols]
        target_ids = pixel_ids[linked_rows + row_step, linked_cols + col_step]
        edge_blocks.append(np.c_[source_ids, target_ids])

    edges = np.concatenate(edge_blocks)
    row_deltas = pixel_rows[edges[:, 0]] - pixel_rows[edges[:, 1]]
    col_deltas = pixel_cols[edges[:, 0]] - pixel_cols[edges[:, 1]]
    edge_lengths = np.hypot(row_deltas, col_deltas)

    distance_to_edge = distance_transform_edt(mask)[pixel_rows - 1, pixel_cols - 1]
    max_spur_length = prune_factor * np.median(distance_to_edge)
    edges = edges[_drop_spurs(len(pixel_rows), edges, edge_lengths, max_spur_length)]

    if len(edges) == 0:
        return MultiLineString()

    node_coordinates = np.c_[transform * (pixel_cols - 0.5, pixel_rows - 0.5)]
    return shapely.line_merge(shapely.multilinestrings(shapely.linestrings(node_coordinates[edges])))


def cuda_device() -> tuple[str, str]:
    try:
        if not cuda.is_available():
            return '', 'no device found'

        device_name = cuda.get_current_device().name
    except Exception as error:
        return '', f'{type(error).__name__}: {error}'

    return (device_name.decode() if isinstance(device_name, bytes) else device_name), ''


def voronoi_centerline(geometry: Shape) -> Geometry:
    if geometry.geom_type == 'MultiPolygon':
        return shapely.union_all([voronoi_centerline(part) for part in geometry.geoms])

    try:
        return Centerline(geometry, interpolation_distance=1.0).geometry
    except Exception as error:
        logger.debug(f'Voronoi centerline failed on a {geometry.area:.1f} m² part: {error}')
        return MultiLineString()


def burn_polygon(polygon: Shape, mask: NDArray[np.bool_], transform: Affine) -> NDArray[np.bool_]:
    return rasterize([(polygon, 1)], out_shape=mask.shape, transform=transform).astype(bool)


def methods(with_cuda: bool = False) -> MethodRegistry:
    registry = {
        'crestline': (
            lambda polygon, mask, transform: crestline(polygon, resolution=0.5),
            lambda polygon, mask, transform: crestline(mask, transform=transform),
        ),
        'pygeoops.centerline': (
            lambda polygon, mask, transform: pygeoops_centerline(polygon),
            lambda polygon, mask, transform: pygeoops_centerline(vectorize(mask, transform)),
        ),
        'skeletonize + prune': (
            lambda polygon, mask, transform: skeleton(burn_polygon(polygon, mask, transform), transform),
            lambda polygon, mask, transform: skeleton(mask, transform),
        ),
        'centerline (Voronoi)': (
            lambda polygon, mask, transform: voronoi_centerline(polygon),
            lambda polygon, mask, transform: voronoi_centerline(vectorize(mask, transform)),
        ),
    }

    if with_cuda:
        registry[CUDA_METHOD] = (
            lambda polygon, mask, transform: crestline(polygon, resolution=1.0, device='cuda'),
            lambda polygon, mask, transform: crestline(mask, transform=transform, device='cuda'),
        )

    return registry


def benchmark_scene(
    scene: Scene,
    method_registry: MethodRegistry,
    repeat: int,
    progress: tqdm,
    retired: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    single_run_after: float = np.inf,
) -> list[Result]:
    results, cpu_outputs = [], {}

    for method_name, (run_on_polygon, run_on_mask) in method_registry.items():
        for source, extract in (('polygon', run_on_polygon), ('noisy mask', run_on_mask)):
            if (method_name, source) in retired:
                progress.update()
                continue

            progress.set_description(f'{scene.label} {method_name} ({source})')
            started = time.perf_counter()
            extract(scene.polygon, scene.mask, scene.transform)
            warm_up_time = time.perf_counter() - started

            is_single_run = method_name.startswith('centerline') or warm_up_time > single_run_after
            best_time = np.inf
            for _ in range(1 if is_single_run else repeat):
                started = time.perf_counter()
                output = extract(scene.polygon, scene.mask, scene.transform)
                best_time = min(best_time, time.perf_counter() - started)

            if method_name == 'crestline':
                cpu_outputs[source] = output

            predicted_lines = as_lines(output)
            result = {
                **scene.fields,
                'source': source,
                'method': method_name,
                'time': best_time,
                'mpx': scene.mask.size / 1e6,
            }
            result.update(
                score(predicted_lines, scene.centerlines),
                dead=dead_ends(predicted_lines),
                true_dead=scene.true_dead_ends,
                loops=loops(predicted_lines),
                holes=scene.holes,
            )

            if method_name == CUDA_METHOD and source in cpu_outputs:
                result['same'] = bool(output.equals_exact(cpu_outputs[source], 0))
                if not result['same']:
                    logger.warning(f'CUDA and CPU crestline disagree on {scene.label} ({source})')

            logger.debug(
                f'{method_name} on {scene.label} ({source}): F1@5m {result["f5"]:.3f}, '
                f'{result["dead"]}/{scene.true_dead_ends} dead ends, {result["loops"]} loops, '
                f'{1000 * best_time:.1f} ms after a {1000 * warm_up_time:.1f} ms warm-up'
            )

            results.append(result)
            progress.set_postfix(
                f5=f'{result["f5"]:.3f}',
                dead=f'{result["dead"]}/{scene.true_dead_ends}',
                time=f'{best_time:.3f}s',
            )
            progress.update()

            del output, predicted_lines
            release_memory()

    return results


def write_table(results: list[Result], group_columns: list[str], table_path: Path) -> pd.DataFrame:
    summary = (
        pd.DataFrame(results)
        .groupby(group_columns, sort=False)
        .agg(**{column: (key, 'mean') for column, key, _scale, _decimals in RESULT_COLUMNS})
        .reset_index()
    )

    for column, _key, scale, _decimals in RESULT_COLUMNS:
        if scale != 1:
            summary[column] *= scale

    summary = summary.round({column: decimals for column, _key, _scale, decimals in RESULT_COLUMNS})
    summary.to_csv(table_path, index=False, lineterminator='\n')
    logger.info(f'Wrote {len(summary)} summary rows to {table_path}')
    return summary


class Canvas:
    def __init__(self, width: float, height: float) -> None:
        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_family = ['Inter', 'DejaVu Sans'] if 'Inter' in installed_fonts else ['DejaVu Sans']
        plt.rcParams.update({'font.family': font_family, 'font.size': 9, 'text.color': SECONDARY_INK_COLOR})

        self.width, self.height = width, height
        self.figure = plt.figure(figsize=(width, height), facecolor=SURFACE_COLOR)

    def box(self, left: float, top: float, width: float = 0.0, height: float = 0.0) -> list[float]:
        return [left / self.width, 1 - (top + height) / self.height, width / self.width, height / self.height]

    def axes(self, left: float, top: float, width: float, height: float) -> Axes:
        return self.figure.add_axes(self.box(left, top, width, height))

    def text(self, left: float, top: float, label: str, **kwargs: Any) -> None:
        self.figure.text(left / self.width, 1 - top / self.height, label, **kwargs)

    def mark(self, lefts: list[float], top: float, **kwargs: Any) -> None:
        self.figure.lines.append(
            plt.Line2D(
                [left / self.width for left in lefts],
                [1 - top / self.height] * len(lefts),
                transform=self.figure.transFigure,
                **kwargs,
            )
        )

    def save(self, figure_path: Path, description: str) -> None:
        self.figure.savefig(figure_path, dpi=200, facecolor=SURFACE_COLOR)
        plt.close(self.figure)
        logger.info(f'Saved {description} to {figure_path}')


def draw_header(canvas: Canvas, title: str, subtitle: str, left_margin: float, right_margin: float) -> None:
    canvas.text(left_margin, 0.5, title, fontsize=14, color=INK_COLOR, weight='semibold')
    canvas.text(left_margin, 0.78, subtitle, fontsize=10)

    legend_right = canvas.width - right_margin
    canvas.text(legend_right, 0.78, 'centerline', fontsize=9, ha='right')
    canvas.mark([legend_right - 0.95, legend_right - 0.72], 0.745, color=ACCENT_COLOR, lw=2)
    canvas.text(legend_right - 1.12, 0.78, 'dead end', fontsize=9, ha='right')
    canvas.mark([legend_right - 1.82], 0.745, marker='o', ms=6.5, color=DEAD_END_COLOR, mec=SURFACE_COLOR, mew=1.2)


def draw_panels(
    canvas: Canvas,
    method_names: list[str],
    method_registry: MethodRegistry,
    scene: Scene,
    window: tuple[float, float, float, float],
    left_margin: float,
    right_margin: float,
    panel_gap: float = 0.14,
    panel_top: float = 1.55,
) -> None:
    window_min_x, window_min_y, window_max_x, window_max_y = window
    crop_col_start, crop_row_start = ~scene.transform * (window_min_x, window_max_y)
    crop_col_end, crop_row_end = ~scene.transform * (window_max_x, window_min_y)
    mask_crop = scene.mask[int(crop_row_start) : int(crop_row_end), int(crop_col_start) : int(crop_col_end)]

    panel_count = len(method_names)
    panel_size = (canvas.width - left_margin - right_margin - panel_gap * (panel_count - 1)) / panel_count

    for panel_index, method_name in enumerate(method_names):
        _, run_on_mask = method_registry[method_name]
        predicted_lines = as_lines(run_on_mask(scene.polygon, scene.mask, scene.transform))
        dead_end_points = end_points(predicted_lines)
        panel_left = left_margin + panel_index * (panel_size + panel_gap)

        panel_axes = canvas.axes(panel_left, panel_top, panel_size, panel_size)
        panel_axes.imshow(
            mask_crop,
            cmap=ListedColormap([SURFACE_COLOR, LAND_COLOR]),
            extent=(window_min_x, window_max_x, window_min_y, window_max_y),
            interpolation='nearest',
        )
        for line in predicted_lines:
            panel_axes.plot(*line.xy, color=ACCENT_COLOR, lw=1.4, solid_capstyle='round', solid_joinstyle='round')

        in_window = (
            (dead_end_points[:, 0] > window_min_x)
            & (dead_end_points[:, 0] < window_max_x)
            & (dead_end_points[:, 1] > window_min_y)
            & (dead_end_points[:, 1] < window_max_y)
        )
        panel_axes.plot(
            *dead_end_points[in_window].T, 'o', ms=5.5, color=DEAD_END_COLOR, mec=SURFACE_COLOR, mew=1.1, zorder=3
        )

        panel_axes.set_xlim(window_min_x, window_max_x)
        panel_axes.set_ylim(window_min_y, window_max_y)
        panel_axes.set_xticks([])
        panel_axes.set_yticks([])
        for spine in panel_axes.spines.values():
            spine.set_color(HAIRLINE_COLOR)

        panel_f1 = score(predicted_lines, scene.centerlines)['f5']
        canvas.text(panel_left, panel_top - 0.33, method_name, fontsize=11, color=INK_COLOR, weight='semibold')
        canvas.text(
            panel_left, panel_top - 0.1, f'F1 {panel_f1:.2f} · {dead_ends(predicted_lines):,} dead ends', fontsize=9.5
        )
