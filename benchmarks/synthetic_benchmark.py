import logging
import os
import time
from collections.abc import Callable
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
from jsonargparse import CLI
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
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
from skimage.morphology import skeletonize
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from crestline import crestline

matplotlib.use('Agg')

Shape: TypeAlias = Polygon | MultiPolygon
Extractor: TypeAlias = Callable[[Shape, NDArray[np.bool_], Affine], Geometry]
MethodRegistry: TypeAlias = dict[str, tuple[Extractor, Extractor]]
Case: TypeAlias = tuple[list[LineString], list[float], Shape]
Result: TypeAlias = dict[str, Any]

CUDA_METHOD = 'crestline (CUDA)'

RESULT_COLUMNS = [
    ('mpx', 'mpx', 1, 2),  # mask size in megapixels
    ('f1_5m', 'f5', 1, 4),  # F1 score with a 5 m match tolerance
    ('f1_2m', 'f2', 1, 4),  # F1 score with a 2 m match tolerance
    ('offset_m', 'offset', 1, 3),  # mean distance of predicted points within 5 m of the truth
    ('dead_ends', 'dead', 1, 1),  # dead ends in the predicted network
    ('true_dead_ends', 'true_dead', 1, 1),  # dead ends the ground truth really has
    ('loops', 'loops', 1, 1),  # closed loops in the predicted network
    ('holes', 'holes', 1, 1),  # holes in the input shape
    ('runtime_ms', 'time', 1000, 2),  # best runtime, stored in seconds and scaled to milliseconds
]

ENDPOINT_SNAP_M = 0.5
"""Grid size in metres that line endpoints are snapped to before they are matched into junctions."""

ROAD_HALF_WIDTH_M = 15.0
"""Buffer radius of every road in the uniform width road cases, which makes each road 30 m wide."""

MIXED_HALF_WIDTHS_M = [6.0, 9.0, 14.0, 20.0, 30.0]
"""Buffer radii the mixed case picks from at random, one per road, giving roads 12 to 60 m wide."""

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(Path(__file__).stem)


def road_network(
    seed: int,
    main_road_count: int = 5,
    side_road_count: int = 8,
    extent: float = 1000.0,
) -> list[LineString]:
    random_generator = np.random.default_rng(seed)
    roads = []

    for road_index in range(main_road_count):
        along_positions = np.linspace(0, extent, 5)
        base_offset = random_generator.uniform(0.15, 0.85) * extent
        drift = np.cumsum(random_generator.normal(0, 50, 5))
        across_positions = base_offset + drift

        if road_index % 2 == 0:
            vertices = np.c_[along_positions, across_positions]
        else:
            vertices = np.c_[across_positions, along_positions]

        roads.append(LineString(vertices))

    for _ in range(side_road_count):
        parent_road = roads[random_generator.integers(main_road_count)]
        anchor = parent_road.interpolate(random_generator.uniform(0.1, 0.9), normalized=True)
        heading, road_length = random_generator.uniform(0, 2 * np.pi), random_generator.uniform(80, 220)
        tip = (anchor.x + road_length * np.cos(heading), anchor.y + road_length * np.sin(heading))
        roads.append(LineString([(anchor.x, anchor.y), tip]))

    return roads


def river(
    seed: int,
    extent: float = 1000.0,
    tributary_count: int = 4,
) -> tuple[list[LineString], list[tuple[float, float]]]:
    random_generator = np.random.default_rng(seed)
    trunk_progress = np.linspace(0, 1, 60)
    trunk_meander = np.cumsum(random_generator.normal(0, 12, 60))
    trunk = LineString(np.c_[trunk_progress * extent, extent * 0.5 + trunk_meander])
    centerlines, radii = [trunk], [(5.0, 30.0)]

    for _ in range(tributary_count):
        junction_fraction = random_generator.uniform(0.3, 0.9)
        junction = trunk.interpolate(junction_fraction, normalized=True)
        heading = random_generator.choice([-1, 1]) * random_generator.uniform(0.6, 1.3) + np.pi
        tributary_length = random_generator.uniform(150, 300)

        tributary_progress = np.linspace(0, 1, 20)
        wiggle = np.cumsum(random_generator.normal(0, 4, 20))
        x_coords = junction.x + tributary_progress * tributary_length * np.cos(heading) - wiggle * np.sin(heading)
        y_coords = junction.y + tributary_progress * tributary_length * np.sin(heading) + wiggle * np.cos(heading)

        centerlines.append(LineString(np.c_[x_coords, y_coords]))
        radii.append((5 + 12.5 * junction_fraction, 5.0))

    return centerlines, radii


def tapered(line: LineString, start_radius: float, end_radius: float, step: float = 2.0) -> Shape:
    station_count = max(int(line.length / step), 2)
    stations = [line.interpolate(distance) for distance in np.linspace(0, line.length, station_count)]
    radii = np.linspace(start_radius, end_radius, station_count)
    segments = zip(stations[:-1], stations[1:], radii)

    return shapely.union_all(
        [LineString([(start.x, start.y), (end.x, end.y)]).buffer(radius) for start, end, radius in segments]
    )


def uniform_width_case(centerlines: list[LineString]) -> Case:
    half_widths = [ROAD_HALF_WIDTH_M] * len(centerlines)
    polygon = MultiLineString(centerlines).buffer(ROAD_HALF_WIDTH_M)
    return centerlines, half_widths, polygon


def make_case(case_kind: str, seed: int) -> Case:
    if case_kind == 'roads':
        return uniform_width_case(road_network(seed))

    if case_kind == 'roads-large':
        return uniform_width_case(road_network(seed, main_road_count=20, side_road_count=32, extent=4000.0))

    if case_kind == 'mixed':
        centerlines = road_network(seed)
        half_widths = list(np.random.default_rng(seed + 100).choice(MIXED_HALF_WIDTHS_M, len(centerlines)))
        polygon = shapely.union_all([line.buffer(half_width) for line, half_width in zip(centerlines, half_widths)])
        return centerlines, half_widths, polygon

    if case_kind == 'river':
        centerlines, radii = river(seed)
        polygon = shapely.union_all([tapered(line, *radius_pair) for line, radius_pair in zip(centerlines, radii)])
        return centerlines, [max(radius_pair) for radius_pair in radii], polygon

    raise ValueError(f'Unknown case kind: {case_kind}')


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


def true_dead_ends(lines: list[LineString], half_widths: list[float]) -> int:
    dead_end_count = 0
    for line_index, line in enumerate(lines):
        for endpoint in (shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])):
            if all(
                endpoint.distance(other_line) > half_widths[other_index] + 0.5 * half_widths[line_index]
                for other_index, other_line in enumerate(lines)
                if other_index != line_index
            ):
                dead_end_count += 1

    return dead_end_count


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
            lambda polygon, mask, transform: crestline(polygon, resolution=1.0),
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


def run(plan: list[tuple[str, int]], method_registry: MethodRegistry, repeat: int) -> list[Result]:
    total_runs = 2 * len(method_registry)
    results, cpu_outputs = [], {}

    with tqdm(total=total_runs, unit='run') as progress:
        for case_kind, seed in plan:
            centerlines, half_widths, polygon = make_case(case_kind, seed)
            mask, transform = noisy_mask(polygon, seed)
            true_dead_end_count = true_dead_ends(centerlines, half_widths)
            hole_count = holes(polygon)
            logger.info(
                f'Case {case_kind} seed {seed}: {len(centerlines)} centerlines, {mask.size / 1e6:.2f} Mpx mask, '
                f'{true_dead_end_count} true dead ends, {hole_count} holes'
            )

            for method_name, (run_on_polygon, run_on_mask) in method_registry.items():
                for source, extract in (('polygon', run_on_polygon), ('noisy mask', run_on_mask)):
                    progress.set_description(f'{case_kind} s{seed} {method_name} ({source})')
                    extract(polygon, mask, transform)

                    timed_runs = repeat if not method_name.startswith('centerline') else 1
                    best_time = np.inf
                    for _ in range(timed_runs):
                        started = time.perf_counter()
                        output = extract(polygon, mask, transform)
                        best_time = min(best_time, time.perf_counter() - started)

                    if method_name == 'crestline':
                        cpu_outputs[source] = output

                    predicted_lines = as_lines(output)
                    result = {
                        'case': case_kind,
                        'seed': seed,
                        'source': source,
                        'method': method_name,
                        'time': best_time,
                        'mpx': mask.size / 1e6,
                    }
                    result.update(
                        score(predicted_lines, centerlines),
                        dead=dead_ends(predicted_lines),
                        true_dead=true_dead_end_count,
                        loops=loops(predicted_lines),
                        holes=hole_count,
                    )

                    if method_name == CUDA_METHOD:
                        result['same'] = bool(output.equals_exact(cpu_outputs[source], 0))
                        if not result['same']:
                            logger.warning(f'CUDA and CPU crestline disagree on {case_kind} seed {seed} ({source})')

                    logger.debug(
                        f'{method_name} on {case_kind} seed {seed} ({source}): F1@5m {result["f5"]:.3f}, '
                        f'{result["dead"]}/{true_dead_end_count} dead ends, {result["loops"]} loops, '
                        f'{1000 * best_time:.1f} ms'
                    )

                    results.append(result)
                    progress.set_postfix(
                        f5=f'{result["f5"]:.3f}',
                        dead=f'{result["dead"]}/{true_dead_end_count}',
                        time=f'{best_time:.3f}s',
                    )
                    progress.update()

    return results


def figure(
    results: list[Result],
    method_registry: MethodRegistry,
    case_kind: str = 'roads',
    seed: int = 1,
    window: tuple[float, float, float, float] = (470, 270, 870, 670),
) -> None:
    logger.info(f'Drawing benchmark figure from {case_kind} seed {seed}')

    surface_color, land_color, hairline_color, axis_color = '#fcfcfb', '#e4e2db', '#e1e0d9', '#c3c2b7'
    ink_color, secondary_ink_color, muted_color = '#0b0b0b', '#52514e', '#898781'
    accent_color, dead_end_color, neutral_color = '#2a78d6', '#eb6834', '#aeaca5'

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = ['Inter', 'DejaVu Sans'] if 'Inter' in installed_fonts else ['DejaVu Sans']
    plt.rcParams.update({'font.family': font_family, 'font.size': 9, 'text.color': secondary_ink_color})

    method_names = [method_name for method_name in method_registry if method_name != CUDA_METHOD]
    centerlines, half_widths, polygon = make_case(case_kind, seed)
    mask, transform = noisy_mask(polygon, seed)
    true_dead_end_count = true_dead_ends(centerlines, half_widths)

    window_min_x, window_min_y, window_max_x, window_max_y = window
    crop_col_start, crop_row_start = ~transform * (window_min_x, window_max_y)
    crop_col_end, crop_row_end = ~transform * (window_max_x, window_min_y)
    mask_crop = mask[int(crop_row_start) : int(crop_row_end), int(crop_col_start) : int(crop_col_end)]

    figure_width, figure_height, left_margin, right_margin, panel_gap = 12.0, 7.5, 0.45, 0.25, 0.14
    panel_size = (figure_width - left_margin - right_margin - panel_gap * (len(method_names) - 1)) / len(method_names)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor=surface_color)

    def figure_box(
        left_inches: float, top_inches: float, width_inches: float = 0.0, height_inches: float = 0.0
    ) -> list[float]:
        return [
            left_inches / figure_width,
            1 - (top_inches + height_inches) / figure_height,
            width_inches / figure_width,
            height_inches / figure_height,
        ]

    def place_text(left_inches: float, top_inches: float, label: str, **kwargs: Any) -> None:
        fig.text(left_inches / figure_width, 1 - top_inches / figure_height, label, **kwargs)

    place_text(
        left_margin,
        0.5,
        f'Noisy road mask, {window_max_x - window_min_x:.0f} × {window_max_y - window_min_y:.0f} m window',
        fontsize=14,
        color=ink_color,
        weight='semibold',
    )
    place_text(
        left_margin,
        0.78,
        f'Scores in each title are for the whole network; it has {true_dead_end_count} true dead ends.',
        fontsize=10,
    )

    legend_right = figure_width - right_margin
    place_text(legend_right, 0.78, 'centerline', fontsize=9, ha='right')
    fig.lines.append(
        plt.Line2D(
            [(legend_right - 0.95) / figure_width, (legend_right - 0.72) / figure_width],
            [1 - 0.745 / figure_height] * 2,
            color=accent_color,
            lw=2,
            transform=fig.transFigure,
        )
    )
    place_text(legend_right - 1.12, 0.78, 'dead end', fontsize=9, ha='right')
    fig.lines.append(
        plt.Line2D(
            [(legend_right - 1.82) / figure_width],
            [1 - 0.745 / figure_height],
            marker='o',
            ms=6.5,
            color=dead_end_color,
            mec=surface_color,
            mew=1.2,
            transform=fig.transFigure,
        )
    )

    for panel_index, method_name in enumerate(method_names):
        _, run_on_mask = method_registry[method_name]
        predicted_lines = as_lines(run_on_mask(polygon, mask, transform))
        dead_end_points = end_points(predicted_lines)
        panel_left = left_margin + panel_index * (panel_size + panel_gap)

        panel_axes = fig.add_axes(figure_box(panel_left, 1.55, panel_size, panel_size))
        panel_axes.imshow(
            mask_crop,
            cmap=ListedColormap([surface_color, land_color]),
            extent=(window_min_x, window_max_x, window_min_y, window_max_y),
            interpolation='nearest',
        )
        for line in predicted_lines:
            panel_axes.plot(*line.xy, color=accent_color, lw=1.4, solid_capstyle='round', solid_joinstyle='round')

        in_window = (
            (dead_end_points[:, 0] > window_min_x)
            & (dead_end_points[:, 0] < window_max_x)
            & (dead_end_points[:, 1] > window_min_y)
            & (dead_end_points[:, 1] < window_max_y)
        )
        panel_axes.plot(
            *dead_end_points[in_window].T, 'o', ms=5.5, color=dead_end_color, mec=surface_color, mew=1.1, zorder=3
        )

        panel_axes.set_xlim(window_min_x, window_max_x)
        panel_axes.set_ylim(window_min_y, window_max_y)
        panel_axes.set_xticks([])
        panel_axes.set_yticks([])
        for spine in panel_axes.spines.values():
            spine.set_color(hairline_color)

        panel_f1 = score(predicted_lines, centerlines)['f5']
        place_text(panel_left, 1.22, method_name, fontsize=11, color=ink_color, weight='semibold')
        place_text(panel_left, 1.45, f'F1 {panel_f1:.2f} · {dead_ends(predicted_lines):,} dead ends', fontsize=9.5)

    noisy_results = [
        result
        for result in results
        if result['source'] == 'noisy mask' and result['case'] in ('roads', 'mixed', 'river')
    ]
    timed_results = [result for result in results if result['source'] == 'noisy mask' and result['case'] == 'roads']

    def method_mean(subset: list[Result], key: str, method_name: str) -> float:
        return np.mean([result[key] for result in subset if result['method'] == method_name])

    mean_true_dead_ends = method_mean(noisy_results, 'true_dead', method_names[0])
    chart_specs = [
        (
            'F1 @ 5 m',
            'higher is better',
            [method_mean(noisy_results, 'f5', method_name) for method_name in method_names],
            False,
            '{:.3f}',
        ),
        (
            'Dead ends',
            f'closer to the true {mean_true_dead_ends:.1f} is better',
            [method_mean(noisy_results, 'dead', method_name) for method_name in method_names],
            True,
            '{:,.0f}',
        ),
        (
            'Runtime, 1.2 Mpx mask',
            'lower is better',
            [1000 * method_mean(timed_results, 'time', method_name) for method_name in method_names],
            True,
            '{:,.0f} ms',
        ),
    ]

    charts_top, label_column_width, chart_gap, chart_height = 5.45, 1.45, 0.4, 1.6
    chart_width = (figure_width - left_margin - label_column_width - right_margin - 2 * chart_gap) / 3
    place_text(
        left_margin,
        charts_top - 0.75,
        'Noisy masks, means over the 1 km road networks, mixed-width road networks and rivers',
        fontsize=10,
    )

    row_positions = np.arange(len(method_names))[::-1]
    for chart_index, (chart_title, chart_hint, chart_values, use_log_scale, value_format) in enumerate(chart_specs):
        chart_left = left_margin + label_column_width + chart_index * (chart_width + chart_gap)
        chart_axes = fig.add_axes(figure_box(chart_left, charts_top, chart_width, chart_height))
        chart_axes.set_facecolor(surface_color)

        x_min, x_max = (min(chart_values) / 2.2, max(chart_values) * 4.5) if use_log_scale else (0.4, 1.04)
        if use_log_scale:
            chart_axes.set_xscale('log')

        chart_axes.set_xlim(x_min, x_max)
        chart_axes.set_ylim(-0.55, len(method_names) - 0.45)
        for row_y in row_positions:
            chart_axes.axhline(row_y, color=hairline_color, lw=0.8, zorder=0)

        if chart_title == 'Dead ends':
            chart_axes.axvline(mean_true_dead_ends, color=muted_color, lw=0.9, zorder=1)

        for row_y, value, method_name in zip(row_positions, chart_values, method_names):
            is_crestline = method_name == 'crestline'
            chart_axes.plot(
                [value],
                [row_y],
                'o',
                ms=8.5,
                color=accent_color if is_crestline else neutral_color,
                mec=surface_color,
                mew=1.8,
                zorder=3,
            )

            is_below_truth = chart_title == 'Dead ends' and value < mean_true_dead_ends
            label_offset = (1 / 1.2 if is_below_truth else 1.2) if use_log_scale else 0.02
            chart_axes.text(
                value * label_offset if use_log_scale else value + label_offset,
                row_y,
                value_format.format(value),
                va='center',
                ha='right' if is_below_truth else 'left',
                fontsize=9,
                color=ink_color if is_crestline else secondary_ink_color,
                weight='semibold' if is_crestline else 'normal',
            )

        chart_axes.set_yticks([])
        if chart_index == 0:
            for row_y, method_name in zip(row_positions, method_names):
                chart_axes.text(
                    -0.04,
                    row_y,
                    method_name,
                    transform=chart_axes.get_yaxis_transform(),
                    ha='right',
                    va='center',
                    fontsize=9.5,
                    color=ink_color if method_name == 'crestline' else secondary_ink_color,
                )

        chart_axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=muted_color, pad=5)
        if use_log_scale:
            tick_values = [tick for tick in (1, 10, 100, 1000, 10000) if x_min <= tick <= x_max]
            chart_axes.xaxis.set_major_locator(FixedLocator(tick_values))
            chart_axes.xaxis.set_minor_locator(NullLocator())
            chart_axes.set_xticklabels([f'{tick:,}' for tick in tick_values])
        else:
            chart_axes.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

        for side in ('top', 'right', 'left'):
            chart_axes.spines[side].set_visible(False)

        chart_axes.spines['bottom'].set_color(axis_color)
        place_text(chart_left, charts_top - 0.38, chart_title, fontsize=10.5, color=ink_color, weight='semibold')
        place_text(chart_left, charts_top - 0.16, chart_hint, fontsize=8.5, color=muted_color)

    figure_path = OUTPUT_DIR / 'synthetic_benchmark.png'
    fig.savefig(figure_path, dpi=200, facecolor=surface_color)
    plt.close(fig)
    logger.info(f'Saved benchmark figure to {figure_path}')


def cuda_benchmark_figure(results: list[Result], gpu_name: str) -> None:
    surface_color, hairline_color, axis_color = '#fcfcfb', '#e1e0d9', '#c3c2b7'
    ink_color, secondary_ink_color, muted_color = '#0b0b0b', '#52514e', '#898781'
    accent_color, neutral_color = '#2a78d6', '#aeaca5'

    plt.rcParams.update({'font.size': 9, 'text.color': secondary_ink_color})

    case_labels = {
        'river': 'rivers',
        'roads': 'road networks',
        'mixed': 'mixed-width roads',
        'roads-large': '4 km road network',
    }
    cuda_cases = {result['case'] for result in results if result['method'] == CUDA_METHOD}
    mean_mpx = {
        case_kind: np.mean([result['mpx'] for result in results if result['case'] == case_kind])
        for case_kind in cuda_cases
    }
    ordered_cases = sorted(mean_mpx, key=mean_mpx.get)
    cuda_matches = [result['same'] for result in results if result['method'] == CUDA_METHOD]

    def mean_runtime_ms(case_kind: str, source: str, method_name: str) -> float:
        return 1000 * np.mean(
            [
                result['time']
                for result in results
                if (result['case'], result['source'], result['method']) == (case_kind, source, method_name)
            ]
        )

    figure_width, left_margin, right_margin, label_column_width, chart_gap = 12.0, 0.45, 0.25, 2.1, 0.5
    chart_height = 0.42 * len(ordered_cases)
    figure_height = 2.0 + chart_height
    chart_width = (figure_width - left_margin - label_column_width - right_margin - chart_gap) / 2
    charts_top = 1.5
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor=surface_color)

    def place_text(left_inches: float, top_inches: float, label: str, **kwargs: Any) -> None:
        fig.text(left_inches / figure_width, 1 - top_inches / figure_height, label, **kwargs)

    if all(cuda_matches):
        match_summary = 'same lines in all'
    else:
        match_summary = f'different lines in {cuda_matches.count(False)} of'

    place_text(left_margin, 0.5, 'crestline on the CPU and on CUDA', fontsize=14, color=ink_color, weight='semibold')
    place_text(
        left_margin,
        0.78,
        f'CPU with {os.cpu_count()} cores · {gpu_name} · best runtime per shape, '
        f'averaged over seeds · {match_summary} {len(cuda_matches)} runs',
        fontsize=10,
    )

    for legend_index, (legend_label, legend_color) in enumerate((('CUDA', accent_color), ('CPU', neutral_color))):
        legend_right = figure_width - right_margin - legend_index * 0.95
        place_text(legend_right, 0.78, legend_label, fontsize=9, ha='right')
        fig.lines.append(
            plt.Line2D(
                [(legend_right - 0.5) / figure_width],
                [1 - 0.745 / figure_height],
                marker='o',
                ms=7,
                color=legend_color,
                mec=surface_color,
                mew=1.4,
                transform=fig.transFigure,
            )
        )

    row_positions = np.arange(len(ordered_cases))[::-1]
    for chart_index, source in enumerate(('polygon', 'noisy mask')):
        cpu_runtimes = [mean_runtime_ms(case_kind, source, 'crestline') for case_kind in ordered_cases]
        gpu_runtimes = [mean_runtime_ms(case_kind, source, CUDA_METHOD) for case_kind in ordered_cases]
        chart_left = left_margin + label_column_width + chart_index * (chart_width + chart_gap)

        chart_axes = fig.add_axes(
            [
                chart_left / figure_width,
                1 - (charts_top + chart_height) / figure_height,
                chart_width / figure_width,
                chart_height / figure_height,
            ]
        )
        chart_axes.set_facecolor(surface_color)
        chart_axes.set_xscale('log')
        x_min, x_max = min(cpu_runtimes + gpu_runtimes) / 2.5, max(cpu_runtimes + gpu_runtimes) * 7
        chart_axes.set_xlim(x_min, x_max)
        chart_axes.set_ylim(-0.55, len(ordered_cases) - 0.45)

        for row_y, cpu_runtime_ms, gpu_runtime_ms in zip(row_positions, cpu_runtimes, gpu_runtimes):
            chart_axes.axhline(row_y, color=hairline_color, lw=0.8, zorder=0)
            chart_axes.plot(
                [cpu_runtime_ms, gpu_runtime_ms],
                [row_y, row_y],
                color=axis_color,
                lw=2,
                solid_capstyle='round',
                zorder=1,
            )
            chart_axes.plot(
                [cpu_runtime_ms], [row_y], 'o', ms=8.5, color=neutral_color, mec=surface_color, mew=1.8, zorder=3
            )
            chart_axes.plot(
                [gpu_runtime_ms], [row_y], 'o', ms=8.5, color=accent_color, mec=surface_color, mew=1.8, zorder=3
            )

            if cpu_runtime_ms >= gpu_runtime_ms:
                speed_label = f'{cpu_runtime_ms / gpu_runtime_ms:.1f}× faster'
            else:
                speed_label = f'{gpu_runtime_ms / cpu_runtime_ms:.1f}× slower'

            if max(cpu_runtime_ms, gpu_runtime_ms) < 1.05 * min(cpu_runtime_ms, gpu_runtime_ms):
                speed_label = 'same'

            chart_axes.text(
                max(cpu_runtime_ms, gpu_runtime_ms) * 1.25,
                row_y,
                f'{gpu_runtime_ms:,.0f} ms · {speed_label}',
                va='center',
                fontsize=9,
                color=ink_color,
            )

        if chart_index == 0:
            for row_y, case_kind in zip(row_positions, ordered_cases):
                chart_axes.text(
                    -0.04,
                    row_y,
                    f'{case_labels.get(case_kind, case_kind)} ({mean_mpx[case_kind]:.1f} Mpx)',
                    transform=chart_axes.get_yaxis_transform(),
                    ha='right',
                    va='center',
                    fontsize=9.5,
                )

        chart_axes.set_yticks([])
        tick_values = [tick for tick in (1, 10, 100, 1000, 10000) if x_min <= tick <= x_max]
        chart_axes.xaxis.set_major_locator(FixedLocator(tick_values))
        chart_axes.xaxis.set_minor_locator(NullLocator())
        chart_axes.set_xticklabels([f'{tick:,} ms' for tick in tick_values])
        chart_axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=muted_color, pad=5)

        for side in ('top', 'right', 'left'):
            chart_axes.spines[side].set_visible(False)

        chart_axes.spines['bottom'].set_color(axis_color)
        place_text(
            chart_left, charts_top - 0.22, f'Runtime, {source}', fontsize=10.5, color=ink_color, weight='semibold'
        )

    figure_path = OUTPUT_DIR / 'synthetic_benchmark_cuda.png'
    fig.savefig(figure_path, dpi=200, facecolor=surface_color)
    plt.close(fig)
    logger.info(f'Saved CUDA comparison figure to {figure_path}')


def table(results: list[Result]) -> None:
    summary = (
        pd.DataFrame(results)
        .groupby(['case', 'source', 'method'], sort=False)
        .agg(
            mpx=('mpx', 'mean'),
            f1_5m=('f5', 'mean'),
            f1_2m=('f2', 'mean'),
            offset_m=('offset', 'mean'),
            dead_ends=('dead', 'mean'),
            true_dead_ends=('true_dead', 'mean'),
            loops=('loops', 'mean'),
            holes=('holes', 'mean'),
            runtime_ms=('time', 'mean'),
        )
        .reset_index()
    )
    summary['runtime_ms'] *= 1000

    decimal_places = {
        'mpx': 2,
        'f1_5m': 4,
        'f1_2m': 4,
        'offset_m': 3,
        'dead_ends': 1,
        'true_dead_ends': 1,
        'loops': 1,
        'holes': 1,
        'runtime_ms': 2,
    }
    table_path = OUTPUT_DIR / 'synthetic_benchmark_results.csv'
    summary.round(decimal_places).to_csv(table_path, index=False, lineterminator='\n')
    logger.info(f'Wrote {len(summary)} summary rows to {table_path}')


def main(
    cases: list[str] | None = None,
    seeds: list[int] | None = None,
    large_seeds: list[int] | None = None,
    repeat: int = 3,
    log_level: str = 'INFO',
) -> None:
    """Run the synthetic centerline benchmark and write the summary table and figures to ``output/``.

    :param cases: Case kinds to run; defaults to: 'roads', 'mixed', 'river' and 'roads-large'
    :param seeds: Seeds for the 1 km cases; defaults to 0 through 5
    :param large_seeds: Seeds for the 4 km and 8 km road networks; defaults to 1 and 2
    :param repeat: Timed runs per method and input; the fastest one is kept
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(log_level.upper())

    if cases is None:
        cases = ['roads', 'mixed', 'river', 'roads-large']

    if seeds is None:
        seeds = [0, 1, 2, 3, 4, 5]

    if large_seeds is None:
        large_seeds = [1, 2]

    gpu_name, cuda_failure_reason = cuda_device()
    if gpu_name:
        logger.info(f'CUDA: {gpu_name}')
    else:
        logger.info(f'CUDA: off ({cuda_failure_reason}), running on the CPU only')

    cuda_available = bool(gpu_name)
    method_registry = methods(cuda_available)
    logger.info(f'Methods: {", ".join(method_registry)}')

    plan = [
        (case_kind, seed) for case_kind in cases for seed in (large_seeds if case_kind.startswith('roads-') else seeds)
    ]
    logger.info(f'Plan: {len(plan)} cases over {", ".join(cases)}, {repeat} timed repeats per run')

    started = time.perf_counter()
    with logging_redirect_tqdm():
        results = run(plan, method_registry, repeat)

    logger.info(f'Benchmark finished in {time.perf_counter() - started:.1f} s with {len(results)} results')

    table(results)

    if 'roads' in cases:
        figure(results, method_registry)

    if cuda_available:
        cuda_benchmark_figure(results, gpu_name)


if __name__ == '__main__':
    CLI(main)
