import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from common import ACCENT_COLOR
from common import AXIS_COLOR
from common import CUDA_METHOD
from common import HAIRLINE_COLOR
from common import INK_COLOR
from common import MUTED_COLOR
from common import NEUTRAL_COLOR
from common import OUTPUT_DIR
from common import SECONDARY_INK_COLOR
from common import SURFACE_COLOR
from common import Canvas
from common import MethodRegistry
from common import Result
from common import Scene
from common import Shape
from common import benchmark_scene
from common import burn_polygon
from common import configure_logging
from common import cuda_device
from common import draw_header
from common import draw_panels
from common import holes
from common import logger
from common import methods
from common import noisy_mask
from common import release_memory
from common import true_dead_ends
from common import write_table
from jsonargparse import CLI
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray
from scipy.ndimage import label
from shapely import Geometry
from shapely import MultiLineString
from shapely import STRtree
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DATA_DIR = Path(__file__).parent / 'data'
ROADS_PATH = DATA_DIR / 'ldz_road_network.gpkg'
CENTERLINES_PATH = DATA_DIR / 'ldz_road_network_centerlines.gpkg'
TABLE_PATH = OUTPUT_DIR / 'real_benchmark_results.csv'
FIGURE_PATH = OUTPUT_DIR / 'real_benchmark.png'

GEOPACKAGE_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
HALF_WIDTH_STEP_M = 2.0
SINGLE_RUN_AFTER_S = 10.0
LABEL_PLACEMENTS = [
    ((7, 0), 'left', 'center'),
    ((4, 5), 'left', 'bottom'),
    ((4, -5), 'left', 'top'),
    ((0, 8), 'center', 'bottom'),
    ((0, -8), 'center', 'top'),
    ((-7, 0), 'right', 'center'),
]
METHOD_STYLES = {
    'crestline': ('crestline', 'o'),
    'pygeoops.centerline': ('pygeoops', 's'),
    'skeletonize + prune': ('skeletonize', '^'),
    'centerline (Voronoi)': ('Voronoi', 'D'),
}


@dataclass
class Network:
    polygon: Shape
    roads: NDArray[np.object_]
    half_widths: NDArray[np.float64]
    tree: STRtree


def read_geopackage(path: Path) -> tuple[list[Geometry], int]:
    with closing(sqlite3.connect(path)) as connection:
        ((table_name, column_name, epsg),) = connection.execute(
            'SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns'
        ).fetchall()
        blobs = [blob for (blob,) in connection.execute(f'SELECT "{column_name}" FROM "{table_name}"') if blob]

    envelope_bytes = [GEOPACKAGE_ENVELOPE_BYTES[(blob[3] >> 1) & 0b111] for blob in blobs]
    return [shapely.from_wkb(bytes(blob[8 + skip :])) for blob, skip in zip(blobs, envelope_bytes)], epsg


def polygonal(geometry: Geometry) -> Shape:
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry

    parts = shapely.get_parts(geometry)
    is_polygonal = np.isin(shapely.get_type_id(parts), (3, 6))
    return shapely.multipolygons(shapely.get_parts(parts[is_polygonal]))


def road_half_widths(polygon: Shape, roads: NDArray[np.object_]) -> NDArray[np.float64]:
    rings = shapely.get_rings(shapely.get_parts(polygon))
    coordinates, ring_index = shapely.get_coordinates(rings, return_index=True)
    is_edge = ring_index[1:] == ring_index[:-1]
    edges = shapely.linestrings(np.stack([coordinates[:-1][is_edge], coordinates[1:][is_edge]], axis=1))

    station_counts = np.maximum(np.ceil(shapely.length(roads) / HALF_WIDTH_STEP_M).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(roads)), station_counts)
    first_station = np.repeat(np.cumsum(station_counts) - station_counts, station_counts)
    fractions = (np.arange(len(owner)) - first_station + 0.5) / station_counts[owner]
    stations = shapely.line_interpolate_point(roads[owner], fractions, normalized=True)

    (station_index, _edge_index), distances = STRtree(edges).query_nearest(stations, return_distance=True)
    distance_to_edge = np.empty(len(stations))
    distance_to_edge[station_index] = distances
    return pd.Series(distance_to_edge).groupby(owner).median().to_numpy()


def load_network() -> Network:
    started = time.perf_counter()
    polygons, polygon_epsg = read_geopackage(ROADS_PATH)
    centerlines, centerline_epsg = read_geopackage(CENTERLINES_PATH)
    if polygon_epsg != centerline_epsg:
        raise ValueError(f'Road polygons are in EPSG:{polygon_epsg} but centerlines in EPSG:{centerline_epsg}')

    polygon = polygonal(polygons[0] if len(polygons) == 1 else shapely.union_all(polygons))
    roads = shapely.get_parts(shapely.get_parts(np.asarray(centerlines, dtype=object)))
    roads = roads[shapely.length(roads) > 0]
    half_widths = road_half_widths(polygon, roads)

    logger.info(
        f'Loaded {len(roads):,} centerlines over {shapely.length(roads).sum() / 1000:,.0f} km and '
        f'{polygon.area / 1e6:.1f} km² of road surface in EPSG:{polygon_epsg}, median half-width '
        f'{np.median(half_widths):.1f} m, in {time.perf_counter() - started:.1f} s'
    )
    return Network(polygon, roads, half_widths, STRtree(roads))


def make_scene(
    network: Network,
    center: tuple[float, float],
    radius_km: int,
    seed: int,
    noise: tuple[float, float],
) -> Scene:
    buffer = shapely.Point(center).buffer(radius_km * 1000.0, quad_segs=64)
    polygon = polygonal(shapely.intersection(network.polygon, buffer))

    candidates = network.tree.query(buffer, predicate='intersects')
    pieces, owner = shapely.get_parts(shapely.intersection(network.roads[candidates], buffer), return_index=True)
    is_line = (shapely.get_type_id(pieces) == 1) & (shapely.length(pieces) > 0)
    centerlines = list(pieces[is_line])
    half_widths = network.half_widths[candidates][owner[is_line]]

    coarse_noise, fine_noise = noise
    mask, transform = noisy_mask(polygon, seed, coarse_noise=coarse_noise, fine_noise=fine_noise)
    return Scene(
        fields={'radius_km': radius_km},
        label=f'{radius_km} km',
        polygon=polygon,
        mask=mask,
        transform=transform,
        centerlines=centerlines,
        true_dead_ends=true_dead_ends(centerlines, half_widths),
        holes=holes(polygon),
    )


def run(
    network: Network,
    center: tuple[float, float],
    radii: list[int],
    method_registry: MethodRegistry,
    repeat: int,
    seed: int,
    noise: tuple[float, float],
    time_limit: float,
) -> pd.DataFrame:
    results: list[Result] = []
    retired: set[tuple[str, str]] = set()
    summary = pd.DataFrame()

    with tqdm(total=2 * len(method_registry) * len(radii), unit='run') as progress:
        for radius_km in radii:
            started = time.perf_counter()
            scene = make_scene(network, center, radius_km, seed, noise)
            logger.info(
                f'Buffer {radius_km} km: {len(scene.centerlines):,} centerlines over '
                f'{sum(line.length for line in scene.centerlines) / 1000:,.0f} km, {scene.mask.size / 1e6:.1f} Mpx '
                f'mask, {scene.true_dead_ends:,} true dead ends, {scene.holes:,} holes, '
                f'prepared in {time.perf_counter() - started:.1f} s'
            )

            scene_results = benchmark_scene(
                scene, method_registry, repeat, progress, frozenset(retired), SINGLE_RUN_AFTER_S
            )
            for result in scene_results:
                if result['time'] > time_limit:
                    retired.add((result['method'], result['source']))
                    logger.warning(
                        f'{result["method"]} ({result["source"]}) took {result["time"]:,.0f} s on the {radius_km} km '
                        f'buffer, over the {time_limit:,.0f} s limit, so it is skipped on larger buffers'
                    )

            results.extend(scene_results)
            summary = write_table(results, ['radius_km', 'source', 'method'], TABLE_PATH)
            del scene
            release_memory()

    return summary


def log_ticks(low: float, high: float) -> list[float]:
    powers = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    decades = [10.0**power for power in powers if low <= 10.0**power <= high]
    if len(decades) >= 2:
        return decades

    return [step * 10.0**power for power in powers for step in (1, 2, 5) if low <= step * 10.0**power <= high]


def format_score(value: float) -> str:
    return f'{value:.2f}'


def format_count(value: float) -> str:
    return f'{value / 1000:,.0f}k' if value >= 1000 else f'{value:,.0f}'


def format_runtime(value_ms: float) -> str:
    return f'{value_ms / 1000:,.0f} s' if value_ms >= 1000 else f'{value_ms:,.0f} ms'


def spread(positions: list[float], gap: float) -> list[float]:
    order = np.argsort(positions)
    placed = np.asarray(positions, dtype=float)[order]
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)

    overflow = placed[-1] - max(positions) if len(placed) else 0.0
    placed -= max(0.0, overflow / 2)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)

    spread_positions = np.empty_like(placed)
    spread_positions[order] = placed
    return list(spread_positions)


def annotate_end(
    axes: Axes,
    text: str,
    anchor: tuple[float, float],
    placement: tuple[tuple[int, int], str, str],
    is_crestline: bool,
) -> Annotation:
    offset, horizontal, vertical = placement
    return axes.annotate(
        text,
        anchor,
        xycoords=blended_transform_factory(axes.transData, axes.transAxes),
        xytext=offset,
        textcoords='offset points',
        ha=horizontal,
        va=vertical,
        fontsize=8.5,
        color=INK_COLOR if is_crestline else SECONDARY_INK_COLOR,
        weight='semibold' if is_crestline else 'normal',
        annotation_clip=False,
    )


def label_line_ends(
    axes: Axes, ends: list[tuple[float, float, str, bool]], series_lines: list[Line2D], gap: float = 0.1
) -> None:
    renderer = axes.figure.canvas.get_renderer()
    to_axes = axes.transAxes.inverted()
    line_paths = [line.get_transform().transform_path(line.get_path()) for line in series_lines]
    last_end_x = max(end[0] for end in ends)
    placed_boxes = []

    def is_clear(box: Bbox) -> bool:
        crosses_line = any(path.intersects_bbox(box, filled=False) for path in line_paths)
        return not crosses_line and not any(box.overlaps(placed_box) for placed_box in placed_boxes)

    for end_x in sorted({end[0] for end in ends}, reverse=True):
        group = [end for end in ends if end[0] == end_x]
        heights = [to_axes.transform(axes.transData.transform((end_x, end_y)))[1] for _, end_y, _, _ in group]
        placements = LABEL_PLACEMENTS if end_x < last_end_x else LABEL_PLACEMENTS[:1]

        for (_, _, text, is_crestline), height in zip(group, spread(heights, gap)):
            candidates = [annotate_end(axes, text, (end_x, height), place, is_crestline) for place in placements]
            boxes = [candidate.get_window_extent(renderer).padded(2) for candidate in candidates]
            chosen = next((index for index, box in enumerate(boxes) if is_clear(box)), 0)

            for index, candidate in enumerate(candidates):
                if index != chosen:
                    candidate.remove()

            placed_boxes.append(boxes[chosen])


def draw_trend(
    axes: Axes,
    series: dict[str, tuple[list[int], list[float]]],
    radii: list[int],
    log_scale: bool,
    tick_format: Callable[[float], str],
    truth: tuple[list[int], list[float]] | None = None,
) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    values = [value for _, method_values in series.values() for value in method_values]
    if truth is not None:
        values += truth[1]

    if log_scale:
        axes.set_yscale('log')
        low, high = min(values) / 1.6, max(values) * 1.6
        ticks = log_ticks(low, high)
    else:
        low, high = np.floor(min(values) * 10) / 10, 1.0
        step = 0.05 if high - low <= 0.2 else 0.1
        ticks = list(np.round(np.arange(low, high + step / 2, step), 2))
        low, high = low - 0.015, high + 0.015

    axes.set_ylim(low, high)
    axes.set_xlim(radii[0] - 0.5, radii[-1] + 0.5)
    for tick in ticks:
        axes.axhline(tick, color=HAIRLINE_COLOR, lw=0.8, zorder=0)

    ends, series_lines = [], []
    if truth is not None:
        series_lines += axes.plot(*truth, color=MUTED_COLOR, lw=1.1, zorder=1, solid_capstyle='round')
        ends.append((truth[0][-1], truth[1][-1], 'true', False))

    for method_name in sorted(series, key=lambda name: name == 'crestline'):
        method_radii, method_values = series[method_name]
        short_name, marker = METHOD_STYLES.get(method_name, (method_name, 'o'))
        is_crestline = method_name == 'crestline'
        series_lines += axes.plot(
            method_radii,
            method_values,
            marker=marker,
            ms=6.5 if is_crestline else 5,
            lw=1.8 if is_crestline else 1.2,
            color=ACCENT_COLOR if is_crestline else NEUTRAL_COLOR,
            mec=SURFACE_COLOR,
            mew=1.1,
            zorder=3 if is_crestline else 2,
            solid_capstyle='round',
            solid_joinstyle='round',
        )
        ends.append((method_radii[-1], method_values[-1], short_name, is_crestline))

    label_line_ends(axes, ends, series_lines)

    axes.yaxis.set_major_locator(FixedLocator(ticks))
    axes.yaxis.set_minor_locator(NullLocator())
    axes.set_yticklabels([tick_format(tick) for tick in ticks])

    labelled_radii = [radius for radius in radii if radius == radii[0] or radius % 2 == 0 or len(radii) <= 6]
    axes.xaxis.set_major_locator(FixedLocator(labelled_radii))
    axes.xaxis.set_minor_locator(FixedLocator(radii))
    axes.set_xticklabels([f'{radius} km' if radius == radii[-1] else str(radius) for radius in labelled_radii])
    axes.tick_params(axis='both', which='both', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)

    for side in ('top', 'right', 'left'):
        axes.spines[side].set_visible(False)

    axes.spines['bottom'].set_color(AXIS_COLOR)


def figure(
    summary: pd.DataFrame,
    method_registry: MethodRegistry,
    network: Network,
    center: tuple[float, float],
    panel_radius_km: int,
    window: tuple[float, float, float, float],
    seed: int,
    noise: tuple[float, float],
    time_limit: float,
) -> None:
    logger.info(f'Drawing benchmark figure from the {panel_radius_km} km buffer')

    method_names = [method_name for method_name in method_registry if method_name != CUDA_METHOD]
    scene = make_scene(network, center, panel_radius_km, seed, noise)
    noisy_pieces = label(scene.mask, np.ones((3, 3)))[1]
    clean_pieces = label(burn_polygon(scene.polygon, scene.mask, scene.transform), np.ones((3, 3)))[1]
    logger.info(f'Noise splits the {panel_radius_km} km buffer from {clean_pieces:,} into {noisy_pieces:,} pieces')

    window_min_x, window_min_y, window_max_x, window_max_y = window
    figure_width, figure_height, left_margin, right_margin = 12.0, 7.6, 0.45, 0.25
    canvas = Canvas(figure_width, figure_height)

    draw_header(
        canvas,
        f'Noisy road mask, {window_max_x - window_min_x:.0f} × {window_max_y - window_min_y:.0f} m window',
        f'Scores in each title are for the whole {panel_radius_km} km buffer: {scene.true_dead_ends:,} true dead '
        f'ends, and the noise breaks it into {noisy_pieces:,} pieces.',
        left_margin,
        right_margin,
    )
    draw_panels(canvas, method_names, method_registry, scene, window, left_margin, right_margin)

    noisy = summary[summary['source'] == 'noisy mask'].sort_values('radius_km')
    radii = sorted(noisy['radius_km'].unique().tolist())
    truth_rows = noisy.drop_duplicates('radius_km')

    def series(column: str) -> dict[str, tuple[list[int], list[float]]]:
        trends = {}
        for method_name in method_names:
            rows = noisy[noisy['method'] == method_name]
            if len(rows):
                trends[method_name] = (rows['radius_km'].tolist(), rows[column].astype(float).tolist())

        return trends

    stopped_early = any(trend_radii[-1] < radii[-1] for trend_radii, _ in series('runtime_ms').values())
    mpx_low, mpx_high = truth_rows['mpx'].min(), truth_rows['mpx'].max()
    caption = (
        f'Noisy masks, buffers of {radii[0]} to {radii[-1]} km around the centre of Łódź '
        f'({mpx_low:,.0f} to {mpx_high:,.0f} Mpx)'
    )
    if stopped_early:
        limit = f'{time_limit / 60:g} min' if time_limit >= 60 else f'{time_limit:g} s'
        caption += f'; a line stops after the first buffer that took a method over {limit}'

    chart_specs = [
        ('F1 @ 5 m', 'higher is better', series('f1_5m'), False, format_score, None),
        (
            'Dead ends',
            'closer to the true count is better',
            series('dead_ends'),
            True,
            format_count,
            (truth_rows['radius_km'].tolist(), truth_rows['true_dead_ends'].astype(float).tolist()),
        ),
        ('Runtime', 'lower is better', series('runtime_ms'), True, format_runtime, None),
    ]

    charts_top, chart_height, tick_room, label_room, chart_gap = 5.45, 1.7, 0.5, 0.8, 0.45
    first_chart_left = left_margin + tick_room
    chart_width = (figure_width - first_chart_left - right_margin - 3 * label_room - 2 * (chart_gap + tick_room)) / 3
    canvas.text(left_margin, charts_top - 0.75, caption, fontsize=10)

    for chart_index, (chart_title, chart_hint, trends, log_scale, tick_format, truth) in enumerate(chart_specs):
        chart_left = first_chart_left + chart_index * (chart_width + label_room + chart_gap + tick_room)
        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        draw_trend(chart_axes, trends, radii, log_scale, tick_format, truth)

        canvas.text(
            chart_left - tick_room, charts_top - 0.38, chart_title, fontsize=10.5, color=INK_COLOR, weight='semibold'
        )
        canvas.text(chart_left - tick_room, charts_top - 0.16, chart_hint, fontsize=8.5, color=MUTED_COLOR)

    canvas.save(FIGURE_PATH, 'benchmark figure')


def main(
    repeat: int = 3,
    seed: int = 0,
    coarse_noise: float = 2.5,
    fine_noise: float = 0.8,
    time_limit: float = 600.0,
    panel_radius: int = 1,
    window_size: float = 400.0,
    window_offset: tuple[float, float] = (600.0, 350.0),
) -> None:
    """Run the Łódź road network benchmark over growing buffers and write the summary table and figure to ``output/``.

    :param repeat: Timed runs per method and input after one warm-up run; the fastest one is kept
    :param seed: Seed of the boundary noise added to the noisy masks
    :param coarse_noise: Standard deviation in metres of the smooth boundary noise, as in the synthetic benchmark
    :param fine_noise: Standard deviation in metres of the fine boundary noise, as in the synthetic benchmark
    :param time_limit: Seconds; a method and input whose best run takes longer is skipped on larger buffers
    :param panel_radius: Buffer radius in km of the scene drawn in the figure's map panels
    :param window_size: Side in metres of the map panels' window
    :param window_offset: Offset in metres of the map panels' window centre from ``center``
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    configure_logging('INFO')
    noise = (coarse_noise, fine_noise)

    radii = list(range(1, 6))

    radii = sorted(radii)
    network = load_network()
    center_point = shapely.centroid(MultiLineString(list(network.roads)))
    center = (center_point.x, center_point.y)

    window_center_x, window_center_y = center[0] + window_offset[0], center[1] + window_offset[1]
    half_window = window_size / 2
    window = (
        window_center_x - half_window,
        window_center_y - half_window,
        window_center_x + half_window,
        window_center_y + half_window,
    )

    gpu_name, cuda_failure_reason = cuda_device()
    if gpu_name:
        logger.info(f'CUDA: {gpu_name}')
    else:
        logger.info(f'CUDA: off ({cuda_failure_reason}), running on the CPU only')

    method_registry = methods(bool(gpu_name))
    logger.info(f'Methods: {", ".join(method_registry)}')

    logger.info(
        f'Plan: buffers of {", ".join(str(radius) for radius in radii)} km around '
        f'({center[0]:,.0f}, {center[1]:,.0f}), {repeat} timed repeats per run after a warm-up, '
        f'{time_limit:,.0f} s limit per run'
    )
    started = time.perf_counter()
    with logging_redirect_tqdm():
        summary = run(network, center, radii, method_registry, repeat, seed, noise, time_limit)

    logger.info(f'Benchmark finished in {time.perf_counter() - started:,.1f} s with {len(summary)} summary rows')

    figure(summary, method_registry, network, center, panel_radius, window, seed, noise, time_limit)


if __name__ == '__main__':
    CLI(main)
