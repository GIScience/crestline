import os
import time
from typing import TypeAlias

import numpy as np
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
from common import configure_logging
from common import cuda_device
from common import draw_header
from common import draw_panels
from common import holes
from common import logger
from common import methods
from common import noisy_mask
from common import true_dead_ends
from common import write_table
from jsonargparse import CLI
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from shapely import LineString
from shapely import MultiLineString
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

Case: TypeAlias = tuple[list[LineString], list[float], Shape]

ROAD_HALF_WIDTH_M = 15.0
"""Buffer radius of every road in the uniform width road cases, which makes each road 30 m wide."""

MIXED_HALF_WIDTHS_M = [6.0, 9.0, 14.0, 20.0, 30.0]
"""Buffer radii the mixed case picks from at random, one per road, giving roads 12 to 60 m wide."""


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


def make_scene(case_kind: str, seed: int) -> Scene:
    centerlines, half_widths, polygon = make_case(case_kind, seed)
    mask, transform = noisy_mask(polygon, seed)
    return Scene(
        fields={'case': case_kind, 'seed': seed},
        label=f'{case_kind} seed {seed}',
        polygon=polygon,
        mask=mask,
        transform=transform,
        centerlines=centerlines,
        true_dead_ends=true_dead_ends(centerlines, half_widths),
        holes=holes(polygon),
    )


def run(plan: list[tuple[str, int]], method_registry: MethodRegistry, repeat: int) -> list[Result]:
    results = []

    with tqdm(total=2 * len(method_registry) * len(plan), unit='run') as progress:
        for case_kind, seed in plan:
            scene = make_scene(case_kind, seed)
            logger.info(
                f'Case {case_kind} seed {seed}: {len(scene.centerlines)} centerlines, '
                f'{scene.mask.size / 1e6:.2f} Mpx mask, {scene.true_dead_ends} true dead ends, {scene.holes} holes'
            )
            results.extend(benchmark_scene(scene, method_registry, repeat, progress))

    return results


def figure(
    results: list[Result],
    method_registry: MethodRegistry,
    case_kind: str = 'roads',
    seed: int = 1,
    window: tuple[float, float, float, float] = (470, 270, 870, 670),
) -> None:
    logger.info(f'Drawing benchmark figure from {case_kind} seed {seed}')

    method_names = [method_name for method_name in method_registry if method_name != CUDA_METHOD]
    scene = make_scene(case_kind, seed)

    window_min_x, window_min_y, window_max_x, window_max_y = window
    figure_width, figure_height, left_margin, right_margin = 12.0, 7.5, 0.45, 0.25
    canvas = Canvas(figure_width, figure_height)

    draw_header(
        canvas,
        f'Noisy road mask, {window_max_x - window_min_x:.0f} × {window_max_y - window_min_y:.0f} m window',
        f'Scores in each title are for the whole network; it has {scene.true_dead_ends} true dead ends.',
        left_margin,
        right_margin,
    )
    draw_panels(canvas, method_names, method_registry, scene, window, left_margin, right_margin)

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
    canvas.text(
        left_margin,
        charts_top - 0.75,
        'Noisy masks, means over the 1 km road networks, mixed-width road networks and rivers',
        fontsize=10,
    )

    row_positions = np.arange(len(method_names))[::-1]
    for chart_index, (chart_title, chart_hint, chart_values, use_log_scale, value_format) in enumerate(chart_specs):
        chart_left = left_margin + label_column_width + chart_index * (chart_width + chart_gap)
        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        chart_axes.set_facecolor(SURFACE_COLOR)

        x_min, x_max = (min(chart_values) / 2.2, max(chart_values) * 4.5) if use_log_scale else (0.4, 1.04)
        if use_log_scale:
            chart_axes.set_xscale('log')

        chart_axes.set_xlim(x_min, x_max)
        chart_axes.set_ylim(-0.55, len(method_names) - 0.45)
        for row_y in row_positions:
            chart_axes.axhline(row_y, color=HAIRLINE_COLOR, lw=0.8, zorder=0)

        if chart_title == 'Dead ends':
            chart_axes.axvline(mean_true_dead_ends, color=MUTED_COLOR, lw=0.9, zorder=1)

        for row_y, value, method_name in zip(row_positions, chart_values, method_names):
            is_crestline = method_name == 'crestline'
            chart_axes.plot(
                [value],
                [row_y],
                'o',
                ms=8.5,
                color=ACCENT_COLOR if is_crestline else NEUTRAL_COLOR,
                mec=SURFACE_COLOR,
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
                color=INK_COLOR if is_crestline else SECONDARY_INK_COLOR,
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
                    color=INK_COLOR if method_name == 'crestline' else SECONDARY_INK_COLOR,
                )

        chart_axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)
        if use_log_scale:
            tick_values = [tick for tick in (1, 10, 100, 1000, 10000) if x_min <= tick <= x_max]
            chart_axes.xaxis.set_major_locator(FixedLocator(tick_values))
            chart_axes.xaxis.set_minor_locator(NullLocator())
            chart_axes.set_xticklabels([f'{tick:,}' for tick in tick_values])
        else:
            chart_axes.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

        for side in ('top', 'right', 'left'):
            chart_axes.spines[side].set_visible(False)

        chart_axes.spines['bottom'].set_color(AXIS_COLOR)
        canvas.text(chart_left, charts_top - 0.38, chart_title, fontsize=10.5, color=INK_COLOR, weight='semibold')
        canvas.text(chart_left, charts_top - 0.16, chart_hint, fontsize=8.5, color=MUTED_COLOR)

    canvas.save(OUTPUT_DIR / 'synthetic_benchmark.png', 'benchmark figure')


def cuda_benchmark_figure(results: list[Result], gpu_name: str) -> None:
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
    canvas = Canvas(figure_width, figure_height)

    if all(cuda_matches):
        match_summary = 'same lines in all'
    else:
        match_summary = f'different lines in {cuda_matches.count(False)} of'

    canvas.text(left_margin, 0.5, 'crestline on the CPU and on CUDA', fontsize=14, color=INK_COLOR, weight='semibold')
    canvas.text(
        left_margin,
        0.78,
        f'CPU with {os.cpu_count()} cores · {gpu_name} · best runtime per shape, '
        f'averaged over seeds · {match_summary} {len(cuda_matches)} runs',
        fontsize=10,
    )

    for legend_index, (legend_label, legend_color) in enumerate((('CUDA', ACCENT_COLOR), ('CPU', NEUTRAL_COLOR))):
        legend_right = figure_width - right_margin - legend_index * 0.95
        canvas.text(legend_right, 0.78, legend_label, fontsize=9, ha='right')
        canvas.mark([legend_right - 0.5], 0.745, marker='o', ms=7, color=legend_color, mec=SURFACE_COLOR, mew=1.4)

    row_positions = np.arange(len(ordered_cases))[::-1]
    for chart_index, source in enumerate(('polygon', 'noisy mask')):
        cpu_runtimes = [mean_runtime_ms(case_kind, source, 'crestline') for case_kind in ordered_cases]
        gpu_runtimes = [mean_runtime_ms(case_kind, source, CUDA_METHOD) for case_kind in ordered_cases]
        chart_left = left_margin + label_column_width + chart_index * (chart_width + chart_gap)

        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        chart_axes.set_facecolor(SURFACE_COLOR)
        chart_axes.set_xscale('log')
        x_min, x_max = min(cpu_runtimes + gpu_runtimes) / 2.5, max(cpu_runtimes + gpu_runtimes) * 7
        chart_axes.set_xlim(x_min, x_max)
        chart_axes.set_ylim(-0.55, len(ordered_cases) - 0.45)

        for row_y, cpu_runtime_ms, gpu_runtime_ms in zip(row_positions, cpu_runtimes, gpu_runtimes):
            chart_axes.axhline(row_y, color=HAIRLINE_COLOR, lw=0.8, zorder=0)
            chart_axes.plot(
                [cpu_runtime_ms, gpu_runtime_ms],
                [row_y, row_y],
                color=AXIS_COLOR,
                lw=2,
                solid_capstyle='round',
                zorder=1,
            )
            chart_axes.plot(
                [cpu_runtime_ms], [row_y], 'o', ms=8.5, color=NEUTRAL_COLOR, mec=SURFACE_COLOR, mew=1.8, zorder=3
            )
            chart_axes.plot(
                [gpu_runtime_ms], [row_y], 'o', ms=8.5, color=ACCENT_COLOR, mec=SURFACE_COLOR, mew=1.8, zorder=3
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
                color=INK_COLOR,
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
        chart_axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)

        for side in ('top', 'right', 'left'):
            chart_axes.spines[side].set_visible(False)

        chart_axes.spines['bottom'].set_color(AXIS_COLOR)
        canvas.text(
            chart_left, charts_top - 0.22, f'Runtime, {source}', fontsize=10.5, color=INK_COLOR, weight='semibold'
        )

    canvas.save(OUTPUT_DIR / 'synthetic_benchmark_cuda.png', 'CUDA comparison figure')


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
    :param repeat: Timed runs per method and input after one warm-up run; the fastest one is kept
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    configure_logging(log_level)

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

    write_table(results, ['case', 'source', 'method'], OUTPUT_DIR / 'synthetic_benchmark_results.csv')

    if 'roads' in cases:
        figure(results, method_registry)

    if cuda_available:
        cuda_benchmark_figure(results, gpu_name)


if __name__ == '__main__':
    CLI(main)
