import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from bench.environment import OUTPUT_DIR
from bench.environment import logger
from bench.methods import CUDA_METHOD
from bench.methods import methods
from bench.plotting import ACCENT_COLOR
from bench.plotting import AXIS_COLOR
from bench.plotting import FIGURE_WIDTH
from bench.plotting import HAIRLINE_COLOR
from bench.plotting import INK_COLOR
from bench.plotting import LEFT_MARGIN
from bench.plotting import MUTED_COLOR
from bench.plotting import NEUTRAL_COLOR
from bench.plotting import RIGHT_MARGIN
from bench.plotting import SECONDARY_INK_COLOR
from bench.plotting import SURFACE_COLOR
from bench.plotting import Canvas
from bench.plotting import draw_chart_heading
from bench.plotting import draw_map_panels
from bench.plotting import fixed_ticks
from bench.plotting import log_ticks
from bench.plotting import style_spines
from bench.scene import Window
from bench.synthetic.cases import make_scene
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
from numpy.typing import NDArray

BENCHMARK_FIGURE_PATH = OUTPUT_DIR / 'synthetic_benchmark.png'
CUDA_FIGURE_PATH = OUTPUT_DIR / 'synthetic_benchmark_cuda.png'
BENCHMARK_FIGURE_HEIGHT = 7.5
PANEL_WINDOW: Window = (470, 270, 870, 670)
SUMMARY_CASES = ('roads', 'mixed', 'river')
CASE_LABELS: dict[str, str] = {
    'river': 'rivers',
    'roads': 'road networks',
    'mixed': 'mixed-width roads',
    'roads-large': '4 km road network',
}


class DotChart(NamedTuple):
    title: str
    hint: str
    values: list[float]
    log_scale: bool
    value_format: str
    truth: float | None = None


def _speed_label(cpu_runtime_ms: float, gpu_runtime_ms: float) -> str:
    if max(cpu_runtime_ms, gpu_runtime_ms) < 1.05 * min(cpu_runtime_ms, gpu_runtime_ms):
        return 'same'

    if cpu_runtime_ms >= gpu_runtime_ms:
        return f'{cpu_runtime_ms / gpu_runtime_ms:.1f}× faster'

    return f'{gpu_runtime_ms / cpu_runtime_ms:.1f}× slower'


def _draw_row_labels(axes: Axes, row_positions: NDArray[np.int64], labels: list[str]) -> None:
    for row_y, row_label in zip(row_positions, labels):
        axes.text(
            -0.04,
            row_y,
            row_label,
            transform=axes.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=9.5,
            color=INK_COLOR if row_label == 'crestline' else SECONDARY_INK_COLOR,
        )


def _noisy_means(runs: pd.DataFrame, cases: tuple[str, ...], method_names: list[str]) -> pd.DataFrame:
    noisy = runs[(runs['source'] == 'noisy mask') & runs['case'].isin(cases)]
    return noisy.groupby('method')[['f5', 'dead', 'true_dead', 'time']].mean().loc[method_names]


def _summary_charts(runs: pd.DataFrame, method_names: list[str]) -> list[DotChart]:
    means = _noisy_means(runs, SUMMARY_CASES, method_names)
    road_means = _noisy_means(runs, ('roads',), method_names)
    true_dead_ends = means['true_dead'].iloc[0]

    return [
        DotChart('F1 @ 5 m', 'higher is better', means['f5'].tolist(), False, '{:.3f}'),
        DotChart(
            'Dead ends',
            f'closer to the true {true_dead_ends:.1f} is better',
            means['dead'].tolist(),
            True,
            '{:,.0f}',
            true_dead_ends,
        ),
        DotChart('Runtime, 1.2 Mpx mask', 'lower is better', (1000 * road_means['time']).tolist(), True, '{:,.0f} ms'),
    ]


def _draw_dot_chart(axes: Axes, chart: DotChart, method_names: list[str], row_positions: NDArray[np.int64]) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    x_min, x_max = (min(chart.values) / 2.2, max(chart.values) * 4.5) if chart.log_scale else (0.4, 1.04)
    if chart.log_scale:
        axes.set_xscale('log')

    axes.set_xlim(x_min, x_max)
    axes.set_ylim(-0.55, len(method_names) - 0.45)
    for row_y in row_positions:
        axes.axhline(row_y, color=HAIRLINE_COLOR, lw=0.8, zorder=0)

    if chart.truth is not None:
        axes.axvline(chart.truth, color=MUTED_COLOR, lw=0.9, zorder=1)

    for row_y, value, method_name in zip(row_positions, chart.values, method_names):
        is_crestline = method_name == 'crestline'
        is_below_truth = chart.truth is not None and value < chart.truth
        axes.plot(
            [value],
            [row_y],
            'o',
            ms=8.5,
            color=ACCENT_COLOR if is_crestline else NEUTRAL_COLOR,
            mec=SURFACE_COLOR,
            mew=1.8,
            zorder=3,
        )

        if chart.log_scale:
            label_x = value / 1.2 if is_below_truth else value * 1.2
        else:
            label_x = value + 0.02

        axes.text(
            label_x,
            row_y,
            chart.value_format.format(value),
            va='center',
            ha='right' if is_below_truth else 'left',
            fontsize=9,
            color=INK_COLOR if is_crestline else SECONDARY_INK_COLOR,
            weight='semibold' if is_crestline else 'normal',
        )

    axes.set_yticks([])
    axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)
    if chart.log_scale:
        ticks = log_ticks(x_min, x_max)
        fixed_ticks(axes.xaxis, ticks, [f'{tick:,g}' for tick in ticks])
    else:
        axes.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

    style_spines(axes)


def benchmark_figure(
    runs: pd.DataFrame,
    case_kind: str = 'roads',
    seed: int = 1,
    window: Window = PANEL_WINDOW,
) -> None:
    logger.info(f'Drawing benchmark figure from {case_kind} seed {seed}')

    cpu_methods = methods()
    method_names = list(cpu_methods)
    scene = make_scene(case_kind, seed)

    canvas = Canvas(FIGURE_WIDTH, BENCHMARK_FIGURE_HEIGHT)
    draw_map_panels(
        canvas,
        cpu_methods,
        scene,
        window,
        f'Scores in each title are for the whole network; it has {scene.true_dead_ends} true dead ends.',
    )

    charts_top, label_column_width, chart_gap, chart_height = 5.45, 1.45, 0.4, 1.6
    chart_width = (canvas.width - LEFT_MARGIN - label_column_width - RIGHT_MARGIN - 2 * chart_gap) / 3
    canvas.text(
        LEFT_MARGIN,
        charts_top - 0.75,
        'Noisy masks, means over the 1 km road networks, mixed-width road networks and rivers',
        fontsize=10,
    )

    row_positions = np.arange(len(method_names))[::-1]
    for chart_index, chart in enumerate(_summary_charts(runs, method_names)):
        chart_left = LEFT_MARGIN + label_column_width + chart_index * (chart_width + chart_gap)
        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        _draw_dot_chart(chart_axes, chart, method_names, row_positions)
        if chart_index == 0:
            _draw_row_labels(chart_axes, row_positions, method_names)

        draw_chart_heading(canvas, chart_left, charts_top, chart.title, chart.hint)

    canvas.save(BENCHMARK_FIGURE_PATH, 'benchmark figure')


def _draw_cuda_header(canvas: Canvas, gpu_name: str, cuda_matches: list[bool]) -> None:
    if all(cuda_matches):
        match_summary = 'same lines in all'
    else:
        match_summary = f'different lines in {cuda_matches.count(False)} of'

    canvas.text(LEFT_MARGIN, 0.5, 'crestline on the CPU and on CUDA', fontsize=14, color=INK_COLOR, weight='semibold')
    canvas.text(
        LEFT_MARGIN,
        0.78,
        f'CPU with {os.cpu_count()} cores · {gpu_name} · best runtime per shape, '
        f'averaged over seeds · {match_summary} {len(cuda_matches)} runs',
        fontsize=10,
    )

    for legend_index, (legend_label, legend_color) in enumerate((('CUDA', ACCENT_COLOR), ('CPU', NEUTRAL_COLOR))):
        legend_right = canvas.width - RIGHT_MARGIN - legend_index * 0.95
        canvas.text(legend_right, 0.78, legend_label, fontsize=9, ha='right')
        canvas.mark([legend_right - 0.5], 0.745, marker='o', ms=7, color=legend_color, mec=SURFACE_COLOR, mew=1.4)


def _draw_runtime_chart(
    axes: Axes, cpu_runtimes: list[float], gpu_runtimes: list[float], row_positions: NDArray[np.int64]
) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    axes.set_xscale('log')
    x_min, x_max = min(cpu_runtimes + gpu_runtimes) / 2.5, max(cpu_runtimes + gpu_runtimes) * 7
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(-0.55, len(row_positions) - 0.45)

    for row_y, cpu_runtime_ms, gpu_runtime_ms in zip(row_positions, cpu_runtimes, gpu_runtimes):
        axes.axhline(row_y, color=HAIRLINE_COLOR, lw=0.8, zorder=0)
        axes.plot(
            [cpu_runtime_ms, gpu_runtime_ms],
            [row_y, row_y],
            color=AXIS_COLOR,
            lw=2,
            solid_capstyle='round',
            zorder=1,
        )
        axes.plot([cpu_runtime_ms], [row_y], 'o', ms=8.5, color=NEUTRAL_COLOR, mec=SURFACE_COLOR, mew=1.8, zorder=3)
        axes.plot([gpu_runtime_ms], [row_y], 'o', ms=8.5, color=ACCENT_COLOR, mec=SURFACE_COLOR, mew=1.8, zorder=3)
        axes.text(
            max(cpu_runtime_ms, gpu_runtime_ms) * 1.25,
            row_y,
            f'{gpu_runtime_ms:,.0f} ms · {_speed_label(cpu_runtime_ms, gpu_runtime_ms)}',
            va='center',
            fontsize=9,
            color=INK_COLOR,
        )

    axes.set_yticks([])
    ticks = log_ticks(x_min, x_max)
    fixed_ticks(axes.xaxis, ticks, [f'{tick:,g} ms' for tick in ticks])
    axes.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)
    style_spines(axes)


def cuda_benchmark_figure(runs: pd.DataFrame, gpu_name: str) -> None:
    cuda_runs = runs[runs['method'] == CUDA_METHOD]
    mean_mpx = cuda_runs.groupby('case')['mpx'].mean().sort_values()
    runtime_ms = 1000 * runs.groupby(['case', 'source', 'method'])['time'].mean()
    cases = mean_mpx.index.tolist()

    label_column_width, chart_gap, charts_top = 2.1, 0.5, 1.5
    chart_height = 0.42 * len(cases)
    chart_width = (FIGURE_WIDTH - LEFT_MARGIN - label_column_width - RIGHT_MARGIN - chart_gap) / 2
    canvas = Canvas(FIGURE_WIDTH, 2.0 + chart_height)
    _draw_cuda_header(canvas, gpu_name, cuda_runs['same'].tolist())

    row_positions = np.arange(len(cases))[::-1]
    row_labels = [f'{CASE_LABELS.get(case_kind, case_kind)} ({mpx:.1f} Mpx)' for case_kind, mpx in mean_mpx.items()]
    for chart_index, source in enumerate(('polygon', 'noisy mask')):
        cpu_runtimes = [runtime_ms[case_kind, source, 'crestline'] for case_kind in cases]
        gpu_runtimes = [runtime_ms[case_kind, source, CUDA_METHOD] for case_kind in cases]
        chart_left = LEFT_MARGIN + label_column_width + chart_index * (chart_width + chart_gap)

        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        _draw_runtime_chart(chart_axes, cpu_runtimes, gpu_runtimes, row_positions)
        if chart_index == 0:
            _draw_row_labels(chart_axes, row_positions, row_labels)

        canvas.text(
            chart_left, charts_top - 0.22, f'Runtime, {source}', fontsize=10.5, color=INK_COLOR, weight='semibold'
        )

    canvas.save(CUDA_FIGURE_PATH, 'CUDA comparison figure')
