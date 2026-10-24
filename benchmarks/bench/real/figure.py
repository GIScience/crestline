from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from bench.environment import OUTPUT_DIR
from bench.environment import logger
from bench.methods import CUDA_METHOD
from bench.methods import MethodRegistry
from bench.plotting import FIGURE_WIDTH
from bench.plotting import LEFT_MARGIN
from bench.plotting import RIGHT_MARGIN
from bench.plotting import Canvas
from bench.plotting import draw_chart_heading
from bench.plotting import draw_header
from bench.plotting import draw_panels
from bench.plotting import map_window_title
from bench.rasters import burn_polygon
from bench.real.charts import Trend
from bench.real.charts import Trends
from bench.real.charts import draw_trend
from bench.real.charts import format_count
from bench.real.charts import format_runtime
from bench.real.charts import format_score
from bench.real.network import Coordinate
from bench.real.network import Network
from bench.real.network import Noise
from bench.real.network import make_scene
from bench.scene import Window
from numpy.typing import NDArray
from scipy.ndimage import label

FIGURE_PATH = OUTPUT_DIR / 'real_benchmark.png'
FIGURE_HEIGHT = 7.6


class TrendChart(NamedTuple):
    title: str
    hint: str
    trends: Trends
    log_scale: bool
    tick_format: Callable[[float], str]
    truth: Trend | None = None


def _count_pieces(mask: NDArray[np.bool_]) -> int:
    return label(mask, np.ones((3, 3)))[1]


def _trends(noisy: pd.DataFrame, method_names: list[str], column: str) -> Trends:
    trends: Trends = {}
    for method_name in method_names:
        rows = noisy[noisy['method'] == method_name]
        if len(rows):
            trends[method_name] = (rows['radius_km'].tolist(), rows[column].astype(float).tolist())

    return trends


def _caption(radii: list[int], truth_rows: pd.DataFrame, runtime_trends: Trends, time_limit: float) -> str:
    mpx_low, mpx_high = truth_rows['mpx'].min(), truth_rows['mpx'].max()
    caption = (
        f'Noisy masks, buffers of {radii[0]} to {radii[-1]} km around the centre of Łódź '
        f'({mpx_low:,.0f} to {mpx_high:,.0f} Mpx)'
    )

    stopped_early = any(trend_radii[-1] < radii[-1] for trend_radii, _ in runtime_trends.values())
    if stopped_early:
        limit = f'{time_limit / 60:g} min' if time_limit >= 60 else f'{time_limit:g} s'
        caption += f'; a line stops after the first buffer that took a method over {limit}'

    return caption


def draw_trend_charts(canvas: Canvas, summary: pd.DataFrame, method_names: list[str], time_limit: float) -> None:
    noisy = summary[summary['source'] == 'noisy mask'].sort_values('radius_km')
    radii = sorted(noisy['radius_km'].unique().tolist())
    truth_rows = noisy.drop_duplicates('radius_km')
    runtime_trends = _trends(noisy, method_names, 'runtime_ms')

    charts = [
        TrendChart('F1 @ 5 m', 'higher is better', _trends(noisy, method_names, 'f1_5m'), False, format_score),
        TrendChart(
            'Dead ends',
            'closer to the true count is better',
            _trends(noisy, method_names, 'dead_ends'),
            True,
            format_count,
            (truth_rows['radius_km'].tolist(), truth_rows['true_dead_ends'].astype(float).tolist()),
        ),
        TrendChart('Runtime', 'lower is better', runtime_trends, True, format_runtime),
    ]

    charts_top, chart_height, tick_room, label_room, chart_gap = 5.45, 1.7, 0.5, 0.8, 0.45
    first_chart_left = LEFT_MARGIN + tick_room
    chart_width = (canvas.width - first_chart_left - RIGHT_MARGIN - 3 * label_room - 2 * (chart_gap + tick_room)) / 3
    canvas.text(LEFT_MARGIN, charts_top - 0.75, _caption(radii, truth_rows, runtime_trends, time_limit), fontsize=10)

    for chart_index, chart in enumerate(charts):
        chart_left = first_chart_left + chart_index * (chart_width + label_room + chart_gap + tick_room)
        chart_axes = canvas.axes(chart_left, charts_top, chart_width, chart_height)
        draw_trend(chart_axes, chart.trends, radii, chart.log_scale, chart.tick_format, chart.truth)
        draw_chart_heading(canvas, chart_left - tick_room, charts_top, chart.title, chart.hint)


def benchmark_figure(
    summary: pd.DataFrame,
    method_registry: MethodRegistry,
    network: Network,
    center: Coordinate,
    panel_radius_km: int,
    window: Window,
    seed: int,
    noise: Noise,
    time_limit: float,
) -> None:
    logger.info(f'Drawing benchmark figure from the {panel_radius_km} km buffer')

    method_names = [method_name for method_name in method_registry if method_name != CUDA_METHOD]
    scene = make_scene(network, center, panel_radius_km, seed, noise)
    noisy_pieces = _count_pieces(scene.mask)
    clean_pieces = _count_pieces(burn_polygon(scene.polygon, scene.mask, scene.transform))
    logger.info(f'Noise splits the {panel_radius_km} km buffer from {clean_pieces:,} into {noisy_pieces:,} pieces')

    canvas = Canvas(FIGURE_WIDTH, FIGURE_HEIGHT)

    draw_header(
        canvas,
        map_window_title(window),
        f'Scores in each title are for the whole {panel_radius_km} km buffer: {scene.true_dead_ends:,} true dead '
        f'ends, and the noise breaks it into {noisy_pieces:,} pieces.',
        LEFT_MARGIN,
        RIGHT_MARGIN,
    )
    draw_panels(canvas, method_names, method_registry, scene, window, LEFT_MARGIN, RIGHT_MARGIN)
    draw_trend_charts(canvas, summary, method_names, time_limit)

    canvas.save(FIGURE_PATH, 'benchmark figure')
