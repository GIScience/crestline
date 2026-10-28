from typing import NamedTuple

import numpy as np
import pandas as pd
from bench.environment import OUTPUT_DIR
from bench.plotting import ACCENT_COLOR
from bench.plotting import DEAD_END_COLOR
from bench.plotting import HAIRLINE_COLOR
from bench.plotting import INK_COLOR
from bench.plotting import MUTED_COLOR
from bench.plotting import NEUTRAL_COLOR
from bench.plotting import SECONDARY_INK_COLOR
from bench.plotting import SURFACE_COLOR
from bench.plotting import Canvas
from bench.plotting import draw_chart_heading
from bench.plotting import style_spines
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator

ABLATION_FIGURE_PATH = OUTPUT_DIR / 'ablation_benchmark.png'
SOURCES = {'noisy mask': 'Noisy masks at 1 m', 'polygon': 'Polygons rasterized at 0.5 m'}


class Metric(NamedTuple):
    key: str
    title: str
    hint: str
    log_scale: bool
    scale: float
    value_format: str


METRICS = [
    Metric('f5', 'F1 @ 5 m', 'higher is better', False, 1, '{:.3f}'),
    Metric('f2', 'F1 @ 2 m', 'higher is better', False, 1, '{:.3f}'),
    Metric('offset', 'Offset', 'm, lower is better', False, 1, '{:.2f}'),
    Metric('dead_end_error', 'Dead-end error', 'vs truth, lower is better', True, 1, '{:,.1f}'),
    Metric('loop_error', 'Loop error', 'vs truth, lower is better', True, 1, '{:,.1f}'),
    Metric('runtime_ms', 'Runtime', 'ms, lower is better', True, 1, '{:,.0f}'),
]


def _dot_color(variant: str, verdict: str) -> tuple[str, str]:
    if variant == 'baseline':
        return ACCENT_COLOR, ACCENT_COLOR

    if verdict == 'worse':
        return DEAD_END_COLOR, DEAD_END_COLOR

    if verdict == 'better':
        return SURFACE_COLOR, INK_COLOR

    return NEUTRAL_COLOR, NEUTRAL_COLOR


def _limits(values: np.ndarray, metric: Metric) -> tuple[float, float]:
    if metric.log_scale:
        positive = values[values > 0]
        low = positive.min() / 2.5 if len(positive) else 0.1
        return low, values.max() * 6

    if metric.key in ('f5', 'f2'):
        return max(0.0, np.floor(10 * values.min()) / 10 - 0.02), 1.0

    return 0.0, values.max() * 1.45


def _log_ticks(low: float, high: float) -> list[float]:
    ticks = [10.0**power for power in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)]
    ticks = [tick for tick in ticks if low <= tick <= high]
    return ticks[::2] if len(ticks) > 4 else ticks


def draw_metric(axes: Axes, rows: pd.DataFrame, metric: Metric, positions: np.ndarray) -> None:
    values = rows['ablated'].to_numpy(float) * metric.scale
    low, high = _limits(values, metric)

    axes.set_facecolor(SURFACE_COLOR)
    if metric.log_scale:
        axes.set_xscale('log')

    axes.set_xlim(low, high)
    axes.set_ylim(-0.6, len(rows) - 0.4)
    for row_y in positions:
        axes.axhline(row_y, color=HAIRLINE_COLOR, lw=0.7, zorder=0)

    full_value = rows.loc[rows['variant'] == 'baseline', 'ablated'].iloc[0] * metric.scale
    axes.axvline(max(full_value, low), color=ACCENT_COLOR, lw=0.8, alpha=0.35, zorder=1)

    for row_y, value, variant, verdict in zip(positions, values, rows['variant'], rows['verdict']):
        face, edge = _dot_color(variant, verdict)
        shown = max(value, low)
        axes.plot([shown], [row_y], 'o', ms=6.5, color=face, mec=edge, mew=1.2, zorder=3)
        offset = shown * 1.25 if metric.log_scale else shown + (high - low) * 0.03
        axes.text(
            offset,
            row_y,
            metric.value_format.format(value),
            va='center',
            ha='left',
            fontsize=7.5,
            color=INK_COLOR if variant == 'baseline' or verdict == 'worse' else SECONDARY_INK_COLOR,
            weight='semibold' if variant == 'baseline' else 'normal',
        )

    axes.set_yticks([])
    axes.tick_params(axis='x', length=0, labelsize=7.5, labelcolor=MUTED_COLOR, pad=4)
    if metric.log_scale:
        ticks = _log_ticks(low, high)
        axes.xaxis.set_major_locator(FixedLocator(ticks))
        axes.xaxis.set_minor_locator(NullLocator())
        axes.set_xticklabels([f'{tick:g}' for tick in ticks])

    style_spines(axes)


LEGEND = [
    ('baseline', ACCENT_COLOR, ACCENT_COLOR),
    ('significantly worse', DEAD_END_COLOR, DEAD_END_COLOR),
    ('significantly better', SURFACE_COLOR, INK_COLOR),
    ('no significant change', NEUTRAL_COLOR, NEUTRAL_COLOR),
]


def _draw_legend(canvas: Canvas, right: float, top: float) -> None:
    cursor = right
    for label, face, edge in reversed(LEGEND):
        canvas.text(cursor, top, label, fontsize=8.5, ha='right')
        cursor -= 0.058 * len(label) + 0.14
        canvas.mark([cursor], top - 0.035, marker='o', ms=6, color=face, mec=edge, mew=1.2)
        cursor -= 0.36


def ablation_figure(table: pd.DataFrame) -> None:
    variants = list(dict.fromkeys(table['variant']))
    variants = ['baseline'] + [variant for variant in variants if variant != 'baseline']
    scenes = int(table['scenes'].max())

    left, right, label_width, gap = 0.35, 0.25, 2.25, 0.22
    row_height, section_gap = 0.235, 1.15
    section_height = row_height * len(variants)
    canvas = Canvas(15.0, 1.45 + 2 * (section_height + section_gap) + 0.2)
    chart_width = (canvas.width - left - right - label_width - gap * (len(METRICS) - 1)) / len(METRICS)

    canvas.text(
        left,
        0.48,
        'Ablation: each stage of crestline switched off',
        fontsize=14,
        color=INK_COLOR,
        weight='semibold',
    )
    canvas.text(
        left,
        0.76,
        f'Means over {scenes} synthetic scenes (road networks, mixed-width networks, rivers). '
        'Colour: paired Wilcoxon signed-rank test against the full method, Holm-corrected p < 0.05.',
        fontsize=9.5,
    )
    _draw_legend(canvas, canvas.width - right, 1.08)

    positions = np.arange(len(variants))[::-1]
    for section_index, (source, source_title) in enumerate(SOURCES.items()):
        top = 1.45 + section_index * (section_height + section_gap) + 0.75
        canvas.text(left, top - 0.62, source_title, fontsize=11, color=INK_COLOR, weight='semibold')

        for metric_index, metric in enumerate(METRICS):
            rows = table[(table['source'] == source) & (table['metric'] == metric.key)].set_index('variant')
            rows = rows.reindex(variants).reset_index()

            chart_left = left + label_width + metric_index * (chart_width + gap)
            axes = canvas.axes(chart_left, top, chart_width, section_height)

            draw_metric(axes, rows, metric, positions)
            draw_chart_heading(canvas, chart_left, top, metric.title, metric.hint)

            if metric_index == 0:
                for row_y, variant in zip(positions, variants):
                    axes.text(
                        -0.06,
                        row_y,
                        variant,
                        transform=axes.get_yaxis_transform(),
                        ha='right',
                        va='center',
                        fontsize=8.5,
                        color=INK_COLOR if variant == 'baseline' else SECONDARY_INK_COLOR,
                        weight='semibold' if variant == 'baseline' else 'normal',
                    )

    canvas.save(ABLATION_FIGURE_PATH, 'ablation figure')
