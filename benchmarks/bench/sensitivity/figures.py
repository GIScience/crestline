import numpy as np
import pandas as pd
from bench.ablation.figures import METRICS
from bench.ablation.figures import SOURCES
from bench.ablation.figures import _draw_legend
from bench.environment import OUTPUT_DIR
from bench.plotting import ACCENT_COLOR
from bench.plotting import DEAD_END_COLOR
from bench.plotting import HAIRLINE_COLOR
from bench.plotting import INK_COLOR
from bench.plotting import MUTED_COLOR
from bench.plotting import NEUTRAL_COLOR
from bench.plotting import SURFACE_COLOR
from bench.plotting import Canvas
from bench.plotting import draw_chart_heading
from bench.plotting import fixed_ticks
from bench.plotting import style_spines
from bench.sensitivity.variants import BASELINE
from bench.sensitivity.variants import sensitivities
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

SENSITIVITY_FIGURE_PATH = OUTPUT_DIR / 'sensitivity_benchmark.png'
PARAMETERS = ('spur', 'tip', 'radius_k')


def _dot_color(variant: str, verdict: str) -> tuple[str, str]:
    if variant == BASELINE:
        return ACCENT_COLOR, ACCENT_COLOR

    if verdict == 'worse':
        return DEAD_END_COLOR, DEAD_END_COLOR

    if verdict == 'better':
        return SURFACE_COLOR, INK_COLOR

    return NEUTRAL_COLOR, NEUTRAL_COLOR


def _sweep(table: pd.DataFrame, parameter: str) -> pd.DataFrame:
    variants = sensitivities()
    baseline = variants[BASELINE]
    names = [
        name
        for name, variant in variants.items()
        if all(getattr(variant, other) == getattr(baseline, other) for other in PARAMETERS if other != parameter)
    ]
    rows = table[table['variant'].isin(names)].copy()
    rows['value'] = [getattr(variants[name], parameter) for name in rows['variant']]
    return rows.sort_values('value')


def _y_ticks(low: float, high: float) -> tuple[list[float], list[str]]:
    ticks = [tick for tick in MaxNLocator(3).tick_values(low, high) if low <= tick <= high]
    step = ticks[1] - ticks[0] if len(ticks) > 1 else max(abs(high - low), 1.0)
    decimals = max(0, -int(np.floor(np.log10(step))))
    while decimals < 6 and any(abs(round(tick, decimals) - tick) > 1e-6 * step for tick in ticks):
        decimals += 1
    return ticks, [f'{round(tick, decimals) + 0.0:,.{decimals}f}' for tick in ticks]


def draw_sweep(axes: Axes, rows: pd.DataFrame) -> None:
    values = rows['value'].to_numpy(float)
    means = rows['variant_mean'].to_numpy(float)
    full_value = rows['full'].iloc[0]

    axes.set_facecolor(SURFACE_COLOR)
    axes.axhline(full_value, color=ACCENT_COLOR, lw=0.8, alpha=0.35, zorder=1)
    axes.plot(values, means, color=HAIRLINE_COLOR, lw=1.4, zorder=2)

    for value, mean, variant, verdict in zip(values, means, rows['variant'], rows['verdict']):
        face, edge = _dot_color(variant, verdict)
        axes.plot([value], [mean], 'o', ms=6.5, color=face, mec=edge, mew=1.2, zorder=3)

    spread = np.nanmax(means) - np.nanmin(means)
    margin = 0.15 * spread if spread > 0 else max(abs(full_value) * 0.1, 0.5)
    low, high = np.nanmin(means) - margin, np.nanmax(means) + margin
    axes.set_ylim(low, high)
    axes.set_xlim(values.min() - 0.08 * np.ptp(values), values.max() + 0.08 * np.ptp(values))

    fixed_ticks(axes.xaxis, list(values), [f'{value:g}' for value in values])
    fixed_ticks(axes.yaxis, *_y_ticks(low, high))
    axes.tick_params(axis='both', length=0, labelsize=7.5, labelcolor=MUTED_COLOR, pad=3)
    axes.grid(axis='y', color=HAIRLINE_COLOR, lw=0.7, zorder=0)
    style_spines(axes)


def sensitivity_figure(table: pd.DataFrame) -> None:
    scenes = int(table['scenes'].max())
    baseline = sensitivities()[BASELINE]

    left, right, label_width, gap = 0.35, 0.25, 1.1, 0.5
    row_height, row_gap, section_gap = 0.95, 0.45, 1.15
    section_height = len(PARAMETERS) * row_height + (len(PARAMETERS) - 1) * row_gap
    canvas = Canvas(15.0, 1.45 + 2 * (section_height + section_gap) + 0.3)
    chart_width = (canvas.width - left - right - label_width - gap * (len(METRICS) - 1)) / len(METRICS)

    canvas.text(
        left,
        0.48,
        'Sensitivity: one parameter swept, the others at their defaults',
        fontsize=14,
        color=INK_COLOR,
        weight='semibold',
    )
    canvas.text(
        left,
        0.76,
        f'Means over {scenes} synthetic scenes (road networks, mixed-width networks, rivers). '
        f'Defaults: spur {baseline.spur:g}, tip {baseline.tip:g}, radius_k {baseline.radius_k:g}. '
        'Colour: paired Wilcoxon signed-rank test against the defaults, Holm-corrected p < 0.05.',
        fontsize=9.5,
    )
    _draw_legend(canvas, canvas.width - right, 1.08)

    for section_index, (source, source_title) in enumerate(SOURCES.items()):
        top = 1.45 + section_index * (section_height + section_gap) + 0.75
        canvas.text(left, top - 0.62, source_title, fontsize=11, color=INK_COLOR, weight='semibold')

        for parameter_index, parameter in enumerate(PARAMETERS):
            row_top = top + parameter_index * (row_height + row_gap)
            canvas.text(left, row_top + row_height / 2, parameter, fontsize=9.5, color=INK_COLOR, va='center')
            sweep = _sweep(table[table['source'] == source], parameter)

            for metric_index, metric in enumerate(METRICS):
                rows = sweep[sweep['metric'] == metric.key]
                chart_left = left + label_width + metric_index * (chart_width + gap)
                draw_sweep(canvas.axes(chart_left, row_top, chart_width, row_height), rows)

                if parameter_index == 0:
                    draw_chart_heading(canvas, chart_left, top, metric.title, metric.hint)

    canvas.save(SENSITIVITY_FIGURE_PATH, 'sensitivity figure')
