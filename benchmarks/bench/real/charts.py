from collections.abc import Callable
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
from bench.plotting import ACCENT_COLOR
from bench.plotting import HAIRLINE_COLOR
from bench.plotting import INK_COLOR
from bench.plotting import MUTED_COLOR
from bench.plotting import NEUTRAL_COLOR
from bench.plotting import SECONDARY_INK_COLOR
from bench.plotting import SURFACE_COLOR
from bench.plotting import style_spines
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray

Trend: TypeAlias = tuple[list[int], list[float]]
Trends: TypeAlias = dict[str, Trend]
LabelPlacement: TypeAlias = tuple[tuple[int, int], str, str]


class LineEnd(NamedTuple):
    x: float
    y: float
    text: str
    is_crestline: bool


LABEL_PLACEMENTS: list[LabelPlacement] = [
    ((7, 0), 'left', 'center'),
    ((4, 5), 'left', 'bottom'),
    ((4, -5), 'left', 'top'),
    ((0, 8), 'center', 'bottom'),
    ((0, -8), 'center', 'top'),
    ((-7, 0), 'right', 'center'),
]
METHOD_STYLES: dict[str, tuple[str, str]] = {
    'crestline': ('crestline', 'o'),
    'pygeoops.centerline': ('pygeoops', 's'),
    'skeletonize + prune': ('skeletonize', '^'),
    'centerline (Voronoi)': ('Voronoi', 'D'),
}


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


def _push_apart(placed: NDArray[np.float64], gap: float) -> None:
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)


def spread(positions: list[float], gap: float) -> list[float]:
    order = np.argsort(positions)
    placed = np.asarray(positions, dtype=float)[order]
    _push_apart(placed, gap)

    overflow = placed[-1] - max(positions) if len(placed) else 0.0
    placed -= max(0.0, overflow / 2)
    _push_apart(placed, gap)

    spread_positions = np.empty_like(placed)
    spread_positions[order] = placed
    return list(spread_positions)


def annotate_end(
    axes: Axes,
    text: str,
    anchor: tuple[float, float],
    placement: LabelPlacement,
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


def label_line_ends(axes: Axes, ends: list[LineEnd], series_lines: list[Line2D], gap: float = 0.1) -> None:
    renderer = axes.figure.canvas.get_renderer()
    to_axes = axes.transAxes.inverted()
    line_paths = [line.get_transform().transform_path(line.get_path()) for line in series_lines]
    last_end_x = max(end.x for end in ends)
    placed_boxes: list[Bbox] = []

    def is_clear(box: Bbox) -> bool:
        crosses_line = any(path.intersects_bbox(box, filled=False) for path in line_paths)
        return not crosses_line and not any(box.overlaps(placed_box) for placed_box in placed_boxes)

    for end_x in sorted({end.x for end in ends}, reverse=True):
        group = [end for end in ends if end.x == end_x]
        heights = [to_axes.transform(axes.transData.transform((end_x, end.y)))[1] for end in group]
        placements = LABEL_PLACEMENTS if end_x < last_end_x else LABEL_PLACEMENTS[:1]

        for end, height in zip(group, spread(heights, gap)):
            candidates = [
                annotate_end(axes, end.text, (end_x, height), placement, end.is_crestline) for placement in placements
            ]
            boxes = [candidate.get_window_extent(renderer).padded(2) for candidate in candidates]
            chosen = next((index for index, box in enumerate(boxes) if is_clear(box)), 0)

            for index, candidate in enumerate(candidates):
                if index != chosen:
                    candidate.remove()

            placed_boxes.append(boxes[chosen])


def _value_range(values: list[float], log_scale: bool) -> tuple[float, float, list[float]]:
    if log_scale:
        low, high = min(values) / 1.6, max(values) * 1.6
        return low, high, log_ticks(low, high)

    low, high = np.floor(min(values) * 10) / 10, 1.0
    step = 0.05 if high - low <= 0.2 else 0.1
    ticks = list(np.round(np.arange(low, high + step / 2, step), 2))
    return low - 0.015, high + 0.015, ticks


def _plot_series(axes: Axes, series: Trends, truth: Trend | None) -> tuple[list[LineEnd], list[Line2D]]:
    ends: list[LineEnd] = []
    series_lines: list[Line2D] = []
    if truth is not None:
        series_lines += axes.plot(*truth, color=MUTED_COLOR, lw=1.1, zorder=1, solid_capstyle='round')
        ends.append(LineEnd(truth[0][-1], truth[1][-1], 'true', False))

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
        ends.append(LineEnd(method_radii[-1], method_values[-1], short_name, is_crestline))

    return ends, series_lines


def _format_ticks(axes: Axes, radii: list[int], ticks: list[float], tick_format: Callable[[float], str]) -> None:
    axes.yaxis.set_major_locator(FixedLocator(ticks))
    axes.yaxis.set_minor_locator(NullLocator())
    axes.set_yticklabels([tick_format(tick) for tick in ticks])

    labelled_radii = [radius for radius in radii if radius == radii[0] or radius % 2 == 0 or len(radii) <= 6]
    axes.xaxis.set_major_locator(FixedLocator(labelled_radii))
    axes.xaxis.set_minor_locator(FixedLocator(radii))
    axes.set_xticklabels([f'{radius} km' if radius == radii[-1] else str(radius) for radius in labelled_radii])
    axes.tick_params(axis='both', which='both', length=0, labelsize=8.5, labelcolor=MUTED_COLOR, pad=5)


def draw_trend(
    axes: Axes,
    series: Trends,
    radii: list[int],
    log_scale: bool,
    tick_format: Callable[[float], str],
    truth: Trend | None = None,
) -> None:
    axes.set_facecolor(SURFACE_COLOR)
    values = [value for _, method_values in series.values() for value in method_values]
    if truth is not None:
        values += truth[1]

    if log_scale:
        axes.set_yscale('log')

    low, high, ticks = _value_range(values, log_scale)
    axes.set_ylim(low, high)
    axes.set_xlim(radii[0] - 0.5, radii[-1] + 0.5)
    for tick in ticks:
        axes.axhline(tick, color=HAIRLINE_COLOR, lw=0.8, zorder=0)

    ends, series_lines = _plot_series(axes, series, truth)
    label_line_ends(axes, ends, series_lines)
    _format_ticks(axes, radii, ticks, tick_format)
    style_spines(axes)
