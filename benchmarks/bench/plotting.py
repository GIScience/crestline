from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bench.environment import logger
from bench.methods import MethodRegistry
from bench.metrics import as_lines
from bench.metrics import dead_ends
from bench.metrics import end_points
from bench.metrics import score
from bench.scene import Scene
from bench.scene import Window
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import NullLocator
from numpy.typing import NDArray
from shapely import LineString

matplotlib.use('Agg')

SURFACE_COLOR = '#fcfcfb'
LAND_COLOR = '#e4e2db'
HAIRLINE_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
INK_COLOR = '#0b0b0b'
SECONDARY_INK_COLOR = '#52514e'
MUTED_COLOR = '#898781'
ACCENT_COLOR = '#2a78d6'
DEAD_END_COLOR = '#eb6834'
NEUTRAL_COLOR = '#aeaca5'

FIGURE_WIDTH = 12.0
LEFT_MARGIN = 0.45
RIGHT_MARGIN = 0.25


class Canvas:
    def __init__(self, width: float, height: float) -> None:
        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_family = ['Inter', 'DejaVu Sans'] if 'Inter' in installed_fonts else ['DejaVu Sans']
        plt.rcParams.update({'font.family': font_family, 'font.size': 9, 'text.color': SECONDARY_INK_COLOR})

        self.width = width
        self.height = height
        self.figure: Figure = plt.figure(figsize=(width, height), facecolor=SURFACE_COLOR)

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


def style_spines(axes: Axes) -> None:
    for side in ('top', 'right', 'left'):
        axes.spines[side].set_visible(False)

    axes.spines['bottom'].set_color(AXIS_COLOR)


def log_ticks(low: float, high: float, steps: tuple[int, ...] = (1,)) -> list[float]:
    powers = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    return [step * 10.0**power for power in powers for step in steps if low <= step * 10.0**power <= high]


def fixed_ticks(axis: Axis, ticks: list[float], labels: list[str]) -> None:
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_ticklabels(labels)


def draw_chart_heading(canvas: Canvas, left: float, top: float, title: str, hint: str) -> None:
    canvas.text(left, top - 0.38, title, fontsize=10.5, color=INK_COLOR, weight='semibold')
    canvas.text(left, top - 0.16, hint, fontsize=8.5, color=MUTED_COLOR)


def _draw_map_header(canvas: Canvas, title: str, subtitle: str) -> None:
    canvas.text(LEFT_MARGIN, 0.5, title, fontsize=14, color=INK_COLOR, weight='semibold')
    canvas.text(LEFT_MARGIN, 0.78, subtitle, fontsize=10)

    legend_right = canvas.width - RIGHT_MARGIN
    canvas.text(legend_right, 0.78, 'centerline', fontsize=9, ha='right')
    canvas.mark([legend_right - 0.95, legend_right - 0.72], 0.745, color=ACCENT_COLOR, lw=2)
    canvas.text(legend_right - 1.12, 0.78, 'dead end', fontsize=9, ha='right')
    canvas.mark([legend_right - 1.82], 0.745, marker='o', ms=6.5, color=DEAD_END_COLOR, mec=SURFACE_COLOR, mew=1.2)


def _window_crop(scene: Scene, window: Window) -> NDArray[np.bool_]:
    min_x, min_y, max_x, max_y = window
    col_start, row_start = ~scene.transform * (min_x, max_y)
    col_end, row_end = ~scene.transform * (max_x, min_y)
    return scene.mask[int(row_start) : int(row_end), int(col_start) : int(col_end)]


def _inside_window(points: NDArray[np.float64], window: Window) -> NDArray[np.bool_]:
    return np.all((points > window[:2]) & (points < window[2:]), axis=1)


def _draw_map(axes: Axes, mask_crop: NDArray[np.bool_], predicted_lines: list[LineString], window: Window) -> None:
    min_x, min_y, max_x, max_y = window
    axes.imshow(
        mask_crop,
        cmap=ListedColormap([SURFACE_COLOR, LAND_COLOR]),
        extent=(min_x, max_x, min_y, max_y),
        interpolation='nearest',
    )
    for line in predicted_lines:
        axes.plot(*line.xy, color=ACCENT_COLOR, lw=1.4, solid_capstyle='round', solid_joinstyle='round')

    dead_end_points = end_points(predicted_lines)
    axes.plot(
        *dead_end_points[_inside_window(dead_end_points, window)].T,
        'o',
        ms=5.5,
        color=DEAD_END_COLOR,
        mec=SURFACE_COLOR,
        mew=1.1,
        zorder=3,
    )

    axes.set_xlim(min_x, max_x)
    axes.set_ylim(min_y, max_y)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(HAIRLINE_COLOR)


def draw_map_panels(
    canvas: Canvas,
    method_registry: MethodRegistry,
    scene: Scene,
    window: Window,
    subtitle: str,
) -> None:
    min_x, min_y, max_x, max_y = window
    _draw_map_header(canvas, f'Noisy road mask, {max_x - min_x:.0f} × {max_y - min_y:.0f} m window', subtitle)

    panel_gap, panel_top = 0.14, 1.55
    panel_count = len(method_registry)
    panel_size = (canvas.width - LEFT_MARGIN - RIGHT_MARGIN - panel_gap * (panel_count - 1)) / panel_count
    mask_crop = _window_crop(scene, window)

    for panel_index, (method_name, method) in enumerate(method_registry.items()):
        predicted_lines = as_lines(method.on_mask(scene.mask, scene.transform))
        panel_left = LEFT_MARGIN + panel_index * (panel_size + panel_gap)
        _draw_map(canvas.axes(panel_left, panel_top, panel_size, panel_size), mask_crop, predicted_lines, window)

        panel_f1 = score(predicted_lines, scene.centerlines)['f5']
        canvas.text(panel_left, panel_top - 0.33, method_name, fontsize=11, color=INK_COLOR, weight='semibold')
        canvas.text(
            panel_left, panel_top - 0.1, f'F1 {panel_f1:.2f} · {dead_ends(predicted_lines):,} dead ends', fontsize=9.5
        )
