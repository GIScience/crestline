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
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
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


def map_window_title(window: Window) -> str:
    window_min_x, window_min_y, window_max_x, window_max_y = window
    return f'Noisy road mask, {window_max_x - window_min_x:.0f} × {window_max_y - window_min_y:.0f} m window'


def style_spines(axes: Axes) -> None:
    for side in ('top', 'right', 'left'):
        axes.spines[side].set_visible(False)

    axes.spines['bottom'].set_color(AXIS_COLOR)


def draw_chart_heading(
    canvas: Canvas,
    left: float,
    top: float,
    title: str,
    hint: str,
) -> None:
    canvas.text(left, top - 0.38, title, fontsize=10.5, color=INK_COLOR, weight='semibold')
    canvas.text(left, top - 0.16, hint, fontsize=8.5, color=MUTED_COLOR)


def draw_header(
    canvas: Canvas,
    title: str,
    subtitle: str,
    left_margin: float,
    right_margin: float,
) -> None:
    canvas.text(left_margin, 0.5, title, fontsize=14, color=INK_COLOR, weight='semibold')
    canvas.text(left_margin, 0.78, subtitle, fontsize=10)

    legend_right = canvas.width - right_margin
    canvas.text(legend_right, 0.78, 'centerline', fontsize=9, ha='right')
    canvas.mark([legend_right - 0.95, legend_right - 0.72], 0.745, color=ACCENT_COLOR, lw=2)
    canvas.text(legend_right - 1.12, 0.78, 'dead end', fontsize=9, ha='right')
    canvas.mark([legend_right - 1.82], 0.745, marker='o', ms=6.5, color=DEAD_END_COLOR, mec=SURFACE_COLOR, mew=1.2)


def _window_crop(scene: Scene, window: Window) -> NDArray[np.bool_]:
    window_min_x, window_min_y, window_max_x, window_max_y = window
    crop_col_start, crop_row_start = ~scene.transform * (window_min_x, window_max_y)
    crop_col_end, crop_row_end = ~scene.transform * (window_max_x, window_min_y)
    return scene.mask[int(crop_row_start) : int(crop_row_end), int(crop_col_start) : int(crop_col_end)]


def _inside_window(points: NDArray[np.float64], window: Window) -> NDArray[np.bool_]:
    window_min_x, window_min_y, window_max_x, window_max_y = window
    return (
        (points[:, 0] > window_min_x)
        & (points[:, 0] < window_max_x)
        & (points[:, 1] > window_min_y)
        & (points[:, 1] < window_max_y)
    )


def _draw_map(
    axes: Axes,
    mask_crop: NDArray[np.bool_],
    predicted_lines: list[LineString],
    window: Window,
) -> None:
    window_min_x, window_min_y, window_max_x, window_max_y = window
    axes.imshow(
        mask_crop,
        cmap=ListedColormap([SURFACE_COLOR, LAND_COLOR]),
        extent=(window_min_x, window_max_x, window_min_y, window_max_y),
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

    axes.set_xlim(window_min_x, window_max_x)
    axes.set_ylim(window_min_y, window_max_y)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(HAIRLINE_COLOR)


def draw_panels(
    canvas: Canvas,
    method_names: list[str],
    method_registry: MethodRegistry,
    scene: Scene,
    window: Window,
    left_margin: float,
    right_margin: float,
    panel_gap: float = 0.14,
    panel_top: float = 1.55,
) -> None:
    mask_crop = _window_crop(scene, window)
    panel_count = len(method_names)
    panel_size = (canvas.width - left_margin - right_margin - panel_gap * (panel_count - 1)) / panel_count

    for panel_index, method_name in enumerate(method_names):
        predicted_lines = as_lines(method_registry[method_name].on_mask(scene.polygon, scene.mask, scene.transform))
        panel_left = left_margin + panel_index * (panel_size + panel_gap)

        panel_axes = canvas.axes(panel_left, panel_top, panel_size, panel_size)
        _draw_map(panel_axes, mask_crop, predicted_lines, window)

        panel_f1 = score(predicted_lines, scene.centerlines)['f5']
        canvas.text(panel_left, panel_top - 0.33, method_name, fontsize=11, color=INK_COLOR, weight='semibold')
        canvas.text(
            panel_left, panel_top - 0.1, f'F1 {panel_f1:.2f} · {dead_ends(predicted_lines):,} dead ends', fontsize=9.5
        )
