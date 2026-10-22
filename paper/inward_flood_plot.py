from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crestline.crestline import _flood
from crestline.crestline import _grid

plt.rcParams.update(
    {
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)

surface, ink, shore_color = '#ffffff', '#0b0b0b', '#b9b7ae'
ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
seq = LinearSegmentedColormap.from_list('seq', ramp)


def flood_mask(size: int = 40, amp: float = 2.5, periods: float = 1.5, half: float = 3.3, margin: float = 4.5):
    r, c = np.ogrid[:size, :size]
    y = size / 2 + amp * np.cos(2 * np.pi * periods * (c - margin) / (size - 2 * margin))
    return (np.abs(r - y) <= half) & (c >= margin) & (c < size - margin)


def flood_shape(mask):
    state = _grid(mask)
    depth = _flood(state)
    shore = binary_dilation(state == 2, structure=np.ones((3, 3), bool)) & (state == 1)
    return depth[2:-2, 2:-2], shore[2:-2, 2:-2]


def outline(ax, region, color, lw):
    h, w = region.shape
    p = np.pad(region, 1)

    segments = []
    for r in range(h):
        for c in range(w + 1):
            if p[r + 1, c] != p[r + 1, c + 1]:
                segments.append([(c, r), (c, r + 1)])

    for r in range(h + 1):
        for c in range(w):
            if p[r, c + 1] != p[r + 1, c + 1]:
                segments.append([(c, r), (c + 1, r)])

    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=lw,
            capstyle='projecting',
            zorder=4,
        )
    )


mask = flood_mask()
depth, shore = flood_shape(mask)
rows, cols = np.nonzero(shore | mask)
box = np.s_[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
depth, shore, inside = depth[box], shore[box], mask[box]

h, w = depth.shape
top = max(int(depth.max()), 2)

fig_w = 3.5
fig = plt.figure(figsize=(fig_w, fig_w * h / w), facecolor=surface)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(surface)

ax.imshow(
    np.ma.masked_where(~inside, depth),
    cmap=seq,
    vmin=1,
    vmax=top,
    interpolation='nearest',
    extent=(0, w, h, 0),
)
ax.imshow(
    np.ma.masked_where(~shore, np.ones(depth.shape)),
    cmap=ListedColormap([shore_color]),
    interpolation='nearest',
    extent=(0, w, h, 0),
)

for (r, c), v in np.ndenumerate(depth):
    if inside[r, c] or shore[r, c]:
        dark = inside[r, c] and v > 0.55 * top
        ax.text(
            c + 0.5,
            r + 0.52,
            str(v),
            ha='center',
            va='center',
            fontsize=6,
            color=surface if dark else ink,
        )

ax.set_xticks(np.arange(w + 1), minor=True)
ax.set_yticks(np.arange(h + 1), minor=True)
ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

for sp in ax.spines.values():
    sp.set_visible(False)

fig.savefig(Path(__file__).parent / 'inward_flood.png', dpi=300, facecolor=surface)
