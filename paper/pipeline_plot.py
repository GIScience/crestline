from pathlib import Path

import matplotlib
import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely import LineString
from shapely.geometry.polygon import Polygon

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crestline.crestline import _crests
from crestline.crestline import _despur
from crestline.crestline import _flood
from crestline.crestline import _grid
from crestline.crestline import _on_ridge
from crestline.crestline import _prune
from crestline.crestline import _rasterize
from crestline.crestline import _thin


def example_shape():
    roads = [
        (LineString([(0, 26), (40, 31), (85, 27), (135, 33)]), 8),
        (LineString([(62, 29), (70, 62), (78, 90)]), 5),
        (LineString([(18, 28), (21, 68), (68, 62)]), 5.5),
        (LineString([(112, 32), (121, 4)]), 6),
    ]
    return shapely.union_all([g.buffer(h) for g, h in roads])


def noisy(polygon: Polygon, seed: int, coarse: float = 1.1, fine: float = 0.45, pad: int = 6):
    mask, _ = _rasterize(polygon, 1.0)
    mask = np.pad(mask, pad)
    sd = distance_transform_edt(mask) - distance_transform_edt(~mask)
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.normal(size=mask.shape), 3.0)
    b = gaussian_filter(rng.normal(size=mask.shape), 1.0)
    return sd + a * coarse / a.std() + b * fine / b.std() > 0.5


def stages(mask, dip=0.65, spur=2.0, tip=0.75):
    depth_p = _flood(_grid(mask))
    crest_p = (_crests(depth_p) == 1) & (depth_p >= 2)
    depth = depth_p[2:-2, 2:-2]
    crest = crest_p[2:-2, 2:-2]

    rows, cols = np.nonzero(crest)
    height = depth[rows, cols]
    order = np.argsort(-height, kind='stable')

    rows, cols, height = rows[order], cols[order], height[order]
    keep = _thin(rows, cols, mask.shape[0], mask.shape[1], np.maximum(0.35 * height, 3))
    nodes, heights = np.c_[rows[keep], cols[keep]], height[keep]
    near = cKDTree(nodes).query_pairs(max(2, spur) * heights.max(), output_type='ndarray')
    span = np.hypot(*(nodes[near[:, 0]] - nodes[near[:, 1]]).T)
    cand = near[span <= 2 * np.maximum(heights[near[:, 0]], heights[near[:, 1]])]

    ridge_mask = _on_ridge(nodes, cand, depth, dip, 2 * heights.max() + 2)
    ridge = cand[ridge_mask]

    prune_mask = _prune(nodes, ridge)
    pruned = ridge[prune_mask]

    spur_mask = _despur(nodes, pruned, heights, near, spur, tip)

    final = pruned[spur_mask]

    shift, same = np.zeros(nodes.shape), np.zeros(len(nodes))
    for d in np.argwhere(np.ones((3, 3))) - 1:
        r, c = (nodes + d).T
        hit = crest_p[r + 2, c + 2] & (depth[r, c] == heights)
        shift, same = shift + hit[:, None] * d, same + hit

    rc = nodes + 0.5 + shift / same[:, None]
    lines = shapely.line_merge(shapely.multilinestrings(shapely.linestrings(np.c_[rc[:, 1], rc[:, 0]][final])))
    return dict(
        depth=depth,
        crest=crest,
        points=nodes,
        rc=rc,
        cand=cand,
        ridge_fail=cand[~ridge_mask],
        ridge=ridge,
        shortcut=ridge[~prune_mask],
        spurs=pruned[~spur_mask],
        final=final,
        lines=lines,
    )


plt.rcParams.update(
    {
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.linewidth': 0.5,
        'text.color': '#0b0b0b',
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False,
    }
)

surface, land, ink = '#ffffff', '#e4e2db', '#0b0b0b'
blue, orange, grey = '#2a78d6', '#f45642', '#6f6e69'
ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
seq = LinearSegmentedColormap.from_list('seq', ramp)

mask = noisy(example_shape(), 2)
st = stages(mask)
h, w = mask.shape
depth = st['depth']
points = st['points'] + 0.5
rc = st['rc']

fig_w = 10
gap = 0.03
pw = (fig_w - 3 * gap) / 4
ph = pw * h / w
top_pad, bottom_pad = 0.2, 0.4
fig_h = ph + top_pad + bottom_pad
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=surface)


def ax_at(i):
    new_ax = fig.add_axes([i * (pw + gap) / fig_w, bottom_pad / fig_h, pw / fig_w, ph / fig_h])
    new_ax.set_xlim(0, w)
    new_ax.set_ylim(h, 0)
    new_ax.set_xticks([])
    new_ax.set_yticks([])
    return new_ax


def base(ax):
    ax.imshow(
        mask,
        cmap=ListedColormap([surface, land]),
        interpolation='nearest',
        extent=(0, w, h, 0),
    )


def overlay(ax, pairs, xy, **kw):
    for i, j in pairs:
        ax.plot([xy[i, 1], xy[j, 1]], [xy[i, 0], xy[j, 0]], **kw)


ax = ax_at(0)
im = ax.imshow(np.ma.masked_where(~mask, depth), cmap=seq, interpolation='nearest', extent=(0, w, h, 0), vmin=1)
ax.set_facecolor(surface)

cax = fig.add_axes([0.02 * pw / fig_w, 0.17 / fig_h, 0.6 * pw / fig_w, 0.06 / fig_h])
color_bar = fig.colorbar(im, cax=cax, orientation='horizontal')
color_bar.outline.set_linewidth(0.4)
color_bar.set_ticks([1, 9])
color_bar.ax.tick_params(labelsize=8, length=2, width=0.4, pad=1)
fig.text((0.64 * pw) / fig_w, 0.2 / fig_h, '$h$ [px]', fontsize=8, va='center')

ax = ax_at(1)
base(ax)
cr = np.ma.masked_where(~st['crest'], np.ones_like(depth))
ax.imshow(cr, cmap=ListedColormap([blue]), interpolation='nearest', extent=(0, w, h, 0))

ax = ax_at(2)
base(ax)
overlay(ax, st['ridge'], points, color=grey, lw=0.45, zorder=2)
overlay(ax, st['ridge_fail'], points, color=orange, lw=0.9, ls=(0, (2.2, 1.2)), zorder=3)
ax.plot(points[:, 1], points[:, 0], '.', ms=1, color=ink, zorder=4)

ax = ax_at(3)
base(ax)

for g in st['lines'].geoms:
    x, y = g.xy
    ax.plot(
        x,
        y,
        color=blue,
        lw=1.1,
        solid_capstyle='round',
        solid_joinstyle='round',
        zorder=4,
    )

titles = [
    '(a) Inward flood depth $h$',
    '(b) Crest pixels',
    '(c) Candidate links',
    '(d) Pruned graph (centerline)',
]

for i, t in enumerate(titles):
    fig.text(
        i * (pw + gap) / fig_w,
        (bottom_pad + ph + 0.06) / fig_h,
        t,
        fontsize=8,
        ha='left',
        va='bottom',
    )

handles = [
    Line2D([], [], marker='s', ls='', ms=3.5, color=blue, label='crest pixel'),
    Line2D([], [], marker='o', ls='', ms=1, color=ink, label='node'),
    Line2D([], [], color=grey, lw=0.8, label='link on ridge'),
    Line2D([], [], color=orange, lw=0.8, ls=(0, (2.2, 1.2)), label='rejected link'),
    Line2D([], [], color=blue, lw=1.2, label='centerline'),
]

fig.legend(
    handles=handles,
    loc='lower left',
    bbox_to_anchor=((pw + gap - 0.15) / fig_w, 0.02),
    ncol=5,
    frameon=False,
    fontsize=8,
    handlelength=2,
    columnspacing=2.0,
    handletextpad=0.5,
)

fig.savefig(Path(__file__).parent / 'pipeline.png', dpi=300, facecolor=surface)
