from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import shapely
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from shapely import STRtree

from crestline import crestline

matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(__file__).parent.parent / 'benchmarks' / 'data'
ROADS = DATA / 'ldz_road_network.gpkg'
CENTERLINES = DATA / 'ldz_road_network_centerlines.gpkg'
RADIUS_KM = 2
RESOLUTION = 0.5
WINDOW_OFFSET = (300.0, 350.0)
WINDOW_WIDTH = 1500.0
WINDOW_HEIGHT = 750.0

plt.rcParams.update(
    {
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)

surface, land, frame = '#ffffff', '#e4e2db', '#e1e0d9'
ink, muted = '#0b0b0b', '#52514e'
accent, dead_color = '#2a78d6', '#eb6834'


def polygonal(geometry):
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry

    parts = shapely.get_parts(geometry)
    return shapely.multipolygons(shapely.get_parts(parts[np.isin(shapely.get_type_id(parts), (3, 6))]))


def half_widths(polygon, roads, step=2.0):
    rings = shapely.get_rings(shapely.get_parts(polygon))
    xy, ring = shapely.get_coordinates(rings, return_index=True)
    same = ring[1:] == ring[:-1]
    edges = shapely.linestrings(np.stack([xy[:-1][same], xy[1:][same]], axis=1))

    counts = np.maximum(np.ceil(shapely.length(roads) / step).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(roads)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    stations = shapely.line_interpolate_point(
        roads[owner],
        (np.arange(len(owner)) - first + 0.5) / counts[owner],
        normalized=True,
    )

    (index, _), distance = STRtree(edges).query_nearest(stations, return_distance=True)
    gap = np.empty(len(stations))
    gap[index] = distance
    return np.array([np.median(part) for part in np.split(gap, np.cumsum(counts)[:-1])])


def true_dead_ends(lines, widths):
    lines, widths = np.asarray(lines, dtype=object), np.asarray(widths, np.float64)
    ends = np.concatenate([shapely.get_point(lines, 0), shapely.get_point(lines, -1)])
    owner = np.tile(np.arange(len(lines)), 2)

    reach = widths.max() + 0.5 * widths[owner] + 1.0
    end_index, line_index = STRtree(lines).query(ends, predicate='dwithin', distance=reach)
    other = line_index != owner[end_index]
    end_index, line_index = end_index[other], line_index[other]

    gap = shapely.distance(ends[end_index], lines[line_index])
    joined = gap <= widths[line_index] + 0.5 * widths[owner[end_index]]

    dead = ~shapely.is_closed(lines)[owner]
    dead[end_index[joined]] = False
    return int(dead.sum())


def as_lines(geometry):
    if geometry is None or geometry.is_empty:
        return []

    parts = shapely.get_parts(shapely.get_parts(geometry))
    return list(parts[(shapely.get_type_id(parts) == 1) & (shapely.length(parts) > 0)])


def dead_end_points(lines, snap=0.5):
    if not lines:
        return np.zeros((0, 2))

    segments = shapely.get_parts(shapely.node(shapely.MultiLineString(lines)))
    ends = shapely.get_coordinates(
        np.concatenate([shapely.get_point(segments, 0), shapely.get_point(segments, -1)]),
    )
    keys, counts = np.unique(
        np.round(ends / snap).astype(np.int64),
        axis=0,
        return_counts=True,
    )
    return keys[counts == 1] * snap


def sample(lines, step=0.5):
    points = []
    for line in lines:
        distances = np.linspace(0, line.length, max(int(np.ceil(line.length / step)), 1) + 1)
        points.append(shapely.get_coordinates(shapely.line_interpolate_point(line, distances)))

    return np.concatenate(points) if points else np.zeros((0, 2))


def f1(predicted, truth, tolerance=5.0):
    predicted_points, true_points = sample(predicted), sample(truth)
    if len(predicted_points) == 0:
        return 0.0

    precision = (cKDTree(true_points).query(predicted_points)[0] <= tolerance).mean()
    recall = (cKDTree(predicted_points).query(true_points)[0] <= tolerance).mean()
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


network = polygonal(shapely.union_all(gpd.read_file(ROADS).geometry.to_numpy()))
roads = shapely.get_parts(shapely.get_parts(gpd.read_file(CENTERLINES).geometry.to_numpy()))
roads = roads[shapely.length(roads) > 0]
center = shapely.centroid(shapely.multilinestrings(roads))

buffer = shapely.Point(center.x, center.y).buffer(RADIUS_KM * 1000.0, quad_segs=64)
polygon = polygonal(shapely.intersection(network, buffer))
candidates = STRtree(roads).query(buffer, predicate='intersects')
pieces, owner = shapely.get_parts(shapely.intersection(roads[candidates], buffer), return_index=True)
is_line = (shapely.get_type_id(pieces) == 1) & (shapely.length(pieces) > 0)
truth = list(pieces[is_line])
true_dead = true_dead_ends(truth, half_widths(network, roads[candidates])[owner[is_line]])

lines = as_lines(crestline(polygon, resolution=RESOLUTION))
ends = dead_end_points(lines)

window_x, window_y = center.x + WINDOW_OFFSET[0], center.y + WINDOW_OFFSET[1]
window = (
    window_x - WINDOW_WIDTH / 2,
    window_y - WINDOW_HEIGHT / 2,
    window_x + WINDOW_WIDTH / 2,
    window_y + WINDOW_HEIGHT / 2,
)

fig_w, margin, head, foot = 3.5, 0.04, 0.04, 0.25
axes_w = fig_w - 2 * margin
axes_h = axes_w * WINDOW_HEIGHT / WINDOW_WIDTH
fig_h = head + axes_h + foot
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=surface)
ax = fig.add_axes([margin / fig_w, foot / fig_h, axes_w / fig_w, axes_h / fig_h])

gpd.GeoSeries([shapely.clip_by_rect(polygon, *window)]).plot(ax=ax, color=land, linewidth=0)

visible = as_lines(shapely.clip_by_rect(shapely.MultiLineString(lines), *window)) if lines else []
ax.add_collection(
    LineCollection(
        [np.asarray(line.coords) for line in visible],
        colors=accent,
        linewidths=0.5,
        capstyle='round',
        joinstyle='round',
    )
)

inside = (ends[:, 0] > window[0]) & (ends[:, 0] < window[2]) & (ends[:, 1] > window[1]) & (ends[:, 1] < window[3])
ax.plot(
    *ends[inside].T,
    'o',
    ms=2.5,
    color=dead_color,
    mec=surface,
    mew=0.6,
    zorder=3,
)

ax.set_xlim(window[0], window[2])
ax.set_ylim(window[1], window[3])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

for spine in ax.spines.values():
    spine.set_color(frame)

handles = [
    Line2D([], [], color=accent, lw=1.5),
    Line2D(
        [],
        [],
        marker='o',
        ms=2.5,
        linestyle='none',
        color=dead_color,
        mec=surface,
        mew=0.6,
    ),
]
fig.legend(
    handles,
    ['centerline', 'dead end'],
    loc='lower center',
    bbox_to_anchor=(0.5, -10.0 / (300 * fig_h)),
    ncol=2,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.5,
)

fig.savefig(Path(__file__).parent / 'real_results.png', dpi=300, facecolor=surface)
