from pathlib import Path

import geopandas as gpd
import matplotlib
from matplotlib.lines import Line2D
from shapely import Point

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROADS = Path(__file__).parent.parent / 'benchmarks' / 'data' / 'ldz_road_network.gpkg'
RADII_KM = [1, 2, 3, 4, 5]

plt.rcParams.update(
    {
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)

road_color, buffer_color = '#52514e', '#2a78d6'
dash = (0, (10, 5))

roads = gpd.read_file(ROADS)
if roads.crs.is_geographic:
    roads = roads.to_crs(roads.estimate_utm_crs())

weights = roads.area
centroids = roads.centroid
center = Point((centroids.x * weights).sum() / weights.sum(), (centroids.y * weights).sum() / weights.sum())
rings = gpd.GeoSeries([center.buffer(r * 1000, quad_segs=64).boundary for r in RADII_KM], crs=roads.crs)

minx, miny, maxx, maxy = rings.iloc[-1].bounds
pad = 0.02 * (maxx - minx)
half_h = (maxy - miny) / 2 + pad
half_w = 1.6 * half_h
window = (center.x - half_w, center.y - half_h, center.x + half_w, center.y + half_h)
roads = roads.cx[window[0] : window[2], window[1] : window[3]]

fig_w, legend_h = 3.5, 0.2
ax_h = fig_w * (window[3] - window[1]) / (window[2] - window[0])
fig_h = ax_h + legend_h
fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
ax = fig.add_axes([0, legend_h / fig_h, 1, ax_h / fig_h])

roads.plot(
    ax=ax,
    color=road_color,
    edgecolor=road_color,
    linewidth=0.01,
)
rings.plot(
    ax=ax,
    color=buffer_color,
    linewidth=0.9,
    linestyle=dash,
)

ax.set_xlim(window[0], window[2])
ax.set_ylim(window[1], window[3])
ax.set_aspect('equal')
ax.set_axis_off()

handles = [
    Line2D(
        [],
        [],
        color=road_color,
        lw=3.0,
    ),
    Line2D(
        [],
        [],
        color=buffer_color,
        lw=1.5,
        linestyle=dash,
    ),
]
labels = ['road network', 'buffers, 1–5 km']
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, legend_h / fig_h),
    ncol=2,
    frameon=False,
    handlelength=2.5,
    columnspacing=1.5,
    borderaxespad=0.3,
    borderpad=0,
)

fig.savefig(Path(__file__).parent / 'real_data.png', dpi=300, facecolor='white')
