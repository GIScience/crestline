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

road_color, buffer_color = '#3d3c39', '#2a78d6'
dash = (0, (5, 3))

roads = gpd.read_file(ROADS)
if roads.crs.is_geographic:
    roads = roads.to_crs(roads.estimate_utm_crs())

weights = roads.area
centroids = roads.centroid
center = Point((centroids.x * weights).sum() / weights.sum(), (centroids.y * weights).sum() / weights.sum())
rings = gpd.GeoSeries([center.buffer(r * 1000, quad_segs=64).boundary for r in RADII_KM], crs=roads.crs)

minx, miny, maxx, maxy = rings.iloc[-1].bounds
roads = roads.cx[minx:maxx, miny:maxy]

fig_w, legend_h = 3.5, 0.35
fig = plt.figure(figsize=(fig_w, fig_w + legend_h), facecolor='white')
ax = fig.add_axes([0, legend_h / (fig_w + legend_h), 1, fig_w / (fig_w + legend_h)])

roads.plot(
    ax=ax,
    color=road_color,
    edgecolor=road_color,
    linewidth=0.1,
)
rings.plot(
    ax=ax,
    color=buffer_color,
    linewidth=1.5,
    linestyle=dash,
)

pad = 0.02 * (maxx - minx)
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
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
    loc='lower center',
    ncol=2,
    frameon=False,
    handlelength=2.5,
    columnspacing=1.5,
)

fig.savefig(Path(__file__).parent / 'real_data.png', dpi=300, facecolor='white')
