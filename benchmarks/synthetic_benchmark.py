import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from jsonargparse import CLI
from numba import njit
from rasterio.features import rasterize
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely import LineString
from shapely import MultiLineString
from tqdm import tqdm

CUDA = 'crestline (CUDA)'
COLUMNS = [
    ('mpx', 'mpx', 1, 2),
    ('f1_5m', 'f5', 1, 4),
    ('f1_2m', 'f2', 1, 4),
    ('offset_m', 'offset', 1, 3),
    ('dead_ends', 'dead', 1, 1),
    ('true_dead_ends', 'true_dead', 1, 1),
    ('loops', 'loops', 1, 1),
    ('holes', 'holes', 1, 1),
    ('runtime_ms', 'time', 1000, 2),
]
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def road_network(seed, n_main=5, n_side=8, extent=1000.0):
    rng = np.random.default_rng(seed)
    roads = []

    for index in range(n_main):
        along = np.linspace(0, extent, 5)
        across = rng.uniform(0.15, 0.85) * extent + np.cumsum(rng.normal(0, 50, 5))
        roads.append(LineString(np.c_[along, across] if index % 2 == 0 else np.c_[across, along]))

    for _ in range(n_side):
        anchor = roads[rng.integers(n_main)].interpolate(rng.uniform(0.1, 0.9), normalized=True)
        angle, length = rng.uniform(0, 2 * np.pi), rng.uniform(80, 220)
        tip = (anchor.x + length * np.cos(angle), anchor.y + length * np.sin(angle))
        roads.append(LineString([(anchor.x, anchor.y), tip]))

    return roads


def river(seed, extent=1000.0, n_trib=4):
    rng = np.random.default_rng(seed)
    progress = np.linspace(0, 1, 60)
    trunk = LineString(np.c_[progress * extent, extent * 0.5 + np.cumsum(rng.normal(0, 12, 60))])
    lines, widths = [trunk], [(5.0, 30.0)]

    for _ in range(n_trib):
        position = rng.uniform(0.3, 0.9)
        junction = trunk.interpolate(position, normalized=True)
        angle = rng.choice([-1, 1]) * rng.uniform(0.6, 1.3) + np.pi
        length = rng.uniform(150, 300)

        steps = np.linspace(0, 1, 20)
        wiggle = np.cumsum(rng.normal(0, 4, 20))
        xs = junction.x + steps * length * np.cos(angle) - wiggle * np.sin(angle)
        ys = junction.y + steps * length * np.sin(angle) + wiggle * np.cos(angle)

        lines.append(LineString(np.c_[xs, ys]))
        widths.append((5 + 12.5 * position, 5.0))

    return lines, widths


def tapered(line, w0, w1, step=2.0):
    count = max(int(line.length / step), 2)
    points = [line.interpolate(distance) for distance in np.linspace(0, line.length, count)]
    radii = np.linspace(w0, w1, count)
    segments = zip(points[:-1], points[1:], radii)

    return shapely.union_all(
        [LineString([(start.x, start.y), (end.x, end.y)]).buffer(radius) for start, end, radius in segments]
    )


def make_case(kind, seed):
    if kind == 'roads':
        lines = road_network(seed)
        return lines, [15.0] * len(lines), MultiLineString(lines).buffer(15)

    if kind == 'roads-large':
        lines = road_network(seed, 20, 32, 4000.0)
        return lines, [15.0] * len(lines), MultiLineString(lines).buffer(15)

    if kind == 'roads-xl':
        lines = road_network(seed, 40, 64, 8000.0)
        return lines, [15.0] * len(lines), MultiLineString(lines).buffer(15)

    if kind == 'mixed':
        lines = road_network(seed)
        half = list(np.random.default_rng(seed + 100).choice([6.0, 9.0, 14.0, 20.0, 30.0], len(lines)))
        return lines, half, shapely.union_all([line.buffer(width) for line, width in zip(lines, half)])

    if kind == 'river':
        lines, widths = river(seed)
        poly = shapely.union_all([tapered(line, *pair) for line, pair in zip(lines, widths)])
        return lines, [max(pair) for pair in widths], poly

    raise ValueError(kind)


def noisy_mask(poly, seed, pad=40, coarse=2.5, fine=0.8):
    rng = np.random.default_rng(seed + 1000)
    minx, miny, maxx, maxy = poly.bounds
    shape = (int(np.ceil(maxy - miny)) + 2 * pad, int(np.ceil(maxx - minx)) + 2 * pad)
    transform = from_origin(minx - pad, maxy + pad, 1, 1)

    clean = rasterize([(poly, 1)], out_shape=shape, transform=transform).astype(bool)
    sdf = distance_transform_edt(clean) - distance_transform_edt(~clean)

    big = gaussian_filter(rng.normal(size=shape), 4.0)
    small = gaussian_filter(rng.normal(size=shape), 1.0)

    return sdf + big * coarse / big.std() + small * fine / small.std() > 0.5, transform


def vectorize(mask, transform):
    found = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    polys = [shapely.geometry.shape(geometry) for geometry, _ in found]
    return shapely.MultiPolygon(polys) if len(polys) > 1 else polys[0]


def as_lines(geom):
    if geom is None or geom.is_empty:
        return []

    out = []
    for part in shapely.get_parts(geom):
        if part.geom_type == 'LineString':
            out.append(part)
        elif part.geom_type in ('MultiLineString', 'GeometryCollection'):
            out.extend(as_lines(part))

    return [line for line in out if line.length > 0]


def sample(lines, step=0.5):
    points = [
        shapely.get_coordinates(
            shapely.line_interpolate_point(
                line, np.linspace(0, line.length, max(int(np.ceil(line.length / step)), 1) + 1)
            )
        )
        for line in lines
    ]

    return np.concatenate(points) if points else np.zeros((0, 2))


def end_keys(lines):
    parts = shapely.get_parts(shapely.node(MultiLineString(lines)))
    ends = [coord for part in parts for coord in (part.coords[0], part.coords[-1])]
    return np.unique(np.round(np.array(ends) / 0.5).astype(np.int64), axis=0, return_counts=True)


def dead_ends(lines):
    if not lines:
        return 0

    _, count = end_keys(lines)
    return int((count == 1).sum())


def end_points(lines):
    if not lines:
        return np.zeros((0, 2))

    keys, count = end_keys(lines)
    return keys[count == 1] * 0.5


def loops(lines):
    if not lines:
        return 0

    return len(shapely.get_parts(shapely.polygonize(shapely.get_parts(shapely.node(MultiLineString(lines))))))


def holes(poly, smallest=50.0):
    return sum(
        1 for polygon in shapely.get_parts(poly) for ring in polygon.interiors if shapely.Polygon(ring).area >= smallest
    )


def true_dead_ends(lines, half):
    count = 0
    for index, line in enumerate(lines):
        for point in (shapely.Point(line.coords[0]), shapely.Point(line.coords[-1])):
            if all(
                point.distance(other) > half[other_index] + 0.5 * half[index]
                for other_index, other in enumerate(lines)
                if other_index != index
            ):
                count += 1

    return count


def score(pred, truth):
    truth_points, pred_points = sample(truth), sample(pred)
    if len(pred_points) == 0:
        return {'f5': 0.0, 'f2': 0.0, 'offset': float('nan')}

    d_pred = cKDTree(truth_points).query(pred_points)[0]
    d_true = cKDTree(pred_points).query(truth_points)[0]

    out = {}
    for tol in (2, 5):
        precision, recall = (d_pred <= tol).mean(), (d_true <= tol).mean()
        out[f'f{tol}'] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    out['offset'] = float(d_pred[d_pred <= 5].mean())
    return out


@njit(cache=True)
def _drop_spurs(n_nodes, edges, length, limit):
    count = np.zeros(n_nodes + 1, np.int64)
    for edge in range(len(edges)):
        count[edges[edge, 0] + 1] += 1
        count[edges[edge, 1] + 1] += 1

    start = np.cumsum(count)
    nbr = np.empty(start[-1], np.int64)
    via = np.empty(start[-1], np.int64)
    fill = start[:-1].copy()
    for edge in range(len(edges)):
        for node, neighbor in ((edges[edge, 0], edges[edge, 1]), (edges[edge, 1], edges[edge, 0])):
            nbr[fill[node]], via[fill[node]] = neighbor, edge
            fill[node] += 1

    degree = count[1:].copy()
    alive = np.ones(len(edges), np.bool_)
    chain = np.empty(len(edges), np.int64)

    changed = True
    while changed:
        changed = False
        for vertex in range(n_nodes):
            if degree[vertex] != 1:
                continue

            cur, came, total, chain_length = vertex, -1, 0.0, 0
            while True:
                nxt, edge = -1, -1
                for slot in range(start[cur], start[cur + 1]):
                    if alive[via[slot]] and via[slot] != came:
                        nxt, edge = nbr[slot], via[slot]
                        break

                if nxt < 0:
                    break

                chain[chain_length] = edge
                chain_length += 1
                total += length[edge]
                cur, came = nxt, edge
                if degree[cur] != 2 or total >= limit:
                    break

            if total < limit and degree[cur] != 2:
                for step in range(chain_length):
                    alive[chain[step]] = False
                    degree[edges[chain[step], 0]] -= 1
                    degree[edges[chain[step], 1]] -= 1

                changed = True

    return alive


def skeleton(mask, transform, prune=2.0):
    from skimage.morphology import skeletonize

    skel = np.pad(skeletonize(mask), 1)
    rows, cols = np.nonzero(skel)
    if len(rows) < 2:
        return MultiLineString()

    ids = np.full(skel.shape, -1, np.int64)
    ids[rows, cols] = np.arange(len(rows))

    edges = []
    for row_step, col_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        linked = skel[rows + row_step, cols + col_step]
        if row_step and col_step:
            linked &= ~skel[rows, cols + col_step] & ~skel[rows + row_step, cols]

        edges.append(np.c_[ids[rows[linked], cols[linked]], ids[rows[linked] + row_step, cols[linked] + col_step]])

    edges = np.concatenate(edges)
    length = np.hypot(rows[edges[:, 0]] - rows[edges[:, 1]], cols[edges[:, 0]] - cols[edges[:, 1]])

    limit = prune * np.median(distance_transform_edt(mask)[rows - 1, cols - 1])
    edges = edges[_drop_spurs(len(rows), edges, length, limit)]
    if len(edges) == 0:
        return MultiLineString()

    xy = np.c_[transform * (cols - 0.5, rows - 0.5)]
    return shapely.line_merge(shapely.multilinestrings(shapely.linestrings(xy[edges])))


def cuda_device():
    try:
        from numba import cuda

        if not cuda.is_available():
            return '', 'no device found'

        name = cuda.get_current_device().name
    except Exception as error:
        return '', f'{type(error).__name__}: {error}'

    return (name.decode() if isinstance(name, bytes) else name), ''


def methods(with_cuda=False):
    from centerline.geometry import Centerline
    from pygeoops import centerline as pygeoops_centerline

    from crestline import crestline

    def voronoi(geom):
        if geom.geom_type == 'MultiPolygon':
            return shapely.union_all([voronoi(part) for part in geom.geoms])

        try:
            return Centerline(geom, interpolation_distance=1.0).geometry
        except Exception:
            return MultiLineString()

    def burn(poly, mask, tr):
        return rasterize([(poly, 1)], out_shape=mask.shape, transform=tr).astype(bool)

    out = {
        'crestline': (
            lambda poly, mask, tr: crestline(poly, resolution=1.0),
            lambda poly, mask, tr: crestline(mask, transform=tr),
        ),
        'pygeoops.centerline': (
            lambda poly, mask, tr: pygeoops_centerline(poly),
            lambda poly, mask, tr: pygeoops_centerline(vectorize(mask, tr)),
        ),
        'skeletonize + prune': (
            lambda poly, mask, tr: skeleton(burn(poly, mask, tr), tr),
            lambda poly, mask, tr: skeleton(mask, tr),
        ),
        'centerline (Voronoi)': (
            lambda poly, mask, tr: voronoi(poly),
            lambda poly, mask, tr: voronoi(vectorize(mask, tr)),
        ),
    }
    if with_cuda:
        out[CUDA] = (
            lambda poly, mask, tr: crestline(poly, resolution=1.0, device='cuda'),
            lambda poly, mask, tr: crestline(mask, transform=tr, device='cuda'),
        )

    return out


def skipped(name, kind):
    return (name.startswith('centerline') and kind == 'roads-large') or (
        not name.startswith('crestline') and kind == 'roads-xl'
    )


def run(plan, ms, repeat):
    total = 2 * sum(not skipped(name, kind) for kind, _ in plan for name in ms)
    rows, cpu = [], {}

    with tqdm(total=total, unit='run') as bar:
        for kind, seed in plan:
            lines, half, poly = make_case(kind, seed)
            mask, tr = noisy_mask(poly, seed)
            n_true, n_holes = true_dead_ends(lines, half), holes(poly)

            for name, (on_poly, on_mask) in ms.items():
                if skipped(name, kind):
                    continue

                for source, fn in (('polygon', on_poly), ('noisy mask', on_mask)):
                    bar.set_description(f'{kind} s{seed} {name} ({source})')
                    fn(poly, mask, tr)

                    best = np.inf
                    for _ in range(repeat if not name.startswith('centerline') else 1):
                        started = time.perf_counter()
                        out = fn(poly, mask, tr)
                        best = min(best, time.perf_counter() - started)

                    if name == 'crestline':
                        cpu[source] = out

                    pred = as_lines(out)
                    row = {
                        'case': kind,
                        'seed': seed,
                        'source': source,
                        'method': name,
                        'time': best,
                        'mpx': mask.size / 1e6,
                    }
                    row.update(
                        score(pred, lines),
                        dead=dead_ends(pred),
                        true_dead=n_true,
                        loops=loops(pred),
                        holes=n_holes,
                    )
                    if name == CUDA:
                        row['same'] = bool(out.equals_exact(cpu[source], 0))

                    rows.append(row)
                    bar.set_postfix(f5=f'{row["f5"]:.3f}', dead=f'{row["dead"]}/{n_true}', time=f'{best:.3f}s')
                    bar.update()

    return rows


def figure(rows, ms, case='roads', seed=1, window=(470, 270, 870, 670)):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FixedLocator
    from matplotlib.ticker import NullLocator

    surface, land, hair, axis = '#fcfcfb', '#e4e2db', '#e1e0d9', '#c3c2b7'
    ink, ink2, muted = '#0b0b0b', '#52514e', '#898781'
    blue, orange, other = '#2a78d6', '#eb6834', '#aeaca5'

    installed = {font.name for font in font_manager.fontManager.ttflist}
    fonts = ['Inter', 'DejaVu Sans'] if 'Inter' in installed else ['DejaVu Sans']
    plt.rcParams.update({'font.family': fonts, 'font.size': 9, 'text.color': ink2})

    names = [method for method in ms if method != CUDA]
    lines, half, poly = make_case(case, seed)
    mask, tr = noisy_mask(poly, seed)

    x0, y0, x1, y1 = window
    c0, r0 = ~tr * (x0, y1)
    c1, r1 = ~tr * (x1, y0)
    crop = mask[int(r0) : int(r1), int(c0) : int(c1)]

    fig_w, fig_h, left, right, gap = 12.0, 7.5, 0.45, 0.25, 0.14
    size = (fig_w - left - right - gap * (len(names) - 1)) / len(names)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=surface)

    def at(pos_x, pos_y, box_w=0.0, box_h=0.0):
        return [pos_x / fig_w, 1 - (pos_y + box_h) / fig_h, box_w / fig_w, box_h / fig_h]

    def text(pos_x, pos_y, label, **kwargs):
        fig.text(pos_x / fig_w, 1 - pos_y / fig_h, label, **kwargs)

    text(
        left,
        0.5,
        f'Noisy road mask, {x1 - x0:.0f} × {y1 - y0:.0f} m window',
        fontsize=14,
        color=ink,
        weight='semibold',
    )
    text(
        left,
        0.78,
        f'Scores in each title are for the whole network; it has {true_dead_ends(lines, half)} true dead ends.',
        fontsize=10,
    )

    keys = fig_w - right
    text(keys, 0.78, 'centerline', fontsize=9, ha='right')
    fig.lines.append(
        plt.Line2D(
            [(keys - 0.95) / fig_w, (keys - 0.72) / fig_w],
            [1 - 0.745 / fig_h] * 2,
            color=blue,
            lw=2,
            transform=fig.transFigure,
        )
    )
    text(keys - 1.12, 0.78, 'dead end', fontsize=9, ha='right')
    fig.lines.append(
        plt.Line2D(
            [(keys - 1.82) / fig_w],
            [1 - 0.745 / fig_h],
            marker='o',
            ms=6.5,
            color=orange,
            mec=surface,
            mew=1.2,
            transform=fig.transFigure,
        )
    )

    for index, name in enumerate(names):
        pred = as_lines(ms[name][1](poly, mask, tr))
        ends = end_points(pred)
        panel_x = left + index * (size + gap)

        ax = fig.add_axes(at(panel_x, 1.55, size, size))
        ax.imshow(crop, cmap=ListedColormap([surface, land]), extent=(x0, x1, y0, y1), interpolation='nearest')
        for line in pred:
            ax.plot(*line.xy, color=blue, lw=1.4, solid_capstyle='round', solid_joinstyle='round')

        inside = (ends[:, 0] > x0) & (ends[:, 0] < x1) & (ends[:, 1] > y0) & (ends[:, 1] < y1)
        ax.plot(*ends[inside].T, 'o', ms=5.5, color=orange, mec=surface, mew=1.1, zorder=3)

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(hair)

        text(panel_x, 1.22, name, fontsize=11, color=ink, weight='semibold')
        text(panel_x, 1.45, f'F1 {score(pred, lines)["f5"]:.2f} · {dead_ends(pred):,} dead ends', fontsize=9.5)

    noisy = [row for row in rows if row['source'] == 'noisy mask' and row['case'] in ('roads', 'mixed', 'river')]
    timed = [row for row in rows if row['source'] == 'noisy mask' and row['case'] == 'roads']
    true_dead = np.mean([row['true_dead'] for row in noisy if row['method'] == names[0]])
    charts = [
        (
            'F1 @ 5 m',
            'higher is better',
            [np.mean([row['f5'] for row in noisy if row['method'] == method]) for method in names],
            False,
            '{:.3f}',
        ),
        (
            'Dead ends',
            f'closer to the true {true_dead:.1f} is better',
            [np.mean([row['dead'] for row in noisy if row['method'] == method]) for method in names],
            True,
            '{:,.0f}',
        ),
        (
            'Runtime, 1.2 Mpx mask',
            'lower is better',
            [1000 * np.mean([row['time'] for row in timed if row['method'] == method]) for method in names],
            True,
            '{:,.0f} ms',
        ),
    ]

    top, names_w, chart_gap, chart_h = 5.45, 1.45, 0.4, 1.6
    chart_w = (fig_w - left - names_w - right - 2 * chart_gap) / 3
    text(
        left,
        top - 0.75,
        'Noisy masks, means over the 1 km road networks, mixed-width road networks and rivers',
        fontsize=10,
    )

    ys = np.arange(len(names))[::-1]
    for index, (title, hint, values, log, fmt) in enumerate(charts):
        chart_x = left + names_w + index * (chart_w + chart_gap)
        ax = fig.add_axes(at(chart_x, top, chart_w, chart_h))
        ax.set_facecolor(surface)

        lo, hi = (min(values) / 2.2, max(values) * 4.5) if log else (0.4, 1.04)
        if log:
            ax.set_xscale('log')

        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.55, len(names) - 0.45)
        for row_y in ys:
            ax.axhline(row_y, color=hair, lw=0.8, zorder=0)

        if title == 'Dead ends':
            ax.axvline(true_dead, color=muted, lw=0.9, zorder=1)

        for row_y, value, name in zip(ys, values, names):
            lead = name == 'crestline'
            ax.plot([value], [row_y], 'o', ms=8.5, color=blue if lead else other, mec=surface, mew=1.8, zorder=3)

            below = title == 'Dead ends' and value < true_dead
            shift = (1 / 1.2 if below else 1.2) if log else 0.02
            ax.text(
                value * shift if log else value + shift,
                row_y,
                fmt.format(value),
                va='center',
                ha='right' if below else 'left',
                fontsize=9,
                color=ink if lead else ink2,
                weight='semibold' if lead else 'normal',
            )

        ax.set_yticks([])
        if index == 0:
            for row_y, name in zip(ys, names):
                ax.text(
                    -0.04,
                    row_y,
                    name,
                    transform=ax.get_yaxis_transform(),
                    ha='right',
                    va='center',
                    fontsize=9.5,
                    color=ink if name == 'crestline' else ink2,
                )

        ax.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=muted, pad=5)
        if log:
            ticks = [tick for tick in (1, 10, 100, 1000, 10000) if lo <= tick <= hi]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([f'{tick:,}' for tick in ticks])
        else:
            ax.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)

        ax.spines['bottom'].set_color(axis)
        text(chart_x, top - 0.38, title, fontsize=10.5, color=ink, weight='semibold')
        text(chart_x, top - 0.16, hint, fontsize=8.5, color=muted)

    fig.savefig(OUTPUT_DIR / 'synthetic_benchmark.png', dpi=200, facecolor=surface)
    plt.close(fig)


def cuda_figure(rows, gpu):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FixedLocator
    from matplotlib.ticker import NullLocator

    surface, hair, axis = '#fcfcfb', '#e1e0d9', '#c3c2b7'
    ink, ink2, muted = '#0b0b0b', '#52514e', '#898781'
    blue, other = '#2a78d6', '#aeaca5'

    installed = {font.name for font in font_manager.fontManager.ttflist}
    fonts = ['Inter', 'DejaVu Sans'] if 'Inter' in installed else ['DejaVu Sans']
    plt.rcParams.update({'font.family': fonts, 'font.size': 9, 'text.color': ink2})

    names = {
        'river': 'rivers',
        'roads': 'road networks',
        'mixed': 'mixed-width roads',
        'roads-large': '4 km road network',
        'roads-xl': '8 km road network',
    }
    mpx = {
        case: np.mean([row['mpx'] for row in rows if row['case'] == case])
        for case in {row['case'] for row in rows if row['method'] == CUDA}
    }
    cases = sorted(mpx, key=mpx.get)
    same = [row['same'] for row in rows if row['method'] == CUDA]

    def mean(case, source, method):
        return 1000 * np.mean(
            [row['time'] for row in rows if (row['case'], row['source'], row['method']) == (case, source, method)]
        )

    fig_w, fig_h, left, right, names_w, gap = 12.0, 2.0 + 0.42 * len(cases), 0.45, 0.25, 2.1, 0.5
    chart_w, chart_h, top = (fig_w - left - names_w - right - gap) / 2, 0.42 * len(cases), 1.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=surface)

    def text(pos_x, pos_y, label, **kwargs):
        fig.text(pos_x / fig_w, 1 - pos_y / fig_h, label, **kwargs)

    verdict = 'same lines in all' if all(same) else f'different lines in {same.count(False)} of'
    text(left, 0.5, 'crestline on the CPU and on CUDA', fontsize=14, color=ink, weight='semibold')
    text(
        left,
        0.78,
        f'CPU with {os.cpu_count()} cores · {gpu} · best runtime per shape, '
        f'averaged over seeds · {verdict} {len(same)} runs',
        fontsize=10,
    )

    for index, (name, color) in enumerate((('CUDA', blue), ('CPU', other))):
        key_x = fig_w - right - index * 0.95
        text(key_x, 0.78, name, fontsize=9, ha='right')
        fig.lines.append(
            plt.Line2D(
                [(key_x - 0.5) / fig_w],
                [1 - 0.745 / fig_h],
                marker='o',
                ms=7,
                color=color,
                mec=surface,
                mew=1.4,
                transform=fig.transFigure,
            )
        )

    ys = np.arange(len(cases))[::-1]
    for index, source in enumerate(('polygon', 'noisy mask')):
        on_cpu = [mean(case, source, 'crestline') for case in cases]
        on_gpu = [mean(case, source, CUDA) for case in cases]
        chart_x = left + names_w + index * (chart_w + gap)

        ax = fig.add_axes([chart_x / fig_w, 1 - (top + chart_h) / fig_h, chart_w / fig_w, chart_h / fig_h])
        ax.set_facecolor(surface)
        ax.set_xscale('log')
        lo, hi = min(on_cpu + on_gpu) / 2.5, max(on_cpu + on_gpu) * 7
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.55, len(cases) - 0.45)

        for row_y, cpu_ms, gpu_ms in zip(ys, on_cpu, on_gpu):
            ax.axhline(row_y, color=hair, lw=0.8, zorder=0)
            ax.plot([cpu_ms, gpu_ms], [row_y, row_y], color=axis, lw=2, solid_capstyle='round', zorder=1)
            ax.plot([cpu_ms], [row_y], 'o', ms=8.5, color=other, mec=surface, mew=1.8, zorder=3)
            ax.plot([gpu_ms], [row_y], 'o', ms=8.5, color=blue, mec=surface, mew=1.8, zorder=3)

            speed = f'{cpu_ms / gpu_ms:.1f}× faster' if cpu_ms >= gpu_ms else f'{gpu_ms / cpu_ms:.1f}× slower'
            if max(cpu_ms, gpu_ms) < 1.05 * min(cpu_ms, gpu_ms):
                speed = 'same'

            ax.text(
                max(cpu_ms, gpu_ms) * 1.25,
                row_y,
                f'{gpu_ms:,.0f} ms · {speed}',
                va='center',
                fontsize=9,
                color=ink,
            )

        if index == 0:
            for row_y, case in zip(ys, cases):
                ax.text(
                    -0.04,
                    row_y,
                    f'{names.get(case, case)} ({mpx[case]:.1f} Mpx)',
                    transform=ax.get_yaxis_transform(),
                    ha='right',
                    va='center',
                    fontsize=9.5,
                )

        ax.set_yticks([])
        ticks = [tick for tick in (1, 10, 100, 1000, 10000) if lo <= tick <= hi]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f'{tick:,} ms' for tick in ticks])
        ax.tick_params(axis='x', length=0, labelsize=8.5, labelcolor=muted, pad=5)

        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)

        ax.spines['bottom'].set_color(axis)
        text(chart_x, top - 0.22, f'Runtime, {source}', fontsize=10.5, color=ink, weight='semibold')

    fig.savefig(OUTPUT_DIR / 'synthetic_benchmark_cuda.png', dpi=200, facecolor=surface)
    plt.close(fig)


def table(rows):
    summary = (
        pd.DataFrame(rows)
        .groupby(['case', 'source', 'method'], sort=False)
        .agg(
            mpx=('mpx', 'mean'),
            f1_5m=('f5', 'mean'),
            f1_2m=('f2', 'mean'),
            offset_m=('offset', 'mean'),
            dead_ends=('dead', 'mean'),
            true_dead_ends=('true_dead', 'mean'),
            loops=('loops', 'mean'),
            holes=('holes', 'mean'),
            runtime_ms=('time', 'mean'),
        )
        .reset_index()
    )
    summary['runtime_ms'] *= 1000

    digits = {
        'mpx': 2,
        'f1_5m': 4,
        'f1_2m': 4,
        'offset_m': 3,
        'dead_ends': 1,
        'true_dead_ends': 1,
        'loops': 1,
        'holes': 1,
        'runtime_ms': 2,
    }
    summary.round(digits).to_csv(OUTPUT_DIR / 'synthetic_benchmark_results.csv', index=False, lineterminator='\n')


def main(
    cases=None,
    seeds=None,
    large_seeds=None,
    repeat: int = 3,
):
    if cases is None:
        cases = ['roads', 'mixed', 'river', 'roads-large', 'roads-xl']

    if seeds is None:
        seeds = [0, 1, 2, 3, 4, 5]

    if large_seeds is None:
        large_seeds = [1, 2]

    gpu, reason = cuda_device()
    tqdm.write(f'CUDA: {gpu}' if gpu else f'CUDA: off ({reason}), running on the CPU only')
    with_cuda = bool(gpu)
    ms = methods(with_cuda)

    plan = [(case, seed) for case in cases for seed in (large_seeds if case.startswith('roads-') else seeds)]
    rows = run(plan, ms, repeat)

    table(rows)

    if 'roads' in cases:
        figure(rows, ms)

    if with_cuda:
        cuda_figure(rows, gpu)


if __name__ == '__main__':
    CLI(main)
