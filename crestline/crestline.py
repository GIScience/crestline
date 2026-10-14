import math
from builtins import float

import numpy as np
import shapely
from numba import jit
from numba import prange
from scipy.spatial import cKDTree
from shapely import MultiLineString
from shapely import Polygon
from shapely import linestrings
from shapely import multilinestrings
from shapely.geometry.multipolygon import MultiPolygon


@jit(parallel=True, cache=True)
def _grid(mask: np.ndarray):
    """
    Build the padded flood grid in one pass.

    :param mask: 2D bool mask (True inside the polygon)
    :return: int8 grid padded by 2 px: 1 = outside (source of the inward flood), 2 = inside (target)
    """
    h, w = mask.shape
    state = np.ones((h + 4, w + 4), np.int8)
    for i in prange(h):
        for j in range(w):
            if mask[i, j]:
                state[i + 2, j + 2] = 2

    return state


@jit(parallel=True, cache=True)
def _flood(state: np.ndarray) -> np.ndarray:
    """
    Flood all target pixels from the sources, one ring per step.

    :param state: world state where: 0 = wall; 1 = source; 2 = target.
    :return: level array
    """
    h, w = state.shape
    level = np.zeros((h, w), np.int32)
    queue = np.empty(h * w, np.int32)
    touch = np.zeros((h, w), np.bool_)

    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            if state[i, j] == 1:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        touch[i, j] |= state[di + i, dj + j] == 2

    sources = np.flatnonzero(touch)
    n = len(sources)
    queue[:n] = sources

    state_r, level_r = state.ravel(), level.ravel()
    ring = np.array([-w, -1, 1, w, -w - 1, -w + 1, w - 1, w + 1])
    start, mid, step, empty = 0, 0, 0, 0

    while empty < 2:
        end = n
        cross = step % 2 == 0

        for k in range(mid if cross else start, end):
            for o in ring[4 if k < mid else 0 : 4 if cross else 8]:
                q = queue[k] + o

                if state_r[q] == 2 and level_r[q] == 0:
                    level_r[q] = step + 1
                    queue[n] = q
                    n += 1

        empty = empty + 1 if n == end else 0
        start, mid, step = mid, end, step + 1

    return level


@jit(parallel=True, cache=True)
def _crests(depth: np.ndarray):
    """
    Mark the crest of the inward flood steps.

    :param depth: inward flood steps (from the polygon perspective), padded by 2 px
    :return: grid with walls and sources marked
    """
    h, w = depth.shape
    state = np.zeros((h, w), np.int8)

    for i in prange(2, h - 2):
        for j in range(2, w - 2):
            d = depth[i, j]

            if d:
                top = True
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        top &= depth[di + i, dj + j] <= d

                state[i, j] = 1 if top else 2

    return state


@jit(parallel=True, cache=True)
def _ratio(depth: np.ndarray, dist: np.ndarray):
    """
    Combine floods into the 0 -> 1 center raster and strip the padding.

    :param depth: 1 = inward flood steps and 0 = outside, padded by 2 px
    :param dist: flood steps from the crest, padded by 2 px
    :return: ratio
    """
    h, w = depth.shape
    out = np.zeros((h - 4, w - 4), np.float32)

    for i in prange(h - 4):
        for j in range(w - 4):
            climb = depth[i + 2, j + 2] - 1
            total = climb + dist[i + 2, j + 2]

            if climb >= 0 and total > 0:
                out[i, j] = climb / total

    return out


@jit(parallel=True, cache=True)
def _scan_fill(x0: int, y0: int, x1: int, y1: int, h: int, w: int):
    """
    Rasterize polygon edges with an even-odd scanline fill.

    :param x0, y0, x1, y1: edge end points in pixel units (x = column, y = row, pointing down)
    :param h: grid height
    :param w: grid width

    :return: bool mask
    """
    count = np.zeros(h + 1, np.int64)
    for e in range(len(x0)):
        lo = max(int(np.ceil(min(y0[e], y1[e]) - 0.5)), 0)
        hi = min(int(np.ceil(max(y0[e], y1[e]) - 0.5)), h)
        for r in range(lo, hi):
            count[r + 1] += 1

    start = np.cumsum(count)
    xs = np.empty(start[-1], np.float64)
    fill = start[:-1].copy()
    for e in range(len(x0)):
        lo = max(int(np.ceil(min(y0[e], y1[e]) - 0.5)), 0)
        hi = min(int(np.ceil(max(y0[e], y1[e]) - 0.5)), h)
        for r in range(lo, hi):
            xs[fill[r]] = x0[e] + (r + 0.5 - y0[e]) * (x1[e] - x0[e]) / (y1[e] - y0[e])
            fill[r] += 1

    mask = np.zeros((h, w), np.bool_)
    for r in prange(h):
        row = np.sort(xs[start[r] : start[r + 1]])
        for k in range(0, len(row) - 1, 2):
            c0 = max(int(np.ceil(row[k] - 0.5)), 0)
            c1 = min(int(np.ceil(row[k + 1] - 0.5)), w)
            for c in range(c0, c1):
                mask[r, c] = True

    return mask


def _rasterize(polygon: Polygon | MultiPolygon, resolution: float):
    """
    Burn a polygon into a bool mask with 1 px of margin.

    :param polygon: shapely Polygon or MultiPolygon
    :param resolution: pixel size in map units
    :return: mask and the map coordinates (left, top) of its upper-left corner
    """
    minx, miny, maxx, maxy = polygon.bounds
    left, top = minx - resolution, maxy + resolution
    h = int(np.ceil((maxy - miny) / resolution)) + 2
    w = int(np.ceil((maxx - minx) / resolution)) + 2

    rings = shapely.get_rings(shapely.get_parts(polygon))
    xy, ring = shapely.get_coordinates(rings, return_index=True)
    px, py = (xy[:, 0] - left) / resolution, (top - xy[:, 1]) / resolution
    same = ring[1:] == ring[:-1]

    mask = _scan_fill(px[:-1][same], py[:-1][same], px[1:][same], py[1:][same], h, w)
    return mask, (left, top)


@jit(cache=True)
def _thin(rows: np.ndarray, cols: np.ndarray, h: int, w: int, radius: int):
    """
    Greedy thinning on a raster: keep a point, block everything within its radius, move on.

    :param rows, cols: crest pixels sorted tallest first
    :param h, w: raster size
    :param radius: suppression radius of each pixel
    :return: bool array, True for kept points
    """
    taken = np.zeros((h, w), np.bool_)
    keep = np.zeros(len(rows), np.bool_)

    for k in range(len(rows)):
        r, c = rows[k], cols[k]
        if taken[r, c]:
            continue

        keep[k] = True
        r2 = radius[k] * radius[k]
        reach = int(radius[k])

        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                if dr * dr + dc * dc <= r2 and 0 <= r + dr < h and 0 <= c + dc < w:
                    taken[dr + r, dc + c] = True

    return keep


@jit(inline='always')
def _dist(points: np.ndarray, i: int, j: int, eps: float = 1e-12):
    return math.hypot(float(points[i, 0] - points[j, 0]), float(points[i, 1] - points[j, 1])) + eps * (i + j)


@jit(parallel=True, cache=True)
def _on_ridge(points: np.ndarray, pairs: np.ndarray, depth: np.ndarray, dip: float, samples: int):
    """
    Ridge test for candidate links.

    :param points: (n, 2) row, col of the thinned crest points
    :param pairs: (m, 2) candidate links
    :param depth: inward flood steps, unpadded
    :param dip: depth along the link must stay above dip * the depth at its lower end
    :param samples: number of samples along each link
    :return: bool mask
    """
    mask = np.zeros(len(pairs), np.bool_)

    for e in prange(len(pairs)):
        r0, c0 = points[pairs[e, 0], 0], points[pairs[e, 0], 1]
        r1, c1 = points[pairs[e, 1], 0], points[pairs[e, 1], 1]
        low = dip * min(depth[r0, c0], depth[r1, c1])

        masked = True
        for s in range(samples):
            u = s / (samples - 1)
            r = int(np.rint(r0 + u * (r1 - r0)))
            c = int(np.rint(c0 + u * (c1 - c0)))

            if depth[r, c] < low:
                masked = False
                break

        mask[e] = masked

    return mask


@jit(cache=True)
def _prune(points: np.ndarray, pairs: np.ndarray):
    """
    Drop shortcut links.

    :param points: (n, 2) row, col of the thinned crest points
    :param pairs: (m, 2) links that passed the ridge test
    :return: bool array, True for links to keep
    """
    n, m = len(points), len(pairs)

    degree = np.zeros(n + 1, np.int64)
    for e in range(m):
        degree[pairs[e, 0] + 1] += 1
        degree[pairs[e, 1] + 1] += 1

    start = np.cumsum(degree)
    nbr = np.empty(start[-1], np.int64)
    fill = start[:-1].copy()
    for e in range(m):
        i, j = pairs[e, 0], pairs[e, 1]
        nbr[fill[i]] = j
        fill[i] += 1
        nbr[fill[j]] = i
        fill[j] += 1

    keep = np.ones(m, np.bool_)
    for e in range(m):
        i, j = pairs[e, 0], pairs[e, 1]
        d_ij = _dist(points, i, j)

        for a in range(start[i], start[i + 1]):
            k = nbr[a]
            if k == j or _dist(points, i, k) >= d_ij:
                continue

            for b in range(start[k], start[k + 1]):
                q = nbr[b]
                if q == i or _dist(points, k, q) >= d_ij:
                    continue
                if q == j:
                    keep[e] = False
                    break

                for c in range(start[j], start[j + 1]):
                    if nbr[c] == q and _dist(points, q, j) < d_ij:
                        keep[e] = False
                        break

                if not keep[e]:
                    break

            if not keep[e]:
                break

    return keep


@jit(cache=True)
def _despur(points: np.ndarray, pairs: np.ndarray, heights: np.ndarray, near: np.ndarray, spur: float, tip: float):
    """
    Remove spurs leaf by leaf.

    :param points: row, col of the thinned crest points
    :param pairs: links that survived _prune
    :param heights:  inward flood depth at each point
    :param near: pairs of points close enough to cover one another
    :param spur: disc scale
    :param tip: depth ratio below which a dead end counts as shallow
    :return: bool array
    """
    n, m = len(points), len(pairs)

    covered = np.zeros(n, np.bool_)
    for e in range(len(near)):
        for v, u in ((near[e, 0], near[e, 1]), (near[e, 1], near[e, 0])):
            if heights[v] < tip * heights[u] and _dist(points, u, v) + heights[v] <= spur * heights[u]:
                covered[v] = True

    count = np.zeros(n + 1, np.int64)
    for e in range(m):
        count[pairs[e, 0] + 1] += 1
        count[pairs[e, 1] + 1] += 1

    start = np.cumsum(count)
    nbr = np.empty(start[-1], np.int64)
    edge = np.empty(start[-1], np.int64)
    fill = start[:-1].copy()

    for e in range(m):
        i, j = pairs[e, 0], pairs[e, 1]

        nbr[fill[i]], edge[fill[i]] = j, e
        fill[i] += 1

        nbr[fill[j]], edge[fill[j]] = i, e
        fill[j] += 1

    degree = count[1:].copy()
    alive = np.ones(m, np.bool_)
    leaves = [v for v in range(n) if degree[v] == 1]

    while leaves:
        v = leaves.pop()
        if degree[v] != 1 or not covered[v]:
            continue

        for a in range(start[v], start[v + 1]):
            if alive[edge[a]]:
                alive[edge[a]] = False
                degree[v] -= 1
                degree[nbr[a]] -= 1

                if degree[nbr[a]] == 1:
                    leaves.append(nbr[a])

    return alive


def crestline(
    source,
    resolution: float | None = None,
    transform=None,
    simplify: float = 0.0,
    dip: float = 0.65,
    spur: float = 2.0,
    tip: float = 0.75,
) -> MultiLineString:
    """
    Trace the centerline of a polygon or a polygon mask.

    :param source: shapely Polygon / MultiPolygon, or a 2D bool mask (True inside)
    :param resolution: pixel size in map units; required for a polygon, ignored for a mask
    :param transform: affine pixel -> map for a mask input (e.g. rasterio's); None keeps pixel coords
    :param simplify: Douglas-Peucker tolerance in output units, 0 = off
    :param dip: keep a link only if the flood depth along it stays above dip * its lower end
    :param spur: prune dead ends whose disc fits inside spur * the disc of a deeper crest point, 0 = off
    :param tip: ...but only if they are shallower than tip * its depth (real road ends stay deep)
    :return: centerline, empty if there is none
    """
    grid = None
    if isinstance(source, shapely.Geometry):
        if source.geom_type not in ('Polygon', 'MultiPolygon'):
            raise TypeError(f'expected a Polygon or MultiPolygon, got {source.geom_type}')
        if not resolution or resolution <= 0:
            raise ValueError('res (pixel size in map units) is required for a polygon')
        if source.is_empty:
            return MultiLineString()

        mask, (left, top) = _rasterize(source, resolution)
        grid = (left, top, resolution)
    else:
        mask = np.asarray(source, bool)
        if mask.ndim != 2:
            raise ValueError('mask must be 2D')

    depth = _flood(_grid(mask))
    crest = (_crests(depth) == 1) & (depth >= 2)
    depth = depth[2:-2, 2:-2]

    rows, cols = np.nonzero(crest[2:-2, 2:-2])
    if len(rows) < 2:
        return MultiLineString()

    height = depth[rows, cols]
    order = np.argsort(-height, kind='stable')
    rows, cols, height = rows[order], cols[order], height[order]
    keep = _thin(rows, cols, mask.shape[0], mask.shape[1], np.maximum(0.35 * height, 3))
    points, heights = np.c_[rows[keep], cols[keep]], height[keep]

    near = cKDTree(points).query_pairs(max(2, spur) * heights.max(), output_type='ndarray')
    span = np.hypot(*(points[near[:, 0]] - points[near[:, 1]]).T)
    pairs = near[span <= 2 * np.maximum(heights[near[:, 0]], heights[near[:, 1]])]
    pairs = pairs[_on_ridge(points, pairs, depth, dip, 2 * heights.max() + 2)]
    pairs = pairs[_prune(points, pairs)]

    if spur > 0:
        pairs = pairs[_despur(points, pairs, heights, near, spur, tip)]

    if len(pairs) == 0:
        return MultiLineString()

    shift, same = np.zeros(points.shape), np.zeros(len(points))
    for d in np.argwhere(np.ones((3, 3))) - 1:
        r, c = (points + d).T
        hit = crest[r + 2, c + 2] & (depth[r, c] == heights)
        shift, same = shift + hit[:, None] * d, same + hit

    rc = points + 0.5 + shift / same[:, None]

    xy = np.c_[rc[:, 1], rc[:, 0]]
    if grid is not None:
        left, top, resolution = grid
        xy = np.c_[left + xy[:, 0] * resolution, top - xy[:, 1] * resolution]
    elif transform is not None:
        xy = np.c_[transform * tuple(xy.T)]

    lines = shapely.line_merge(multilinestrings(linestrings(xy[pairs])))
    if simplify:
        lines = lines.simplify(simplify)

    return lines if isinstance(lines, MultiLineString) else MultiLineString([lines])
