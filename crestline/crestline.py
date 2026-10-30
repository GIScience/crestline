import math
from builtins import float

import numpy as np
import shapely
from affine import Affine
from numba import jit
from numba import prange
from pyproj import CRS
from pyproj import Transformer
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from scipy.spatial import cKDTree
from shapely import MultiLineString
from shapely import Polygon
from shapely import linestrings
from shapely import multilinestrings
from shapely import transform as tx
from shapely.geometry.multipolygon import MultiPolygon


def _project(polygon: Polygon | MultiPolygon, crs: int | None) -> tuple[Polygon | MultiPolygon, CRS | None]:
    if crs is None or CRS.from_epsg(crs).is_projected:
        return polygon, crs if crs is None else CRS.from_epsg(crs)

    crs = CRS.from_epsg(crs)
    lon, lat = polygon.centroid.coords[0]

    proj_crs = ProjectedCRS(
        name=f'aeqd {lat:.6f} {lon:.6f}',
        conversion=AzimuthalEquidistantConversion(latitude_natural_origin=lat, longitude_natural_origin=lon),
        geodetic_crs=crs,
    )

    return _reproject_crs(polygon, crs, proj_crs), proj_crs


def _reproject_crs(geom: shapely.Geometry, src_crs: CRS, dst_crs: CRS) -> shapely.Geometry:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tx(geom, transformer.transform, interleaved=False)


@jit(parallel=True, cache=True)
def _grid(mask: np.ndarray):
    """
    Build the padded flood grid in one pass.

    :param mask: 2D bool mask (True inside the polygon)
    :return: int8 grid padded by 2 px: 1 = outside (source of the inward flood), 2 = inside (target)
    """
    h, w = mask.shape
    state = np.empty((h + 4, w + 4), np.int8)

    for i in prange(h + 4):
        state[i] = 1

        if 2 <= i < h + 2:
            for j in range(w):
                state[i, j + 2] += mask[i - 2, j]

    return state


@jit(parallel=True, cache=True)
def _flood(state: np.ndarray, neighbourhood: int = 0) -> np.ndarray:
    """
    Flood all target pixels from the sources, one ring per step.

    :param state: world state where: 0 = wall; 1 = source; 2 = target.
    :param neighbourhood: 4 or 8 to flood with that neighbourhood only, 0 to alternate them (octagonal)
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
        cross = neighbourhood == 4 or (neighbourhood != 8 and step % 2 == 0)

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


@jit(inline='always')
def _top_aux(depth: np.ndarray, i: int, j: int, reach: int) -> bool:
    for r in (min(1, reach), reach):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if depth[di + i, dj + j] > depth[i, j]:
                    return False

    return True


@jit(parallel=True, cache=True)
def _crests(depth: np.ndarray, reach: int = 2):
    """
    Mark the crest of the inward flood steps.

    :param depth: inward flood steps (from the polygon perspective), padded by 2 px
    :param reach: a crest pixel has nothing deeper within this many pixels, at most 2
    :return: grid with walls and sources marked
    """
    h, w = depth.shape
    state = np.zeros((h, w), np.int8)

    for i in prange(2, h - 2):
        for j in range(2, w - 2):
            if depth[i, j]:
                state[i, j] = 1 if _top_aux(depth, i, j, reach) else 2

    return state


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


def _links(points: np.ndarray, heights: np.ndarray, near: np.ndarray) -> np.ndarray:
    """
    Keep the candidate links no longer than twice the depth of their deeper end.

    :param points: (n, 2) row, col of the thinned crest points
    :param heights: inward flood depth at each point
    :param near: (m, 2) candidate links
    :return: the links short enough for their depth
    """
    span = np.hypot(*(points[near[:, 0]] - points[near[:, 1]]).T)
    return near[span <= 2 * np.maximum(heights[near[:, 0]], heights[near[:, 1]])]


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
def _prune(points: np.ndarray, pairs: np.ndarray, cycle: int = 4):
    """
    Drop shortcut links.

    :param points: (n, 2) row, col of the thinned crest points
    :param pairs: (m, 2) links that passed the ridge test
    :param cycle: drop the longest link of every cycle of up to this many links: 4, 3 (triangles only) or 0 (off)
    :return: bool array, True for links to keep
    """
    n, m = len(points), len(pairs)
    if cycle < 3:
        return np.ones(m, np.bool_)

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
                if cycle < 4:
                    continue

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
def _faces(points: np.ndarray, pairs: np.ndarray):
    """
    Trace the faces of the planar link graph.

    :param points: (n, 2) row, col of the thinned crest points
    :param pairs: (m, 2) links
    :return: face of every half link (2e runs pairs[e, 0] -> pairs[e, 1], 2e + 1 back), the half link after it on
        its face, and the signed area of every face (> 0 inside a loop)
    """
    n, m = len(points), len(pairs)

    count = np.zeros(n + 1, np.int64)
    for e in range(m):
        count[pairs[e, 0] + 1] += 1
        count[pairs[e, 1] + 1] += 1

    start = np.cumsum(count)
    half, angle = np.empty(2 * m, np.int64), np.empty(2 * m, np.float64)
    fill = start[:-1].copy()
    for h in range(2 * m):
        i, j = pairs[h // 2, h % 2], pairs[h // 2, 1 - h % 2]
        a, k = math.atan2(points[j, 0] - points[i, 0], points[j, 1] - points[i, 1]), fill[i]
        while k > start[i] and angle[k - 1] > a:
            half[k], angle[k] = half[k - 1], angle[k - 1]
            k -= 1
        half[k], angle[k] = h, a
        fill[i] += 1

    slot = np.empty(2 * m, np.int64)
    for k in range(2 * m):
        slot[half[k]] = k

    face, after = np.full(2 * m, -1, np.int64), np.empty(2 * m, np.int64)
    area = np.zeros(2 * m + 1, np.float64)
    f = 0
    for h0 in range(2 * m):
        if face[h0] >= 0:
            continue

        h = h0
        while face[h] < 0:
            i, j = pairs[h // 2, h % 2], pairs[h // 2, 1 - h % 2]
            face[h] = f
            area[f] += 0.5 * (points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1])
            k = slot[h ^ 1] + 1
            after[h] = half[k if k < start[j + 1] else start[j]]
            h = after[h]
        f += 1

    return face, after, area[:f]


@jit(cache=True)
def _occluded(
    points: np.ndarray,
    pairs: np.ndarray,
    heights: np.ndarray,
    mask: np.ndarray,
    hole: float,
    loop: float = 2 / 3,
):
    """
    Open the loops around holes too small for the depth around them: an occluder, not a block.

    :param points: row, col of the thinned crest points
    :param pairs: links that survived _despur
    :param heights: inward flood depth at each point
    :param mask: 2D bool mask (True inside)
    :param hole: a loop stays if the background inside it covers (hole * its deepest point)^2 pixels...
    :param loop: ...or if it outgrows the disc of loop * its deepest point
    :return: bool array, True for links to keep; the path holding the shallowest link goes, from junction to junction
    """
    n, m = len(points), len(pairs)
    keep = np.ones(m, np.bool_)

    cut = True
    while cut:
        cut = False
        alive = np.flatnonzero(keep)
        face, after, area = _faces(points, pairs[alive])

        before, first, degree = np.empty_like(after), np.full(len(area), -1, np.int64), np.zeros(n, np.int64)
        ring = np.empty((len(after), 2), np.int64)
        for h in range(len(after)):
            before[after[h]] = h
            first[face[h]] = h if first[face[h]] < 0 else first[face[h]]
            degree[pairs[alive[h // 2], h % 2]] += 1

        for f in range(len(area)):
            if area[f] <= 0:
                continue

            size, deep, weak, link, intact, h = 0, 0, np.inf, -1, True, first[f]
            while size == 0 or h != first[f]:
                e = alive[h // 2]
                i, j = pairs[e, h % 2], pairs[e, 1 - h % 2]
                ring[size, 0], ring[size, 1], size = i, j, size + 1
                deep, intact = max(deep, heights[i]), intact and keep[e]
                if face[h ^ 1] != f and min(heights[i], heights[j]) < weak:
                    weak, link = min(heights[i], heights[j]), h
                h = after[h]

            if not intact or link < 0 or area[f] > np.pi * (loop * deep) ** 2:
                continue

            found, need, xs = 0, (hole * deep) ** 2, np.empty(size, np.float64)
            for r in range(points[ring[:size, 0], 0].min(), points[ring[:size, 0], 0].max() + 1):
                c = 0
                for k in range(size):
                    (r0, c0), (r1, c1) = points[ring[k, 0]], points[ring[k, 1]]
                    if (r0 > r) != (r1 > r):
                        xs[c], c = c0 + (r - r0) * (c1 - c0) / (r1 - r0), c + 1

                cross = np.sort(xs[:c])
                for a in range(0, c - 1, 2):
                    found += np.sum(~mask[r, int(np.ceil(cross[a])) : int(np.floor(cross[a + 1])) + 1])

                if found >= need:
                    break

            if found >= need:
                continue

            head, tail = link, link
            while degree[pairs[alive[head // 2], 1 - head % 2]] == 2 and after[head] != tail:
                head = after[head]
            while degree[pairs[alive[tail // 2], tail % 2]] == 2 and before[tail] != head:
                tail = before[tail]

            if pairs[alive[head // 2], 1 - head % 2] == pairs[alive[tail // 2], tail % 2]:
                continue

            h, cut = tail, True
            while True:
                e = alive[h // 2]
                keep[e] = False
                degree[pairs[e, 0]] -= 1
                degree[pairs[e, 1]] -= 1
                if h == head:
                    break
                h = after[h]

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


@jit(cache=True)
def _tips(points: np.ndarray, pairs: np.ndarray, heights: np.ndarray, crest: np.ndarray, depth: np.ndarray, tip: float):
    """
    Move every dead end to the far end of the crest it was thinned from.

    :param points: row, col of the thinned crest points, moved in place
    :param pairs: final links
    :param heights: inward flood depth at each point, updated in place
    :param crest: crest raster, padded by 2 px
    :param depth: inward flood steps, unpadded
    :param tip: the new end must stay deeper than tip * the old one
    """
    n, (h, w) = len(points), depth.shape
    degree, other = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for e in range(len(pairs)):
        i, j = pairs[e, 0], pairs[e, 1]
        degree[i], degree[j], other[i], other[j] = degree[i] + 1, degree[j] + 1, j, i

    for v in range(n):
        if degree[v] != 1:
            continue

        r, c = points[v, 0], points[v, 1]
        dr, dc = r - points[other[v], 0], c - points[other[v], 1]
        radius = max(0.35 * heights[v], 3.0)
        reach, best, end = int(radius), 0, (r, c)

        for i in range(max(r - reach, 0), min(r + reach + 1, h)):
            for j in range(max(c - reach, 0), min(c + reach + 1, w)):
                ahead = (i - r) * dr + (j - c) * dc
                inside = (i - r) ** 2 + (j - c) ** 2 <= radius * radius
                if inside and ahead > best and crest[i + 2, j + 2] and depth[i, j] >= tip * heights[v]:
                    best, end = ahead, (i, j)

        points[v, 0], points[v, 1] = end
        heights[v] = depth[end]


@jit(parallel=True, cache=True)
def _center(points: np.ndarray, heights: np.ndarray, crest: np.ndarray, depth: np.ndarray):
    """
    Center every point on the crest pixels of its own depth within a radius of sqrt(5) px.

    :param points: row, col of the thinned crest points
    :param heights: inward flood depth at each point
    :param crest: crest raster, padded by 2 px
    :param depth: inward flood steps, unpadded
    :return: (n, 2) row, col
    """
    rc = np.empty((len(points), 2), np.float64)
    for k in prange(len(points)):
        r, c, dr, dc, same = points[k, 0], points[k, 1], 0, 0, 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i * i + j * j <= 5 and crest[r + i + 2, c + j + 2] and depth[r + i, c + j] == heights[k]:
                    dr, dc, same = dr + i, dc + j, same + 1

        rc[k, 0], rc[k, 1] = r + 0.5 + dr / same, c + 0.5 + dc / same

    return rc


def crestline(
    source,
    epsg_crs: int | None = None,
    resolution: float | None = None,
    transform: Affine | None = None,
    simplify: float = 0.0,
    dip: float = 0.65,
    spur: float = 2.0,
    tip: float = 0.75,
    hole: float = 0.25,
    device: str = 'cpu',
) -> MultiLineString:
    """
    Trace the centerline of a polygon or a polygon mask.

    :param source: shapely Polygon / MultiPolygon, or a 2D bool mask (True inside)
    :param epsg_crs: coordinate reference system (int code)
    :param resolution: pixel size in map units; required for a polygon, ignored for a mask
    :param transform: affine pixel -> map for a mask input (e.g. rasterio's); None keeps pixel coords
    :param simplify: Douglas-Peucker tolerance in output units, 0 = off
    :param dip: keep a link only if the flood depth along it stays above dip * its lower end
    :param spur: prune dead ends whose disc fits inside spur * the disc of a deeper crest point, 0 = off
    :param tip: ...but only if they are shallower than tip * its depth (real road ends stay deep)
    :param hole: open loops around holes narrower than hole * the surrounding depth (occluders), 0 = off
    :param device: 'cpu', or 'cuda' to flood and find the crest on an NVIDIA GPU (needs numba-cuda); same result
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

        source, proj_crs = _project(source, epsg_crs)
        mask, (left, top) = _rasterize(source, resolution)
        grid = (left, top, resolution)
    else:
        mask = np.asarray(source, bool)
        if mask.ndim != 2:
            raise ValueError('mask must be 2D')

    if device == 'cuda':
        from .crestline_cuda import flood

        depth, crest = flood(mask)
    elif device == 'cpu':
        depth = _flood(_grid(mask))
        crest = (_crests(depth) == 1) & (depth >= 2)
    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

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
    pairs = _links(points, heights, near)
    pairs = pairs[_on_ridge(points, pairs, depth, dip, 2 * heights.max() + 2)]
    pairs = pairs[_prune(points, pairs)]

    if spur > 0:
        pairs = pairs[_despur(points, pairs, heights, near, spur, tip)]

    if hole > 0:
        pairs = pairs[_occluded(points, pairs, heights, mask, hole)]

    if len(pairs) == 0:
        return MultiLineString()

    _tips(points, pairs, heights, crest, depth, tip)
    rc = _center(points, heights, crest, depth)

    xy = np.c_[rc[:, 1], rc[:, 0]]
    if grid is not None:
        left, top, resolution = grid
        xy = np.c_[left + xy[:, 0] * resolution, top - xy[:, 1] * resolution]
    elif transform is not None:
        xy = np.c_[transform * tuple(xy.T)]

    lines = shapely.line_merge(multilinestrings(linestrings(xy[pairs])))
    if simplify:
        lines = lines.simplify(simplify)
    if epsg_crs is not None and CRS.from_epsg(epsg_crs).is_geographic:
        lines = _reproject_crs(lines, src_crs=proj_crs, dst_crs=CRS.from_epsg(epsg_crs))

    return lines if isinstance(lines, MultiLineString) else MultiLineString([lines])
