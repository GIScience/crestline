import math

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
    """
    Move a geographic polygon to an azimuthal equidistant CRS centred on it; projected input passes through.

    :return: polygon and its CRS
    """
    if crs is None:
        return polygon, None

    source_crs = CRS.from_epsg(crs)
    if source_crs.is_projected:
        return polygon, source_crs

    lon, lat = polygon.centroid.coords[0]
    local_crs = ProjectedCRS(
        name=f'aeqd {lat:.6f} {lon:.6f}',
        conversion=AzimuthalEquidistantConversion(latitude_natural_origin=lat, longitude_natural_origin=lon),
        geodetic_crs=source_crs,
    )

    return _reproject_crs(polygon, source_crs, local_crs), local_crs


def _reproject_crs(geom: shapely.Geometry, src_crs: CRS, dst_crs: CRS) -> shapely.Geometry:
    """
    Reproject a geometry between two CRSs.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tx(geom, transformer.transform, interleaved=False)


@jit(parallel=True, cache=True)
def _grid(mask: np.ndarray):
    """
    Pad the mask by 2 px into a flood grid.

    :param mask: 2D bool, True inside
    :return: int8 grid, 1 = outside (flood source), 2 = inside (flood target)
    """
    n_rows, n_cols = mask.shape
    state = np.empty((n_rows + 4, n_cols + 4), np.int8)

    for row in prange(n_rows + 4):
        state[row] = 1

        if 2 <= row < n_rows + 2:
            for col in range(n_cols):
                state[row, col + 2] += mask[row - 2, col]

    return state


@jit(parallel=True, cache=True)
def _flood(state: np.ndarray, neighbourhood: int = 0) -> np.ndarray:
    """
    Flood the targets from the sources one ring per step (breadth first).

    :param state: 0 = wall, 1 = source, 2 = target
    :param neighbourhood: 4 or 8 only, 0 alternates them (octagonal distance)
    :return: int32 step at which each target was reached, 0 elsewhere
    """
    n_rows, n_cols = state.shape
    level = np.empty((n_rows, n_cols), np.int32)
    queue = np.empty(n_rows * n_cols, np.int32)
    is_seed = np.zeros((n_rows, n_cols), np.bool_)

    for row in prange(n_rows):
        for col in range(n_cols):
            level[row, col] = -(state[row, col] == 2)

    for row in prange(1, n_rows - 1):
        for col in range(1, n_cols - 1):
            if state[row, col] == 1:
                for d_row in range(-1, 2):
                    for d_col in range(-1, 2):
                        is_seed[row, col] |= state[row + d_row, col + d_col] == 2

    seeds = np.flatnonzero(is_seed)
    queued = len(seeds)
    queue[:queued] = seeds

    flat_level = level.ravel()
    offsets = np.array([-n_cols, -1, 1, n_cols, -n_cols - 1, -n_cols + 1, n_cols - 1, n_cols + 1])
    older_ring, newest_ring, step, idle_steps = 0, 0, 0, 0

    while idle_steps < 2:
        ring_end = queued
        cross_step = neighbourhood == 4 or (neighbourhood != 8 and step % 2 == 0)

        for slot in range(newest_ring if cross_step else older_ring, ring_end):
            for offset in offsets[4 if slot < newest_ring else 0 : 4 if cross_step else 8]:
                pixel = queue[slot] + offset

                if flat_level[pixel] == -1:
                    flat_level[pixel] = step + 1
                    queue[queued] = pixel
                    queued += 1

        idle_steps = idle_steps + 1 if queued == ring_end else 0
        older_ring, newest_ring, step = newest_ring, ring_end, step + 1

    level[level < 0] = 0

    return level


@jit(inline='always')
def _top_aux(depth: np.ndarray, i: int, j: int, reach: int) -> bool:
    """
    True if nothing within reach px of (i, j) is deeper; the 3x3 block is checked first to fail fast.
    """
    for radius in (min(1, reach), reach):
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                if depth[i + d_row, j + d_col] > depth[i, j]:
                    return False

    return True


@jit(parallel=True, cache=True)
def _crests(depth: np.ndarray, reach: int = 2):
    """
    Mark the crest of the inward flood.

    :param depth: inward flood steps, padded by 2 px
    :param reach: a crest pixel has nothing deeper within this many px (at most 2)
    :return: int8 grid, 1 = crest, 2 = other inside pixels, 0 = outside
    """
    n_rows, n_cols = depth.shape
    state = np.zeros((n_rows, n_cols), np.int8)

    for row in prange(2, n_rows - 2):
        for col in range(2, n_cols - 2):
            if depth[row, col]:
                state[row, col] = 1 if _top_aux(depth, row, col, reach) else 2

    return state


@jit(parallel=True, cache=True)
def _scan_fill(x0: int, y0: int, x1: int, y1: int, h: int, w: int):
    """
    Rasterize polygon edges with an even-odd scanline fill at pixel centres.

    :param x0, y0, x1, y1: edge end points in px (x = column, y = row, pointing down)
    :param h, w: mask size
    :return: bool mask
    """
    crossings_per_row = np.zeros(h + 1, np.int64)

    for edge in range(len(x0)):
        first_row = max(int(np.ceil(min(y0[edge], y1[edge]) - 0.5)), 0)
        last_row = min(int(np.ceil(max(y0[edge], y1[edge]) - 0.5)), h)
        for row in range(first_row, last_row):
            crossings_per_row[row + 1] += 1

    row_start = np.cumsum(crossings_per_row)
    crossings = np.empty(row_start[-1], np.float64)
    cursor = row_start[:-1].copy()

    for edge in range(len(x0)):
        first_row = max(int(np.ceil(min(y0[edge], y1[edge]) - 0.5)), 0)
        last_row = min(int(np.ceil(max(y0[edge], y1[edge]) - 0.5)), h)
        for row in range(first_row, last_row):
            crossings[cursor[row]] = x0[edge] + (row + 0.5 - y0[edge]) * (x1[edge] - x0[edge]) / (y1[edge] - y0[edge])
            cursor[row] += 1

    mask = np.zeros((h, w), np.bool_)
    for row in prange(h):
        xs = np.sort(crossings[row_start[row] : row_start[row + 1]])
        for k in range(0, len(xs) - 1, 2):
            first_col = max(int(np.ceil(xs[k] - 0.5)), 0)
            last_col = min(int(np.ceil(xs[k + 1] - 0.5)), w)
            for col in range(first_col, last_col):
                mask[row, col] = True

    return mask


def _rasterize(polygon: Polygon | MultiPolygon, resolution: float):
    """
    Burn a polygon into a bool mask with 1 px of margin.

    :param resolution: pixel size in map units
    :return: mask and the map coordinates (left, top) of its upper-left corner
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    left, top = min_x - resolution, max_y + resolution
    n_rows = int(np.ceil((max_y - min_y) / resolution)) + 2
    n_cols = int(np.ceil((max_x - min_x) / resolution)) + 2

    rings = shapely.get_rings(shapely.get_parts(polygon))
    xy, ring_of = shapely.get_coordinates(rings, return_index=True)
    cols, rows = (xy[:, 0] - left) / resolution, (top - xy[:, 1]) / resolution
    is_edge = ring_of[1:] == ring_of[:-1]

    mask = _scan_fill(cols[:-1][is_edge], rows[:-1][is_edge], cols[1:][is_edge], rows[1:][is_edge], n_rows, n_cols)
    return mask, (left, top)


@jit(cache=True)
def _thin(rows: np.ndarray, cols: np.ndarray, h: int, w: int, radius: int):
    """
    Greedy thinning: keep a pixel, block everything within its radius, move on.

    :param rows, cols: crest pixels, deepest first
    :param h, w: raster size
    :param radius: suppression radius of each pixel
    :return: bool array, True for kept pixels
    """
    taken = np.zeros((h, w), np.bool_)
    keep = np.zeros(len(rows), np.bool_)

    for pixel in range(len(rows)):
        row, col = rows[pixel], cols[pixel]
        if taken[row, col]:
            continue

        keep[pixel] = True
        reach, radius_squared = int(radius[pixel]), radius[pixel] * radius[pixel]

        for d_row in range(-reach, reach + 1):
            for d_col in range(-reach, reach + 1):
                inside_disc = d_row * d_row + d_col * d_col <= radius_squared
                if inside_disc and 0 <= row + d_row < h and 0 <= col + d_col < w:
                    taken[row + d_row, col + d_col] = True

    return keep


@jit(inline='always')
def _dist(points: np.ndarray, i: int, j: int, eps: float = 1e-12):
    """
    Distance between points i and j, nudged by eps * (i + j) so that no two links tie.
    """
    return math.hypot(float(points[i, 0] - points[j, 0]), float(points[i, 1] - points[j, 1])) + eps * (i + j)


def _links(points: np.ndarray, heights: np.ndarray, near: np.ndarray) -> np.ndarray:
    """
    Keep the candidate links no longer than twice the depth of their deeper end.

    :param points: (n, 2) row, col of the crest points
    :param heights: flood depth of each point
    :param near: (m, 2) candidate links
    :return: the short enough links
    """
    return near[_short(points, heights, near)]


@jit(parallel=True, cache=True)
def _short(points: np.ndarray, heights: np.ndarray, near: np.ndarray):
    """
    Mask of the links in near no longer than twice the depth of their deeper end.
    """
    short = np.empty(len(near), np.bool_)

    for link in prange(len(near)):
        a, b = near[link, 0], near[link, 1]
        span = math.hypot(float(points[a, 0] - points[b, 0]), float(points[a, 1] - points[b, 1]))
        short[link] = span <= 2 * max(heights[a], heights[b])

    return short


@jit(parallel=True, cache=True)
def _on_ridge(points: np.ndarray, pairs: np.ndarray, depth: np.ndarray, dip: float, samples: int):
    """
    Keep the links whose depth never drops below dip * the depth of their shallower end.

    :param points: (n, 2) row, col of the crest points
    :param pairs: (m, 2) candidate links
    :param depth: inward flood steps, unpadded
    :param dip: allowed depth ratio along a link
    :param samples: samples per link
    :return: bool mask
    """
    on_ridge = np.zeros(len(pairs), np.bool_)

    for link in prange(len(pairs)):
        row0, col0 = points[pairs[link, 0], 0], points[pairs[link, 0], 1]
        row1, col1 = points[pairs[link, 1], 0], points[pairs[link, 1], 1]
        floor = dip * min(depth[row0, col0], depth[row1, col1])

        on_ridge[link] = True
        for sample in range(samples):
            t = sample / (samples - 1)
            row = int(np.rint(row0 + t * (row1 - row0)))
            col = int(np.rint(col0 + t * (col1 - col0)))

            if depth[row, col] < floor:
                on_ridge[link] = False
                break

    return on_ridge


@jit(parallel=True, cache=True)
def _prune(points: np.ndarray, pairs: np.ndarray, cycle: int = 4):
    """
    Drop the longest link of every short cycle (shortcuts).

    :param points: (n, 2) row, col of the crest points
    :param pairs: (m, 2) links
    :param cycle: longest cycle checked: 4, 3 (triangles only) or 0 (off)
    :return: bool array, True for links to keep
    """
    n, m = len(points), len(pairs)
    if cycle < 3:
        return np.ones(m, np.bool_)

    degree = np.zeros(n + 1, np.int64)
    for link in range(m):
        degree[pairs[link, 0] + 1] += 1
        degree[pairs[link, 1] + 1] += 1

    start = np.cumsum(degree)
    neighbour = np.empty(start[-1], np.int64)
    length = np.empty(start[-1], np.float64)
    cursor = start[:-1].copy()

    for link in range(m):
        a, b = pairs[link, 0], pairs[link, 1]
        span = _dist(points, a, b)

        for here, there in ((a, b), (b, a)):
            neighbour[cursor[here]], length[cursor[here]] = there, span
            cursor[here] += 1

    keep = np.ones(m, np.bool_)
    for link in prange(m):
        a, b = pairs[link, 0], pairs[link, 1]
        span = _dist(points, a, b)
        shortcut = False

        for to_middle in range(start[a], start[a + 1]):
            middle = neighbour[to_middle]
            if middle == b or length[to_middle] >= span:
                continue

            for to_corner in range(start[middle], start[middle + 1]):
                corner = neighbour[to_corner]
                if corner == a or length[to_corner] >= span:
                    continue

                if corner == b:
                    shortcut = True
                elif cycle >= 4:
                    for to_b in range(start[b], start[b + 1]):
                        if neighbour[to_b] == corner and length[to_b] < span:
                            shortcut = True
                            break

                if shortcut:
                    break

            if shortcut:
                break

        keep[link] = not shortcut

    return keep


@jit(cache=True)
def _faces(points: np.ndarray, pairs: np.ndarray):
    """
    Trace the faces of the planar link graph.

    Half link 2e runs pairs[e, 0] -> pairs[e, 1], 2e + 1 runs back.

    :param points: (n, 2) row, col of the crest points
    :param pairs: (m, 2) links
    :return: face of every half link, the half link after it on its face, signed face areas (> 0 inside a loop)
    """
    n, m = len(points), len(pairs)

    count = np.zeros(n + 1, np.int64)
    for link in range(m):
        count[pairs[link, 0] + 1] += 1
        count[pairs[link, 1] + 1] += 1

    start = np.cumsum(count)
    outgoing, heading = np.empty(2 * m, np.int64), np.empty(2 * m, np.float64)
    cursor = start[:-1].copy()

    for half in range(2 * m):
        tail, head = pairs[half // 2, half % 2], pairs[half // 2, 1 - half % 2]
        angle = math.atan2(points[head, 0] - points[tail, 0], points[head, 1] - points[tail, 1])

        slot = cursor[tail]
        while slot > start[tail] and heading[slot - 1] > angle:
            outgoing[slot], heading[slot] = outgoing[slot - 1], heading[slot - 1]
            slot -= 1

        outgoing[slot], heading[slot] = half, angle
        cursor[tail] += 1

    slot_of = np.empty(2 * m, np.int64)

    for slot in range(2 * m):
        slot_of[outgoing[slot]] = slot

    face, after = np.full(2 * m, -1, np.int64), np.empty(2 * m, np.int64)
    area = np.zeros(2 * m + 1, np.float64)
    n_faces = 0

    for first in range(2 * m):
        if face[first] >= 0:
            continue

        half = first
        while face[half] < 0:
            tail, head = pairs[half // 2, half % 2], pairs[half // 2, 1 - half % 2]
            face[half] = n_faces
            area[n_faces] += 0.5 * (points[tail, 0] * points[head, 1] - points[head, 0] * points[tail, 1])

            slot = slot_of[half ^ 1] + 1
            after[half] = outgoing[slot if slot < start[head + 1] else start[head]]
            half = after[half]

        n_faces += 1

    return face, after, area[:n_faces]


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
    Open the loops around holes too small for the depth around them (occluders, not blocks).

    The path holding the loop's shallowest link is removed, from junction to junction.

    :param points: row, col of the crest points
    :param pairs: links
    :param heights: flood depth of each point
    :param mask: 2D bool, True inside
    :param hole: a loop stays if it holds (hole * its deepest point)^2 background px...
    :param loop: ...or is larger than the disc of loop * its deepest point
    :return: bool array, True for links to keep
    """
    n, m = len(points), len(pairs)
    keep = np.ones(m, np.bool_)

    opened = True
    while opened:
        opened = False
        alive = np.flatnonzero(keep)
        face, after, area = _faces(points, pairs[alive])
        n_halves = len(after)

        tail_of, head_of = np.empty(n_halves, np.int64), np.empty(n_halves, np.int64)
        before, first = np.empty_like(after), np.full(len(area), -1, np.int64)
        degree = np.zeros(n, np.int64)

        for half in range(n_halves):
            link = alive[half // 2]
            tail_of[half], head_of[half] = pairs[link, half % 2], pairs[link, 1 - half % 2]
            before[after[half]] = half
            if first[face[half]] < 0:
                first[face[half]] = half
            degree[tail_of[half]] += 1

        ring = np.empty((n_halves, 2), np.int64)
        for f in range(len(area)):
            if area[f] <= 0:
                continue

            size, deepest, weakest, weakest_half, intact = 0, 0, np.inf, -1, True
            half = first[f]

            while size == 0 or half != first[f]:
                tail, head = tail_of[half], head_of[half]
                ring[size, 0], ring[size, 1] = tail, head
                size += 1

                deepest = max(deepest, heights[tail])
                intact = intact and keep[alive[half // 2]]
                shallow_end = min(heights[tail], heights[head])
                if face[half ^ 1] != f and shallow_end < weakest:
                    weakest, weakest_half = shallow_end, half

                half = after[half]

            if not intact or weakest_half < 0 or area[f] > np.pi * (loop * deepest) ** 2:
                continue

            background, needed, xs = 0, (hole * deepest) ** 2, np.empty(size, np.float64)
            ring_rows = points[ring[:size, 0], 0]
            for row in range(ring_rows.min(), ring_rows.max() + 1):
                n_crossings = 0

                for k in range(size):
                    (row0, col0), (row1, col1) = points[ring[k, 0]], points[ring[k, 1]]
                    if (row0 > row) != (row1 > row):
                        xs[n_crossings] = col0 + (row - row0) * (col1 - col0) / (row1 - row0)
                        n_crossings += 1

                crossings = np.sort(xs[:n_crossings])
                for k in range(0, n_crossings - 1, 2):
                    first_col, last_col = int(np.ceil(crossings[k])), int(np.floor(crossings[k + 1]))
                    background += np.sum(~mask[row, first_col : last_col + 1])

                if background >= needed:
                    break

            if background >= needed:
                continue

            path_start, path_end = weakest_half, weakest_half

            while degree[head_of[path_end]] == 2 and after[path_end] != path_start:
                path_end = after[path_end]

            while degree[tail_of[path_start]] == 2 and before[path_start] != path_end:
                path_start = before[path_start]

            if head_of[path_end] == tail_of[path_start]:
                continue

            half, opened = path_start, True
            while True:
                link = alive[half // 2]
                keep[link] = False
                degree[pairs[link, 0]] -= 1
                degree[pairs[link, 1]] -= 1

                if half == path_end:
                    break
                half = after[half]

    return keep


@jit(cache=True)
def _despur(points: np.ndarray, pairs: np.ndarray, heights: np.ndarray, near: np.ndarray, spur: float, tip: float):
    """
    Remove spurs leaf by leaf: dead ends whose disc fits inside the disc of a much deeper point.

    :param points: row, col of the crest points
    :param pairs: links
    :param heights: flood depth of each point
    :param near: point pairs close enough to cover one another
    :param spur: disc scale of the deeper point
    :param tip: depth ratio below which a dead end counts as shallow
    :return: bool array, True for links to keep
    """
    n, m = len(points), len(pairs)

    covered = np.zeros(n, np.bool_)
    for pair in range(len(near)):
        for small, large in ((near[pair, 0], near[pair, 1]), (near[pair, 1], near[pair, 0])):
            shallow = heights[small] < tip * heights[large]
            if shallow and _dist(points, large, small) + heights[small] <= spur * heights[large]:
                covered[small] = True

    count = np.zeros(n + 1, np.int64)
    for link in range(m):
        count[pairs[link, 0] + 1] += 1
        count[pairs[link, 1] + 1] += 1

    start = np.cumsum(count)
    neighbour = np.empty(start[-1], np.int64)
    link_at = np.empty(start[-1], np.int64)
    cursor = start[:-1].copy()

    for link in range(m):
        a, b = pairs[link, 0], pairs[link, 1]

        for here, there in ((a, b), (b, a)):
            neighbour[cursor[here]], link_at[cursor[here]] = there, link
            cursor[here] += 1

    degree = count[1:].copy()
    alive = np.ones(m, np.bool_)
    leaves = [point for point in range(n) if degree[point] == 1]

    while leaves:
        leaf = leaves.pop()
        if degree[leaf] != 1 or not covered[leaf]:
            continue

        for slot in range(start[leaf], start[leaf + 1]):
            if not alive[link_at[slot]]:
                continue

            alive[link_at[slot]] = False
            degree[leaf] -= 1
            degree[neighbour[slot]] -= 1

            if degree[neighbour[slot]] == 1:
                leaves.append(neighbour[slot])

    return alive


@jit(cache=True)
def _bridge(points: np.ndarray, pairs: np.ndarray, near: np.ndarray, depth: np.ndarray):
    """
    Join dead ends that face each other across a break of the crest, when each is the other's nearest and the
    straight line between them stays inside the shape.

    :param points: row, col of the crest points
    :param pairs: links
    :param near: point pairs close enough to cover one another
    :param depth: inward flood steps, unpadded
    :return: the links with the bridges appended
    """
    n = len(points)
    degree, neighbour = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for link in range(len(pairs)):
        a, b = pairs[link, 0], pairs[link, 1]
        degree[a] += 1
        degree[b] += 1
        neighbour[a], neighbour[b] = b, a

    nearest, mate = np.full(n, np.inf), np.full(n, -1, np.int64)
    for pair in range(len(near)):
        u, v = near[pair, 0], near[pair, 1]
        if degree[u] != 1 or degree[v] != 1 or neighbour[u] == v:
            continue

        d_row, d_col = points[v, 0] - points[u, 0], points[v, 1] - points[u, 1]
        u_heading = (points[u, 0] - points[neighbour[u], 0], points[u, 1] - points[neighbour[u], 1])
        v_heading = (points[v, 0] - points[neighbour[v], 0], points[v, 1] - points[neighbour[v], 1])
        u_faces_v = d_row * u_heading[0] + d_col * u_heading[1] > 0
        v_faces_u = d_row * v_heading[0] + d_col * v_heading[1] < 0
        span = _dist(points, u, v)

        if not (u_faces_v and v_faces_u) or (span >= nearest[u] and span >= nearest[v]):
            continue

        n_steps, inside = int(np.ceil(span)), True
        for step in range(n_steps + 1):
            row = int(np.rint(points[u, 0] + step * d_row / n_steps))
            col = int(np.rint(points[u, 1] + step * d_col / n_steps))
            if depth[row, col] == 0:
                inside = False
                break

        if not inside:
            continue

        if span < nearest[u]:
            nearest[u], mate[u] = span, v

        if span < nearest[v]:
            nearest[v], mate[v] = span, u

    mutual = np.array([u for u in range(n) if mate[u] > u and mate[mate[u]] == u], np.int64)
    bridged = np.empty((len(pairs) + len(mutual), 2), pairs.dtype)
    bridged[: len(pairs)] = pairs
    bridged[len(pairs) :, 0], bridged[len(pairs) :, 1] = mutual, mate[mutual]

    return bridged


@jit(cache=True)
def _tips(
    points: np.ndarray,
    pairs: np.ndarray,
    heights: np.ndarray,
    crest: np.ndarray,
    depth: np.ndarray,
    tip: float,
    radius_k: float = 0.35,
):
    """
    Move every dead end, in place, to the far end of the crest it was thinned from.

    :param points: row, col of the crest points
    :param pairs: links
    :param heights: flood depth of each point
    :param crest: crest raster, padded by 2 px
    :param depth: inward flood steps, unpadded
    :param tip: the new end must stay deeper than tip * the old one
    :param radius_k: radius scaling for crest height; default: 0.35
    """
    n, (n_rows, n_cols) = len(points), depth.shape
    degree, neighbour = np.zeros(n, np.int64), np.zeros(n, np.int64)

    for link in range(len(pairs)):
        a, b = pairs[link, 0], pairs[link, 1]
        degree[a] += 1
        degree[b] += 1
        neighbour[a], neighbour[b] = b, a

    for end in range(n):
        if degree[end] != 1:
            continue

        row, col = points[end, 0], points[end, 1]
        d_row, d_col = row - points[neighbour[end], 0], col - points[neighbour[end], 1]
        radius = max(radius_k * heights[end], 3.0)
        reach, farthest, target = int(radius), 0, (row, col)

        for r in range(max(row - reach, 0), min(row + reach + 1, n_rows)):
            for c in range(max(col - reach, 0), min(col + reach + 1, n_cols)):
                ahead = (r - row) * d_row + (c - col) * d_col
                inside_disc = (r - row) ** 2 + (c - col) ** 2 <= radius * radius

                if inside_disc and ahead > farthest and crest[r + 2, c + 2] and depth[r, c] >= tip * heights[end]:
                    farthest, target = ahead, (r, c)

        points[end, 0], points[end, 1] = target
        heights[end] = depth[target]


@jit(parallel=True, cache=True)
def _center(points: np.ndarray, heights: np.ndarray, crest: np.ndarray, depth: np.ndarray):
    """
    Move every point to the mean of the crest pixels of its own depth within sqrt(5) px.

    :param points: row, col of the crest points
    :param heights: flood depth of each point
    :param crest: crest raster, padded by 2 px
    :param depth: inward flood steps, unpadded
    :return: (n, 2) float row, col of the pixel centres
    """
    centers = np.empty((len(points), 2), np.float64)

    for point in prange(len(points)):
        row, col = points[point, 0], points[point, 1]
        sum_row, sum_col, count = 0, 0, 0

        for d_row in range(-2, 3):
            for d_col in range(-2, 3):
                if d_row * d_row + d_col * d_col > 5 or not crest[row + d_row + 2, col + d_col + 2]:
                    continue

                if depth[row + d_row, col + d_col] == heights[point]:
                    sum_row, sum_col, count = sum_row + d_row, sum_col + d_col, count + 1

        centers[point, 0], centers[point, 1] = row + 0.5 + sum_row / count, col + 0.5 + sum_col / count

    return centers


def crestline(
    source,
    epsg_crs: int | None = None,
    resolution: float | None = None,
    transform: Affine | None = None,
    simplify: float = 0.0,
    radius_k: float = 0.35,
    dip: float = 0.65,
    spur: float = 2.0,
    tip: float = 0.75,
    hole: float = 0.25,
    device: str = 'cpu',
) -> MultiLineString:
    """
    Trace the centerline of a polygon or a polygon mask.

    :param source: shapely Polygon / MultiPolygon, or a 2D bool mask (True inside)
    :param epsg_crs: EPSG code of the input
    :param resolution: pixel size in map units, required for a polygon
    :param transform: affine pixel -> map for a mask (e.g. rasterio's); None keeps pixel coordinates
    :param simplify: Douglas-Peucker tolerance in output units, 0 = off
    :param radius_k: radius scaling for crest depth
    :param dip: a link's depth must stay above dip * the depth of its shallower end
    :param spur: prune dead ends whose disc fits inside spur * the disc of a deeper point, 0 = off
    :param tip: ...if they are shallower than tip * its depth
    :param hole: open loops around holes narrower than hole * the surrounding depth, 0 = off
    :param device: 'cpu' or 'cuda' (needs numba-cuda), same result
    :return: centerline, empty if there is none
    """
    pixel_grid = None
    if isinstance(source, shapely.Geometry):
        if source.geom_type not in ('Polygon', 'MultiPolygon'):
            raise TypeError(f'expected a Polygon or MultiPolygon, got {source.geom_type}')
        if not resolution or resolution <= 0:
            raise ValueError('res (pixel size in map units) is required for a polygon')
        if source.is_empty:
            return MultiLineString()

        source, proj_crs = _project(source, epsg_crs)
        mask, (left, top) = _rasterize(source, resolution)
        pixel_grid = (left, top, resolution)
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

    crest_rows, crest_cols = np.nonzero(crest[2:-2, 2:-2])
    if len(crest_rows) < 2:
        return MultiLineString()

    crest_depth = depth[crest_rows, crest_cols]
    by_depth = np.argsort(-crest_depth, kind='stable')
    crest_rows, crest_cols, crest_depth = crest_rows[by_depth], crest_cols[by_depth], crest_depth[by_depth]

    kept = _thin(crest_rows, crest_cols, mask.shape[0], mask.shape[1], np.maximum(radius_k * crest_depth, 3))
    points, heights = np.c_[crest_rows[kept], crest_cols[kept]], crest_depth[kept]
    deepest = heights.max()

    near = cKDTree(points).query_pairs(max(2, spur) * deepest, output_type='ndarray')
    pairs = _links(points, heights, near)
    pairs = pairs[_on_ridge(points, pairs, depth, dip, 2 * deepest + 2)]
    pairs = pairs[_prune(points, pairs)]

    if spur > 0:
        pairs = pairs[_despur(points, pairs, heights, near, spur, tip)]

    pairs = _bridge(points, pairs, near, depth)

    if hole > 0:
        pairs = pairs[_occluded(points, pairs, heights, mask, hole)]

    if len(pairs) == 0:
        return MultiLineString()

    _tips(points, pairs, heights, crest, depth, tip)
    centers = _center(points, heights, crest, depth)

    xy = np.c_[centers[:, 1], centers[:, 0]]

    if pixel_grid is not None:
        left, top, resolution = pixel_grid
        xy = np.c_[left + xy[:, 0] * resolution, top - xy[:, 1] * resolution]
    elif transform is not None:
        xy = np.c_[transform @ tuple(xy.T)]

    lines = shapely.line_merge(multilinestrings(linestrings(xy[pairs])))

    if simplify:
        lines = lines.simplify(simplify)

    if epsg_crs is not None and CRS.from_epsg(epsg_crs).is_geographic:
        lines = _reproject_crs(lines, src_crs=proj_crs, dst_crs=CRS.from_epsg(epsg_crs))

    return lines if isinstance(lines, MultiLineString) else MultiLineString([lines])
