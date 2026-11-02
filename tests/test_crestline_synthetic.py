import functools
import logging
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.spatial import cKDTree
from shapely import Polygon

from crestline.crestline import _bridge
from crestline.crestline import _flood
from crestline.crestline import _grid
from crestline.crestline import _on_ridge
from crestline.crestline import _rasterize
from crestline.crestline import crestline

log = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def _mock_warm():
    m = np.zeros((40, 120), bool)
    m[10:30, 10:110] = True
    _flood(_grid(m))
    _rasterize(shapely.box(0, 0, 4, 4), 1.0)
    _on_ridge(np.array([[1, 1], [1, 3]]), np.array([[0, 1]]), np.ones((4, 6), np.int32), 0.5, 4)
    crestline(m)


def _record_running_time(fn, repeat=3):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        best, out = float('inf'), None
        for _ in range(repeat):
            t = time.perf_counter()
            out = fn(*args, **kwargs)
            best = min(best, time.perf_counter() - t)
        return out, best

    return inner


_timed_flood = _record_running_time(_flood)
_timed_rasterize = _record_running_time(_rasterize)


def _centres_test_case(shape, left, top, res):
    h, w = shape
    xs = left + (np.arange(w) + 0.5) * res
    ys = top - (np.arange(h) + 0.5) * res
    return np.meshgrid(xs, ys)


def _comb_test_case(n, length, tall, short):
    """Square-wave polygon: n vertical risers of height (tall - short), evenly spaced along x."""
    step = length / n
    pts = [(0.0, 0.0)]
    for k in range(n):
        y = tall if k % 2 == 0 else short
        pts += [(k * step, y), ((k + 1) * step, y)]
    pts.append((length, 0.0))
    return np.array(pts)


@pytest.mark.parametrize(
    'merge, holed, expected_missing',
    [(False, True, 50), (True, False, 0)],
    ids=['overlapping', 'unioned'],
)
def test_rasterize_scan_fill_overlapping_parts(merge, holed, expected_missing):
    parts = [shapely.box(0, 0, 10, 10), shapely.box(5, 0, 15, 10)]
    truth = shapely.union_all(parts)

    mask, (left, top) = _rasterize(truth if merge else shapely.MultiPolygon(parts), 1.0)
    inside = shapely.contains_xy(truth, *_centres_test_case(mask.shape, left, top, 1.0))

    missing = inside & ~mask

    assert missing.any() == holed
    assert not (mask & ~inside).any()  # never fills outside the true footprint
    assert missing.sum() == expected_missing


def test_rasterize_cost_driven_by_vertical_edges():
    xy = _comb_test_case(n=400, length=4000.0, tall=2000.0, short=20.0)
    upright, lying = Polygon(xy), Polygon(xy[:, ::-1])

    (up_mask, _), t_up = _timed_rasterize(upright, 1.0)
    (lie_mask, _), t_lie = _timed_rasterize(lying, 1.0)

    assert up_mask.size == lie_mask.size
    assert t_up > t_lie


def test_flood_cost_is_pixel_bound_not_depth_bound_on_cuda():
    total = 4_000_000
    runs = [_timed_flood(_grid(np.ones((width, total // width), bool))) for width in (20, 200, 2000)]
    depths, times = [int(level.max()) for level, _ in runs], [t for _, t in runs]

    assert depths == [10, 100, 1000]
    assert max(times) < 3 * min(times)


@pytest.mark.skipif(not Path('/proc/self/status').exists(), reason='needs /proc (Linux)')
def test_flood_memory_scales_with_bbox():
    def peak_bytes(side):
        probe = textwrap.dedent(f"""
            import numpy as np
            from crestline.crestline import _flood, _grid

            def hwm():
                with open('/proc/self/status') as fh:
                    return next(int(ln.split()[1]) * 1024 for ln in fh if ln.startswith('VmHWM:'))

            _flood(_grid(np.ones((8, 40), bool)))  # compile before measuring, not during
            mask = np.zeros(({side}, {side}), bool)
            mask[{side} // 2:{side} // 2 + 20, 500:2500] = True
            state = _grid(mask)

            open('/proc/self/clear_refs', 'w').write('5')
            base = hwm()
            assert _flood(state).any()
            print(hwm() - base)
        """)
        done = subprocess.run(
            [sys.executable, '-c', probe], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return int(done.stdout.split()[-1])

    small, large = peak_bytes(3000), peak_bytes(6000)
    assert small > 0, 'ru_maxrss never rose above its baseline - the measurement, not _flood, is broken'

    assert large > 3 * small


@pytest.mark.parametrize('neck, kept', [(10, True), (7, True), (6, False), (0, False)])
def test_on_ridge_rejects_links_crossing_a_neck(neck, kept):
    depth = np.full((21, 61), 10, np.int32)
    depth[:, 28:33] = neck

    points = np.array([[10, 5], [10, 55]])
    mask = _on_ridge(points, np.array([[0, 1]]), depth, 0.65, 64)

    assert bool(mask[0]) is kept


def _bridged(points, pairs, depth, radius=10):
    points, pairs = np.array(points), np.array(pairs, np.int64).reshape(-1, 2)
    near = cKDTree(points).query_pairs(radius, output_type='ndarray')
    out = _bridge(points, pairs, near, depth)

    assert (out[: len(pairs)] == pairs).all()
    return {tuple(sorted(row)) for row in out[len(pairs) :].tolist()}


FACING = [(5, 2), (5, 6), (5, 10), (5, 16), (5, 20), (5, 24)]
CHAINS = [(0, 1), (1, 2), (3, 4), (4, 5)]


def test_bridge_joins_dead_ends_facing_each_other_across_a_break():
    assert _bridged(FACING, CHAINS, np.full((11, 30), 3, np.int32)) == {(2, 3)}


def test_bridge_never_crosses_the_background():
    depth = np.full((11, 30), 3, np.int32)
    depth[:, 13] = 0

    assert _bridged(FACING, CHAINS, depth) == set()


def test_bridge_leaves_side_by_side_dead_ends_apart():
    points = [(5, 2), (5, 6), (5, 10), (9, 2), (9, 6), (9, 10)]

    assert _bridged(points, CHAINS, np.full((15, 15), 3, np.int32)) == set()


def test_bridge_joins_only_mutual_nearest_dead_ends():
    points = [(5, 2), (5, 6), (5, 10), (5, 14), (5, 18), (5, 22), (8, 16), (8, 20), (8, 24)]
    pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]

    assert _bridged(points, pairs, np.full((12, 30), 3, np.int32)) == {(2, 3)}


def test_bridge_without_links_adds_nothing():
    assert _bridged(FACING, [], np.full((11, 30), 3, np.int32)) == set()
