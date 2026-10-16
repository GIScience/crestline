"""
Synthetic stress / limit tests.

These complement test_crestline.py (which checks correctness on small shapes) by pinning down
*where the cost lives* and *which guard is load-bearing* on adversarial inputs.

No plots: the rasters here are far too large to render.
"""

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
from shapely import Polygon

from crestline.crestline import _flood
from crestline.crestline import _grid
from crestline.crestline import _on_ridge
from crestline.crestline import _rasterize
from crestline.crestline import crestline

log = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def _mock_warm():
    """Every kernel is @jit; without a warm-up the first call times compilation, not work."""
    m = np.zeros((40, 120), bool)
    m[10:30, 10:110] = True
    _flood(_grid(m))
    _rasterize(shapely.box(0, 0, 4, 4), 1.0)
    _on_ridge(np.array([[1, 1], [1, 3]]), np.array([[0, 1]]), np.ones((4, 6), np.int32), 0.5, 4)
    crestline(m)


def _record_running_time(fn, repeat=3):
    """
    Decorator: run `fn` `repeat` times and return (its result, best wall clock in seconds).

    Min-of-N is the least noisy estimator for CPU-bound work. Returning the result as well means
    a test never has to call the kernel a second time just to inspect what it produced.
    """

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


def _mock_parallel_roads(gap, wide=60, narrow=12, length=400, margin=5):
    """A wide road and a narrow road running side by side, separated by `gap` empty pixels."""
    h = 2 * margin + wide + gap + narrow
    mask = np.zeros((h, length + 2 * margin), bool)
    mask[margin : margin + wide, margin:-margin] = True
    mask[margin + wide + gap : margin + wide + gap + narrow, margin:-margin] = True
    return mask


@pytest.mark.parametrize(
    'merge, holed, expected_missing', [(False, True, 50), (True, False, 0)],
    ids=['overlapping', 'unioned']
)
def test_rasterize_scan_fill_overlapping_parts(merge, holed, expected_missing):
    """
    _scan_fill in _rasterize uses even-odd parity, not nonzero winding.

    Two *overlapping* MultiPolygon parts therefore XOR their overlap into a phantom hole, which
    _flood then treats as a real obstacle. union_all() beforehand is mandatory, not cosmetic.
    """
    parts = [shapely.box(0, 0, 10, 10), shapely.box(5, 0, 15, 10)]
    truth = shapely.union_all(parts)

    mask, (left, top) = _rasterize(truth if merge else shapely.MultiPolygon(parts), 1.0)
    inside = shapely.contains_xy(truth, *_centres_test_case(mask.shape, left, top, 1.0))

    missing = inside & ~mask
    log.info('%s: %d pixels lost to even-odd parity', 'merged' if merge else 'raw', missing.sum())

    assert missing.any() == holed
    assert not (mask & ~inside).any()  # never fills outside the true footprint
    assert missing.sum() == expected_missing


def test_rasterize_cost_driven_by_vertical_edges():
    """
    Passes 1 and 2 of _scan_fill are serial over edges and cost one iteration per *scanline the
    edge crosses*. A horizontal edge is free (lo >= hi); a near-vertical one is not.

    Same polygon, same vertex count, same pixel count - only transposed.
    """
    xy = _comb_test_case(n=400, length=4000.0, tall=2000.0, short=20.0)
    upright, lying = Polygon(xy), Polygon(xy[:, ::-1])

    (up_mask, _), t_up = _timed_rasterize(upright, 1.0)
    (lie_mask, _), t_lie = _timed_rasterize(lying, 1.0)

    assert up_mask.size == lie_mask.size
    log.info('vertical risers %.1f ms vs horizontal %.1f ms (%.1fx)', t_up * 1e3, t_lie * 1e3, t_up / t_lie)
    assert t_up > t_lie


# TODO: ensure this test will be on cuda
def test_flood_cost_is_pixel_bound_not_depth_bound_on_cuda():
    """
    Bars of equal pixel count but 100x different depth. A bar's max flood depth is width // 2,
    so width is a clean depth knob at constant area and constant bounding box.

    On CPU each pixel is enqueued exactly once, so runtime should barely move. This is what
    localises the depth sensitivity to crestline/cuda.py, where flood() pays one kernel launch
    and one blocking copy_to_host() *per step*.
    """
    total = 4_000_000
    runs = [_timed_flood(_grid(np.ones((width, total // width), bool))) for width in (20, 200, 2000)]
    depths, times = [int(level.max()) for level, _ in runs], [t for _, t in runs]

    log.info('depths %s -> %s ms', depths, [round(t * 1e3) for t in times])

    assert depths == [10, 100, 1000]              # 100x more sequential steps...
    assert max(times) < 3 * min(times)            # ...for near-constant CPU time


def test_flood_seeding_time_scales_with_bbox():
    """
    The `touch` loop runs a 3x3 scan over *every* pixel of the padded grid before the BFS starts.

    Identical road, 16x the empty bounding box: time should grow with the box. A sparse network
    over a large extent therefore spends most of its runtime seeding nothing.
    """

    def bar_in(side):
        mask = np.zeros((side, side), bool)
        mask[side // 2 : side // 2 + 20, 100:300] = True
        return _grid(mask)

    _, t_small = _timed_flood(bar_in(600))
    _, t_large = _timed_flood(bar_in(2400))

    log.info('600^2 %.1f ms vs 2400^2 %.1f ms (%.1fx for 16x the bbox)', t_small * 1e3, t_large * 1e3, t_large / t_small)
    assert t_large > 4 * t_small


# TODO: confirm this memory test's logic (I'm not familiar with)
@pytest.mark.skipif(not Path('/proc/self/status').exists(), reason='needs /proc (Linux)')
def test_flood_memory_scales_with_bbox():
    """
    _flood allocates level (int32) + queue (int32) + touch (bool) = ~9 bytes per *bounding box*
    pixel, however empty the box is. `queue` is sized h*w although it can only ever hold flooded
    pixels. A 100 km x 100 km extent at 1 m would reserve ~90 GB for a handful of roads.

    Resident cost is lower than that - measured ~3 B/px - because level/queue pages are faulted
    lazily and a sparse raster writes almost none of them; `touch` is the one array the seeding
    loop fills completely. So we assert the *scaling*, not an absolute figure: same 40k filled
    pixels, 4x the bounding box, and memory must follow the box.
    """
    def peak_bytes(side):
        # Measure in a child so an earlier test's allocations are out of the picture - but read
        # VmHWM, not ru_maxrss: the latter is inherited across fork+exec, so a heavy parent leaves
        # the child's watermark already above anything _flood adds. clear_refs resets VmHWM to the
        # current RSS, which ru_maxrss has no equivalent for.
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

    log.info('%.0f -> %.0f MB (%.1fx) for 4x the bbox, %.1f B/px', small / 2**20, large / 2**20, large / small, large / 6000**2)
    assert large > 3 * small


@pytest.mark.parametrize('neck, kept', [(10, True), (7, True), (6, False), (0, False)])
def test_on_ridge_rejects_links_crossing_a_neck(neck, kept):
    """
    A link survives iff the depth everywhere along it stays >= dip * min(depth at its two ends).

    Ends sit at depth 10 and dip is 0.65, so the threshold is exactly 6.5.
    """
    depth = np.full((21, 61), 10, np.int32)
    depth[:, 28:33] = neck

    points = np.array([[10, 5], [10, 55]])
    mask = _on_ridge(points, np.array([[0, 1]]), depth, 0.65, 64)

    assert bool(mask[0]) is kept


@pytest.mark.parametrize('gap', [5, 20])
def test_crestline_unequal_parallel_roads_stay_separate(gap):
    """
    For equal widths the span rule alone keeps parallel roads apart. For *unequal* widths it does
    not: a 60 px road beside a 12 px road is 36 + gap apart, while the rule allows 2 * max(30, 6)
    = 60. _on_ridge is then the only thing rejecting the cross-link - this test is what holds it.

    spur=0 disables _despur, which is a separate failure mode: a narrow point is "covered" when
    dist + 6 <= spur * 30, i.e. closer than 54 px. At gap=5 the roads are 41 px apart, so every
    narrow point is covered and leaf-peeling cascades through the whole road, erasing it. At
    gap=20 they are 56 px apart and it survives. Verified: default spur gives 1 and 2 components.
    """
    assert 36 + gap <= 60, 'precondition: the span rule must accept this link'

    lines = crestline(_mock_parallel_roads(gap), spur=0)
    pieces = shapely.get_parts(shapely.union_all(shapely.buffer(shapely.get_parts(lines), 0.5)))

    log.info('gap %d px: %d connected component(s)', gap, len(pieces))
    assert len(pieces) == 2


@pytest.mark.parametrize(
    'gap, kwargs, parts, length',
    [
        (5, dict(spur=0, dip=0.0), 1, 1096.5),
        (12, dict(), 1, 340.0),
        (13, dict(), 2, 727.5)
    ],
    ids=['dip-off-cross-links', 'despur-erases-narrow', 'despur-out-of-reach'],
)
def test_crestline_unequal_parallel_roads_merge_when_a_guard_is_off(gap, kwargs, parts, length):
    """
    The two ways the previous test's separation breaks, and the one gap where it comes back.

    Total length tells the two failures apart - both collapse to a single component, but for
    opposite reasons:

    dip=0 disables _on_ridge, so the span rule (which already accepts the link, see above) is the
    only check left and the roads get rungs between them: 728 px of road plus ~370 px of rungs.

    The default spur=2.0 does the reverse - it *deletes* the narrow road. Every narrow crest point
    sits dist+6 <= 2*30 from a wide one, so all 98 are "covered" and leaf-peeling walks the whole
    road away: 340 px left, i.e. the wide road alone. The reach ends at gap=12: at 13 the narrow
    road's two *end* points fall out of range (the wide crest stops short of the raster edge, so
    the nearest wide point is diagonally further than the 54 px budget) and the cascade has no leaf
    to start from. 1 px of gap is the difference between keeping a road and losing it.
    """
    lines = crestline(_mock_parallel_roads(gap), **kwargs)
    pieces = shapely.get_parts(shapely.union_all(shapely.buffer(shapely.get_parts(lines), 0.5)))

    log.info('gap %d px, %s: %d component(s), %.1f px', gap, kwargs, len(pieces), lines.length)
    assert len(pieces) == parts
    assert lines.length == pytest.approx(length, rel=0.02)
