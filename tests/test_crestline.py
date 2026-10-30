from pathlib import Path

import matplotlib
import shapely
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from numba import jit
from numba import prange
from pyproj import CRS
from pyproj import Geod
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import LineString
from shapely import MultiLineString
from shapely import Point
from shapely import box

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from crestline.crestline import _crests
from crestline.crestline import _flood
from crestline.crestline import _grid
from crestline.crestline import _project
from crestline.crestline import _rasterize
from crestline.crestline import crestline

TEST_OUTPUT_DIR = Path(__file__).parent / 'output'


def _mock_parallel_roads(gap, wide=60, narrow=12, length=400, margin=5):
    """A wide road and a narrow road running side by side, separated by `gap` empty pixels."""
    h = 2 * margin + wide + gap + narrow
    mask = np.zeros((h, length + 2 * margin), bool)
    mask[margin : margin + wide, margin:-margin] = True
    mask[margin + wide + gap : margin + wide + gap + narrow, margin:-margin] = True
    return mask


def test_project_geographic_polygon():
    WGS84 = Geod(ellps='WGS84')
    polygon = box(xmin=8.68189868, ymin=49.41572259, xmax=8.68576529, ymax=49.41671358)
    west, south, east, north = polygon.bounds
    expected_w = WGS84.inv(west, south, east, south)[2]
    expected_h = WGS84.inv(west, south, west, north)[2]

    proj_polygon, proj_crs = _project(polygon, crs=4326)

    proj_bounds = proj_polygon.bounds
    w, h = proj_bounds[2] - proj_bounds[0], proj_bounds[3] - proj_bounds[1]
    assert (w, h) == pytest.approx((expected_w, expected_h), abs=0.01)


def test_project_projected_polygon():
    polygon = box(xmin=476928, ymin=5473720, xmax=477208, ymax=5473829)  #
    proj_polygon, proj_crs = _project(polygon, crs=25832)

    assert polygon == proj_polygon
    assert isinstance(proj_crs, CRS)


def test_project_polygon_with_no_crs():
    polygon = box(xmin=476928, ymin=5473720, xmax=477208, ymax=5473829)  #
    proj_polygon, proj_crs = _project(polygon, crs=None)

    assert polygon == proj_polygon
    assert proj_crs is None


def test_grid():
    mask = np.zeros((6, 8), bool)
    mask[1:5, 1:3] = True  # vertical bar
    mask[3:5, 3:8] = True  # horizontal bar running into the right edge
    mask[0, 7] = True  # lone pixel in the top-right corner

    state = _grid(mask)

    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(state, cmap=ListedColormap(['#D3D1C7', '#378ADD']), vmin=1, vmax=2)
    for (r, c), v in np.ndenumerate(state):
        ax.text(c, r, str(v), ha='center', va='center', fontsize=9, color='white' if v == 2 else '#444441')
    ax.add_patch(Rectangle((1.5, 1.5), w, h, fill=False, ec='#D85A30', lw=2))  # the original mask
    ax.set_title('_grid: 2 = inside (target), 1 = outside (source); coral = original mask', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    output_dir = TEST_OUTPUT_DIR / 'test_grid'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / 'grid.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    assert state.shape == (h + 4, w + 4) and state.dtype == np.int8
    assert np.array_equal(state[2:-2, 2:-2], np.where(mask, 2, 1))
    assert (state[:2] == 1).all() and (state[-2:] == 1).all()
    assert (state[:, :2] == 1).all() and (state[:, -2:] == 1).all()
    assert np.array_equal(state, np.pad(np.where(mask, 2, 1).astype(np.int8), 2, constant_values=1))


def test_flood():
    state = np.pad(np.full((8, 12), 2, np.int8), 2)
    state[5, 4] = 1
    state[2:8, 9] = 0

    lev = _flood(state)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('0.25')
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(np.ma.masked_where(state != 2, lev), cmap=cmap)

    for (r, c), v in np.ndenumerate(lev):
        if v:
            ax.text(c, r, str(v), ha='center', va='center', fontsize=8, color='w' if v < lev.max() / 2 else 'k')

    ax.text(4, 5, 'S', ha='center', va='center', color='r', fontsize=10)
    ax.set_title('_flood: step at which each pixel gets wet')

    output_dir = TEST_OUTPUT_DIR / 'test_flood'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'flood.png', dpi=120)
    plt.close(fig)

    assert not lev[state != 2].any()
    assert (lev[state == 2] > 0).all()
    assert lev[4, 4] == 1 and lev[4, 3] == 2
    assert lev[3, 10] > lev[3, 8]


@pytest.mark.parametrize(
    'profile, expected',
    [
        ([0, 0, 1, 2, 3, 2, 1, 0, 0], [0, 0, 2, 2, 1, 2, 2, 0, 0]),  # single-pixel ridge
        ([0, 0, 1, 2, 3, 3, 2, 1, 0, 0], [0, 0, 2, 2, 1, 1, 2, 2, 0, 0]),  # 2-px plateau: both crest
        ([0, 0, 1, 2, 2, 3, 2, 1, 0, 0], [0, 0, 2, 2, 2, 1, 2, 2, 0, 0]),  # terrace is not a crest
    ],
    ids=['ridge', 'plateau', 'terrace'],
)
def test_crests(profile, expected, request):
    depth = np.zeros((7, len(profile)), np.int32)
    depth[2:5] = profile

    state = _crests(depth)

    name = request.node.callspec.id
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(
        range(len(profile)),
        profile,
        edgecolor='k',
        lw=0.5,
        color=['#D85A30' if v == 1 else '#378ADD' for v in state[3]],
    )

    ax.set_xticks(range(len(profile)))
    ax.set_xlabel('column')
    ax.set_ylabel('flood depth')
    ax.set_title(f'{name}: red = crest, blue = slope', fontsize=9)

    output_dir = TEST_OUTPUT_DIR / 'test_crests'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    assert (state[2:5] == expected).all()
    assert not state[[0, 1, 5, 6]].any()


def test_ratio():
    depth = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    to_go = [0, 2, 1, 0, 1, 2, 0, 4, 3, 2, 1, 0, 1, 2, 3, 4, 0]
    expected = [0, 0, 0.5, 1, 0.5, 0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0]

    def pad(row):
        return np.pad(np.array([row], np.int32), 2)

    t = _ratio(pad(depth), pad(to_go))

    x = np.arange(len(depth))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x, depth, color='#D3D1C7')
    ax.set_ylabel('flood depth')
    ax2 = ax.twinx()
    ax2.plot(x, t[0], 'o-', color='#D85A30')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('ratio')
    ax.set_title('_ratio: grey = flood depth, red = ratio. Both roads peak at 1', fontsize=9)

    output_dir = TEST_OUTPUT_DIR / 'test_ratio'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'ratio.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    assert t.shape == (1, len(depth))
    assert np.allclose(t[0], expected)


def test_rasterize():
    square = shapely.Polygon([(0, 0), (40, 0), (40, 40), (0, 40)], holes=[[(10, 10), (30, 10), (30, 30), (10, 30)]])
    poly = shapely.MultiPolygon([square, shapely.Point(70, 20).buffer(15)])
    resolution = 2.0

    mask, (left, top) = _rasterize(poly, resolution)
    h, w = mask.shape

    xs = left + (np.arange(w) + 0.5) * resolution
    ys = top - (np.arange(h) + 0.5) * resolution
    x, y = np.meshgrid(xs, ys)
    expected = shapely.contains_xy(poly, x, y)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(
        mask,
        cmap='Blues',
        vmin=0,
        vmax=1.3,
        interpolation='nearest',
        extent=(left, left + w * resolution, top - h * resolution, top),
    )
    for ring in shapely.get_rings(shapely.get_parts(poly)):
        ax.plot(*ring.xy, color='#D85A30', lw=1.5)
    bad = mask != expected
    ax.plot(x[bad], y[bad], 'rx', ms=6)
    ax.set_title(f'_rasterize (res = {resolution:g}): {bad.sum()} mismatched pixels', fontsize=10)

    output_dir = TEST_OUTPUT_DIR / 'test_rasterize'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'rasterize.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    assert mask.shape == (22, 45)
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    assert np.array_equal(mask, expected)


def burn(geom, pad=20):
    minx, miny, maxx, maxy = geom.bounds
    shape = (int(np.ceil(maxy - miny)) + 2 * pad, int(np.ceil(maxx - minx)) + 2 * pad)
    transform = from_origin(minx - pad, maxy + pad, 1, 1)
    return rasterize([(geom, 1)], out_shape=shape, transform=transform).astype(bool)


def bar(width):
    m = np.zeros((60, 200), bool)
    m[20 : 20 + width, 10:190] = True
    return m


def road_network(seed=42, n_main=5, n_side=8, extent=1000.0):
    rng = np.random.default_rng(seed)
    roads = []

    for i in range(n_main):
        along = np.linspace(0, extent, 5)
        across = rng.uniform(0.15, 0.85) * extent + np.cumsum(rng.normal(0, 50, 5))
        roads.append(LineString(np.c_[along, across] if i % 2 == 0 else np.c_[across, along]))

    for _ in range(n_side):
        p = roads[rng.integers(n_main)].interpolate(rng.uniform(0.1, 0.9), normalized=True)
        ang, length = rng.uniform(0, 2 * np.pi), rng.uniform(80, 220)
        roads.append(LineString([(p.x, p.y), (p.x + length * np.cos(ang), p.y + length * np.sin(ang))]))

    return MultiLineString(roads)


def bumpy(seed=1, bumps=25):
    rng = np.random.default_rng(seed)
    m = np.zeros((80, 300), bool)
    m[25:55, 10:290] = True
    for _ in range(bumps):
        c, r, s = rng.integers(20, 280), rng.choice([22, 55]), rng.integers(2, 5)
        m[r : r + s, c : c + s] = True
    return m


def blocks():
    rows = [LineString([(0, y), (300, y)]) for y in (0, 100, 200)]
    cols = [LineString([(x, 0), (x, 200)]) for x in (0, 150, 300)]
    return burn(MultiLineString(rows + cols).buffer(10))


def mixed(seed=1):
    roads = road_network(seed).geoms
    half = np.random.default_rng(seed + 100).choice([6.0, 9.0, 14.0, 20.0, 30.0], len(roads))
    return burn(shapely.union_all([g.buffer(h) for g, h in zip(roads, half)]))


CASES = {
    'empty': (lambda: np.zeros((50, 50), bool), 0),
    'bar-even': (lambda: bar(20), 1),
    'bar-odd': (lambda: bar(21), 1),
    'disk': (lambda: burn(Point(0, 0).buffer(30)), 0),
    'diagonal': (lambda: burn(LineString([(0, 0), (300, 200)]).buffer(12)), 1),
    'l-shape': (lambda: burn(LineString([(0, 0), (200, 0), (200, 150)]).buffer(10)), 1),
    'network': (lambda: burn(road_network().buffer(50)), 45),
    'bumpy': (bumpy, 1),
    'blocks': (blocks, 8),
    'mixed': (mixed, 40),
}


@jit(parallel=True, cache=True)
def _ratio(depth: np.ndarray, dist: np.ndarray):
    h, w = depth.shape
    out = np.zeros((h - 4, w - 4), np.float32)

    for i in prange(h - 4):
        for j in range(w - 4):
            climb = depth[i + 2, j + 2] - 1
            total = climb + dist[i + 2, j + 2]

            if climb >= 0 and total > 0:
                out[i, j] = climb / total

    return out


def center_raster(mask):
    state = np.pad(np.where(mask, 2, 1).astype(np.int8), 2, constant_values=1)
    depth = _flood(state)
    dist = _flood(_crests(depth))
    return _ratio(depth, dist)


@pytest.mark.parametrize('name', CASES)
def test_centerline(name):
    make, n_lines = CASES[name]
    mask = make()
    lines = crestline(mask, simplify=1.0)

    h, w = mask.shape
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_facecolor('#0b0b12')
    ax.imshow(np.ma.masked_where(~mask, center_raster(mask)), cmap='coolwarm', vmin=0, vmax=1, extent=(0, w, h, 0))

    for g in lines.geoms:
        ax.plot(*g.xy, 'o-', color='#00e5ff', lw=2, ms=3)
    ax.set_title(f'{name}: {len(lines.geoms)} line(s)', fontsize=10)

    output_dir = TEST_OUTPUT_DIR / 'test_centerline'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    assert len(lines.geoms) <= n_lines

    for g in lines.geoms:
        for x, y in g.coords:
            assert mask[int(y), int(x)]


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

    assert len(pieces) == 2


@pytest.mark.parametrize(
    'gap, kwargs, parts, length',
    [(5, dict(spur=0, dip=0.0), 1, 1096.5), (12, dict(), 1, 340.0), (13, dict(), 2, 727.5)],
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

    assert len(pieces) == parts
    assert lines.length == pytest.approx(length, rel=0.02)


@pytest.mark.parametrize(
    'make, n_pieces, n_dead, n_loops',
    [(bumpy, 1, 2, 0), (blocks, 1, 0, 4), (mixed, 1, 14, 5)],
    ids=['edge-noise', 'holes', 'mixed-widths'],
)
def test_topology(make, n_pieces, n_dead, n_loops):
    lines = crestline(make())
    ends = [c for g in shapely.get_parts(shapely.node(lines)) for c in (g.coords[0], g.coords[-1])]
    _, count = np.unique(np.round(np.array(ends), 3), axis=0, return_counts=True)

    assert len(shapely.get_parts(shapely.union_all(shapely.buffer(shapely.get_parts(lines), 0.5)))) == n_pieces
    assert (count == 1).sum() == n_dead
    assert len(shapely.get_parts(shapely.polygonize(shapely.get_parts(shapely.node(lines))))) == n_loops


@pytest.mark.parametrize('width, centre', [(20, 30.0), (21, 30.5)], ids=['even', 'odd'])
def test_centred(width, centre):
    assert np.allclose(shapely.get_coordinates(crestline(bar(width)))[:, 1], centre)


def has_cuda():
    try:
        from numba import cuda

        return cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not has_cuda(), reason='no CUDA device')
@pytest.mark.parametrize('name', CASES)
def test_cuda(name):
    mask = CASES[name][0]()

    assert crestline(mask).equals_exact(crestline(mask, device='cuda'), 0)
