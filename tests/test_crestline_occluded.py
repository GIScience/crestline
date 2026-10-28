import numpy as np
import pytest
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import LineString
from shapely import MultiLineString

from crestline.crestline import _faces
from crestline.crestline import crestline


def _burn(geom, pad=20):
    minx, miny, maxx, maxy = geom.bounds
    shape = (int(np.ceil(maxy - miny)) + 2 * pad, int(np.ceil(maxx - minx)) + 2 * pad)
    transform = from_origin(minx - pad, maxy + pad, 1, 1)
    return rasterize([(geom, 1)], out_shape=shape, transform=transform).astype(bool)


def _road(width=41, length=400):
    mask = np.zeros((width + 40, length + 20), bool)
    mask[20 : 20 + width, 10 : 10 + length] = True
    return mask


def _occlude(mask, *boxes):
    mask = mask.copy()
    for r, c, h, w in boxes:
        mask[r : r + h, c : c + w] = False
    return mask


def _topology(lines):
    noded = shapely.get_parts(shapely.node(lines))
    ends = [c for g in noded for c in (g.coords[0], g.coords[-1])]
    _, count = np.unique(np.round(np.array(ends), 3), axis=0, return_counts=True)
    pieces = shapely.get_parts(shapely.union_all(shapely.buffer(shapely.get_parts(lines), 0.5)))
    return len(pieces), int((count == 1).sum()), len(shapely.get_parts(shapely.polygonize(noded)))


def _ring():
    return _burn(shapely.box(0, 0, 200, 200).exterior.buffer(15))


OCCLUDED = {
    'car': (lambda: _occlude(_road(), (36, 150, 3, 6)), 2),
    'centre': (lambda: _occlude(_road(), (39, 200, 3, 3)), 2),
    'trees': (lambda: _occlude(_road(), (24, 60, 3, 3), (52, 120, 3, 3), (25, 180, 2, 2), (54, 240, 2, 3)), 2),
    'crossing': (
        lambda: _occlude(
            _burn(MultiLineString([[(0, 100), (300, 100)], [(150, 0), (150, 200)]]).buffer(18)),
            (125, 168, 3, 3),
            (130, 100, 2, 4),
        ),
        4,
    ),
}


def test_faces():
    points = np.array([[0, 0], [0, 10], [10, 10], [10, 0], [0, 20], [10, 20], [20, 0]])
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5], [5, 2], [3, 6]])

    face, after, area = _faces(points, pairs)

    assert sorted(area) == [-200, 100, 100]
    assert face[14] == face[15] == np.argmin(area)
    assert all(face[after[h]] == face[h] for h in range(len(face)))


@pytest.mark.parametrize('name', OCCLUDED)
def test_occluders_open_their_loops(name):
    make, n_dead = OCCLUDED[name]
    mask = make()

    lines, kept = crestline(mask), crestline(mask, hole=0)

    assert _topology(kept)[2] >= 1
    assert _topology(lines) == (1, n_dead, 0)
    assert _topology(kept)[1] == n_dead
    for x, y in shapely.get_coordinates(lines):
        assert mask[int(y), int(x)]


def test_occluders_leave_the_road_axis_alone():
    mask = _occlude(_road(), (24, 60, 3, 3), (52, 120, 3, 3), (25, 180, 2, 2), (54, 240, 2, 3))

    rows = shapely.get_coordinates(crestline(mask))[:, 1]

    assert np.abs(rows - 40.5).max() <= 7


@pytest.mark.parametrize('mask', [_ring(), _occlude(_ring(), (15, 100, 3, 3))], ids=['block', 'block-with-occluder'])
def test_blocks_keep_their_loop(mask):
    assert _topology(crestline(mask)) == (1, 0, 1)


def test_occluder_capping_a_road_end_is_left_alone():
    mask = _occlude(_burn(LineString([(0, 0), (300, 0)]).buffer(20)), (35, 315, 3, 3))

    assert _topology(crestline(mask)) == _topology(crestline(mask, hole=0))


def test_tips_reach_the_cap_centres():
    lines = crestline(_burn(LineString([(0, 0), (200, 0)]).buffer(12)))

    assert sorted(shapely.get_coordinates(lines)[[0, -1], 0]) == pytest.approx([32, 232], abs=1)
