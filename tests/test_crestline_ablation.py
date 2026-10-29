from dataclasses import dataclass
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import shapely
from shapely import LineString
from shapely import MultiLineString
from shapely import buffer
from shapely import get_parts
from shapely import union_all

from crestline.crestline import _center
from crestline.crestline import _crests
from crestline.crestline import _despur
from crestline.crestline import _flood
from crestline.crestline import _links
from crestline.crestline import _prune
from crestline.crestline import _thin
from crestline.crestline import crestline

MODS = [(55, 64), (19, 100), (55, 136), (19, 172), (55, 208), (19, 244)]


@dataclass
class Scene:
    mask: np.ndarray
    truth: MultiLineString


def _scene(shape, *roads):
    mask, axes = np.zeros(shape, bool), []

    for top, width, left, right in roads:
        mask[top : top + width, left:right] = True
        axes.append(LineString([(left, top + width / 2), (right, top + width / 2)]))

    return Scene(mask, MultiLineString(axes))


@pytest.fixture(scope='module')
def parallel_roads():
    return _scene((102, 410), (5, 60, 5, 405), (85, 12, 5, 405))


@pytest.fixture(scope='module')
def noisy_road():
    road = _scene((80, 300), (25, 30, 10, 290))

    for row, col in MODS:
        road.mask[row : row + 6, col : col + 6] = True

    return road


@pytest.fixture(scope='module')
def even_and_odd_roads():
    return [_scene((60, 200), (20, width, 10, 190)) for width in (20, 21)]


@pytest.fixture(scope='module')
def occluded_road():
    road = _scene((81, 420), (20, 41, 10, 410))
    road.mask[36:39, 150:156] = False
    return road


@pytest.fixture(scope='module')
def diagonal_road():
    rows, cols = np.indices((200, 200))
    return Scene(
        mask=np.abs(rows - cols) <= 21,
        truth=MultiLineString([[(0, 0), (200, 200)]]),
    )


def connected_components(lines):
    parts = get_parts(lines)
    parts = buffer(parts, 0.5)
    parts = union_all(parts)
    parts = get_parts(parts)
    return len(parts)


def loops(lines):
    parts = shapely.get_parts(shapely.node(lines))
    parts = shapely.polygonize(parts)
    parts = shapely.get_parts(parts)
    return len(parts)


def dead_ends(lines):
    parts = shapely.get_parts(shapely.node(lines))
    ends = [c for g in parts for c in (g.coords[0], g.coords[-1])]
    _, count = np.unique(np.round(ends, 3), axis=0, return_counts=True)
    return int((count == 1).sum())


def offset(lines, truth):
    samples = shapely.segmentize(lines, 1)
    samples = shapely.get_coordinates(samples)
    samples = shapely.points(samples)
    return float(shapely.distance(samples, truth).mean())


def vertices(lines, axis):
    lines = shapely.get_coordinates(lines)
    points = shapely.points(lines)
    return int((shapely.distance(points, axis) < 1).sum())


def test_without_the_ridge_test_links_bridge_the_gap_between_parallel_roads(parallel_roads):
    full, ablated = crestline(parallel_roads.mask), crestline(parallel_roads.mask, dip=0)

    assert connected_components(full) == 2
    assert loops(full) == 0
    assert connected_components(ablated) == 1
    assert loops(ablated) > 0
    assert offset(ablated, parallel_roads.truth) > offset(full, parallel_roads.truth)


def test_without_shortcut_pruning_loops_and_length_explode_along_a_noisy_edge(noisy_road):
    full = crestline(noisy_road.mask)
    with patch('crestline.crestline._prune', partial(_prune, cycle=0)):
        ablated = crestline(noisy_road.mask)

    assert loops(full) == 0
    assert loops(ablated) > 10
    assert ablated.length > 10 * full.length


def test_without_the_spur_depth_guard_every_dead_end_road_is_eaten(parallel_roads):
    def no_depth_guard(points, pairs, heights, near, spur, tip):
        return _despur(points, pairs, heights, near, spur, np.inf)

    full = crestline(parallel_roads.mask)
    with patch('crestline.crestline._despur', no_depth_guard):
        ablated = crestline(parallel_roads.mask)

    assert connected_components(full) == 2
    assert connected_components(ablated) == 0


def test_without_spur_pruning_every_bump_on_a_noisy_edge_leaves_a_dead_end(noisy_road):
    full, ablated = crestline(noisy_road.mask), crestline(noisy_road.mask, spur=0)

    assert dead_ends(full) == 2
    assert dead_ends(ablated) == dead_ends(full) + len(MODS)


def test_with_a_constant_thinning_radius_only_wide_roads_are_oversampled(parallel_roads):
    def constant_radius(rows, cols, h, w, radius):
        return _thin(rows, cols, h, w, np.full_like(radius, 3))

    full = crestline(parallel_roads.mask)
    with patch('crestline.crestline._thin', constant_radius):
        ablated = crestline(parallel_roads.mask)
    wide, narrow = parallel_roads.truth.geoms

    assert vertices(ablated, wide) > 2 * vertices(full, wide)
    assert vertices(ablated, narrow) == vertices(full, narrow)


def test_with_a_constant_10px_link_length_wide_roads_fall_apart(parallel_roads):
    def constant_length(points, heights, near):
        return _links(points, np.full_like(heights, 5), near)

    full = crestline(parallel_roads.mask)
    with patch('crestline.crestline._links', constant_length):
        ablated = crestline(parallel_roads.mask)
    wide, narrow = parallel_roads.truth.geoms

    assert vertices(full, wide) > 0
    assert vertices(ablated, wide) == 0
    assert vertices(ablated, narrow) == vertices(full, narrow)


def test_with_a_3x3_crest_window_terraces_add_a_crest_fragment_to_every_noise_spur(noisy_road):
    full = crestline(noisy_road.mask, spur=0)
    with patch('crestline.crestline._crests', partial(_crests, reach=1)):
        ablated = crestline(noisy_road.mask, spur=0)

    assert len(shapely.get_coordinates(ablated)) >= len(shapely.get_coordinates(full)) + len(MODS)


def test_without_subpixel_refinement_the_offset_rises_by_a_quarter_pixel_on_average(even_and_odd_roads):
    def own_pixel_only(points, heights, crest, depth):
        own = np.zeros_like(crest)
        own[points[:, 0] + 2, points[:, 1] + 2] = True
        return _center(points, heights, own, depth)

    full = np.mean([offset(crestline(road.mask), road.truth) for road in even_and_odd_roads])
    with patch('crestline.crestline._center', own_pixel_only):
        ablated = np.mean([offset(crestline(road.mask), road.truth) for road in even_and_odd_roads])

    assert ablated - full == pytest.approx(0.25, abs=0.05)


def test_with_one_flood_neighbourhood_only_diagonal_roads_change_their_sampling(diagonal_road, parallel_roads):
    octagonal = vertices(crestline(diagonal_road.mask), diagonal_road.truth)
    axis_aligned = crestline(parallel_roads.mask)

    with patch('crestline.crestline._flood', partial(_flood, neighbourhood=8)):
        assert vertices(crestline(diagonal_road.mask), diagonal_road.truth) > octagonal
        assert crestline(parallel_roads.mask).equals(axis_aligned)

    with patch('crestline.crestline._flood', partial(_flood, neighbourhood=4)):
        assert vertices(crestline(diagonal_road.mask), diagonal_road.truth) < octagonal
        assert crestline(parallel_roads.mask).equals(axis_aligned)


def test_without_opening_occluders_a_parked_car_leaves_a_false_loop(occluded_road):
    full, ablated = crestline(occluded_road.mask), crestline(occluded_road.mask, hole=0)

    assert loops(full) == 0
    assert loops(ablated) > 0
