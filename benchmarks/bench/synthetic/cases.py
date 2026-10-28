from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import shapely
from bench.metrics import holes
from bench.metrics import true_dead_ends
from bench.rasters import noisy_mask
from bench.scene import Scene
from bench.scene import Shape
from shapely import LineString
from shapely import MultiLineString

ROAD_HALF_WIDTH_M = 15.0
MIXED_HALF_WIDTHS_M: list[float] = [6.0, 9.0, 14.0, 20.0, 30.0]


class Case(NamedTuple):
    centerlines: list[LineString]
    half_widths: list[float]
    polygon: Shape


def _main_road(random_generator: np.random.Generator, road_index: int, extent: float) -> LineString:
    along_positions = np.linspace(0, extent, 5)
    base_offset = random_generator.uniform(0.15, 0.85) * extent
    drift = np.cumsum(random_generator.normal(0, 50, 5))
    across_positions = base_offset + drift

    if road_index % 2 == 0:
        return LineString(np.c_[along_positions, across_positions])

    return LineString(np.c_[across_positions, along_positions])


def _side_road(random_generator: np.random.Generator, main_roads: list[LineString]) -> LineString:
    parent_road = main_roads[random_generator.integers(len(main_roads))]
    anchor = parent_road.interpolate(random_generator.uniform(0.1, 0.9), normalized=True)
    heading, road_length = random_generator.uniform(0, 2 * np.pi), random_generator.uniform(80, 220)
    tip = (anchor.x + road_length * np.cos(heading), anchor.y + road_length * np.sin(heading))
    return LineString([(anchor.x, anchor.y), tip])


def road_network(
    seed: int,
    main_road_count: int = 5,
    side_road_count: int = 8,
    extent: float = 1000.0,
) -> list[LineString]:
    random_generator = np.random.default_rng(seed)
    main_roads = [_main_road(random_generator, road_index, extent) for road_index in range(main_road_count)]
    side_roads = [_side_road(random_generator, main_roads) for _ in range(side_road_count)]
    return main_roads + side_roads


def _tributary(random_generator: np.random.Generator, trunk: LineString) -> tuple[LineString, tuple[float, float]]:
    junction_fraction = random_generator.uniform(0.3, 0.9)
    junction = trunk.interpolate(junction_fraction, normalized=True)
    heading = random_generator.choice([-1, 1]) * random_generator.uniform(0.6, 1.3) + np.pi
    tributary_length = random_generator.uniform(150, 300)

    tributary_progress = np.linspace(0, 1, 20)
    wiggle = np.cumsum(random_generator.normal(0, 4, 20))
    x_coords = junction.x + tributary_progress * tributary_length * np.cos(heading) - wiggle * np.sin(heading)
    y_coords = junction.y + tributary_progress * tributary_length * np.sin(heading) + wiggle * np.cos(heading)

    return LineString(np.c_[x_coords, y_coords]), (5 + 12.5 * junction_fraction, 5.0)


def river(
    seed: int,
    extent: float = 1000.0,
    tributary_count: int = 4,
) -> tuple[list[LineString], list[tuple[float, float]]]:
    random_generator = np.random.default_rng(seed)
    trunk_progress = np.linspace(0, 1, 60)
    trunk_meander = np.cumsum(random_generator.normal(0, 12, 60))
    trunk = LineString(np.c_[trunk_progress * extent, extent * 0.5 + trunk_meander])
    centerlines, radii = [trunk], [(5.0, 30.0)]

    for _ in range(tributary_count):
        tributary, tributary_radii = _tributary(random_generator, trunk)
        centerlines.append(tributary)
        radii.append(tributary_radii)

    return centerlines, radii


def tapered(line: LineString, start_radius: float, end_radius: float, step: float = 2.0) -> Shape:
    station_count = max(int(line.length / step), 2)
    stations = [line.interpolate(distance) for distance in np.linspace(0, line.length, station_count)]
    radii = np.linspace(start_radius, end_radius, station_count)
    segments = zip(stations[:-1], stations[1:], radii)

    return shapely.union_all(
        [LineString([(start.x, start.y), (end.x, end.y)]).buffer(radius) for start, end, radius in segments]
    )


def uniform_width_case(centerlines: list[LineString]) -> Case:
    half_widths = [ROAD_HALF_WIDTH_M] * len(centerlines)
    polygon = MultiLineString(centerlines).buffer(ROAD_HALF_WIDTH_M)
    return Case(centerlines, half_widths, polygon)


def _roads_case(seed: int) -> Case:
    return uniform_width_case(road_network(seed))


def _large_roads_case(seed: int) -> Case:
    return uniform_width_case(road_network(seed, main_road_count=20, side_road_count=32, extent=4000.0))


def _mixed_case(seed: int) -> Case:
    centerlines = road_network(seed)
    half_widths = list(np.random.default_rng(seed + 100).choice(MIXED_HALF_WIDTHS_M, len(centerlines)))
    polygon = shapely.union_all([line.buffer(half_width) for line, half_width in zip(centerlines, half_widths)])
    return Case(centerlines, half_widths, polygon)


def _river_case(seed: int) -> Case:
    centerlines, radii = river(seed)
    polygon = shapely.union_all([tapered(line, *radius_pair) for line, radius_pair in zip(centerlines, radii)])
    return Case(centerlines, [max(radius_pair) for radius_pair in radii], polygon)


CASE_BUILDERS: dict[str, Callable[[int], Case]] = {
    'roads': _roads_case,
    'roads-large': _large_roads_case,
    'mixed': _mixed_case,
    'river': _river_case,
}


def make_case(case_kind: str, seed: int) -> Case:
    if case_kind not in CASE_BUILDERS:
        raise ValueError(f'Unknown case kind: {case_kind}')

    return CASE_BUILDERS[case_kind](seed)


def make_scene(case_kind: str, seed: int) -> Scene:
    case = make_case(case_kind, seed)
    mask, transform = noisy_mask(case.polygon, seed)
    return Scene(
        fields={'case': case_kind, 'seed': seed},
        label=f'{case_kind} seed {seed}',
        polygon=case.polygon,
        mask=mask,
        transform=transform,
        centerlines=case.centerlines,
        true_dead_ends=true_dead_ends(case.centerlines, case.half_widths),
        holes=holes(case.polygon),
    )
