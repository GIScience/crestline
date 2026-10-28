import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import shapely
from bench.environment import DATA_DIR
from bench.environment import logger
from bench.metrics import holes
from bench.metrics import true_dead_ends
from bench.rasters import noisy_mask
from bench.scene import Scene
from bench.scene import Shape
from numpy.typing import NDArray
from shapely import Geometry
from shapely import LineString
from shapely import MultiLineString
from shapely import Polygon
from shapely import STRtree

Coordinate: TypeAlias = tuple[float, float]
Noise: TypeAlias = tuple[float, float]

ROADS_PATH = DATA_DIR / 'ldz_road_network.gpkg'
CENTERLINES_PATH = DATA_DIR / 'ldz_road_network_centerlines.gpkg'
GEOPACKAGE_HEADER_BYTES = 8
GEOPACKAGE_ENVELOPE_BYTES: dict[int, int] = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
HALF_WIDTH_STEP_M = 2.0


@dataclass
class Network:
    polygon: Shape
    roads: NDArray[np.object_]
    half_widths: NDArray[np.float64]
    tree: STRtree


def _geometry_offset(blob: bytes) -> int:
    return GEOPACKAGE_HEADER_BYTES + GEOPACKAGE_ENVELOPE_BYTES[(blob[3] >> 1) & 0b111]


def read_geopackage(path: Path) -> tuple[list[Geometry], int]:
    with closing(sqlite3.connect(path)) as connection:
        ((table_name, column_name, epsg),) = connection.execute(
            'SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns'
        ).fetchall()
        blobs: list[bytes] = [
            blob for (blob,) in connection.execute(f'SELECT "{column_name}" FROM "{table_name}"') if blob
        ]

    offsets = [_geometry_offset(blob) for blob in blobs]
    return [shapely.from_wkb(bytes(blob[offset:])) for blob, offset in zip(blobs, offsets)], epsg


def polygonal(geometry: Geometry) -> Shape:
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry

    parts = shapely.get_parts(geometry)
    is_polygonal = np.isin(shapely.get_type_id(parts), (3, 6))
    return shapely.multipolygons(shapely.get_parts(parts[is_polygonal]))


def _boundary_edges(polygon: Shape) -> NDArray[np.object_]:
    rings = shapely.get_rings(shapely.get_parts(polygon))
    coordinates, ring_index = shapely.get_coordinates(rings, return_index=True)
    is_edge = ring_index[1:] == ring_index[:-1]
    return shapely.linestrings(np.stack([coordinates[:-1][is_edge], coordinates[1:][is_edge]], axis=1))


def _stations(roads: NDArray[np.object_]) -> tuple[NDArray[np.object_], NDArray[np.int64]]:
    station_counts = np.maximum(np.ceil(shapely.length(roads) / HALF_WIDTH_STEP_M).astype(np.int64), 1)
    owner = np.repeat(np.arange(len(roads)), station_counts)
    first_station = np.repeat(np.cumsum(station_counts) - station_counts, station_counts)
    fractions = (np.arange(len(owner)) - first_station + 0.5) / station_counts[owner]
    return shapely.line_interpolate_point(roads[owner], fractions, normalized=True), owner


def road_half_widths(polygon: Shape, roads: NDArray[np.object_]) -> NDArray[np.float64]:
    edges = _boundary_edges(polygon)
    stations, owner = _stations(roads)

    (station_index, _edge_index), distances = STRtree(edges).query_nearest(stations, return_distance=True)
    distance_to_edge = np.empty(len(stations))
    distance_to_edge[station_index] = distances
    return pd.Series(distance_to_edge).groupby(owner).median().to_numpy()


def load_network() -> Network:
    started = time.perf_counter()
    polygons, polygon_epsg = read_geopackage(ROADS_PATH)
    centerlines, centerline_epsg = read_geopackage(CENTERLINES_PATH)
    if polygon_epsg != centerline_epsg:
        raise ValueError(f'Road polygons are in EPSG:{polygon_epsg} but centerlines in EPSG:{centerline_epsg}')

    polygon = polygonal(polygons[0] if len(polygons) == 1 else shapely.union_all(polygons))
    roads = shapely.get_parts(shapely.get_parts(np.asarray(centerlines, dtype=object)))
    roads = roads[shapely.length(roads) > 0]
    half_widths = road_half_widths(polygon, roads)

    logger.info(
        f'Loaded {len(roads):,} centerlines over {shapely.length(roads).sum() / 1000:,.0f} km and '
        f'{polygon.area / 1e6:.1f} km² of road surface in EPSG:{polygon_epsg}, median half-width '
        f'{np.median(half_widths):.1f} m, in {time.perf_counter() - started:.1f} s'
    )
    return Network(polygon, roads, half_widths, STRtree(roads))


def network_center(network: Network) -> Coordinate:
    center_point = shapely.centroid(MultiLineString(list(network.roads)))
    return center_point.x, center_point.y


def _clip_roads(network: Network, buffer: Polygon) -> tuple[list[LineString], NDArray[np.float64]]:
    candidates = network.tree.query(buffer, predicate='intersects')
    pieces, owner = shapely.get_parts(shapely.intersection(network.roads[candidates], buffer), return_index=True)
    is_line = (shapely.get_type_id(pieces) == 1) & (shapely.length(pieces) > 0)
    return list(pieces[is_line]), network.half_widths[candidates][owner[is_line]]


def make_scene(network: Network, center: Coordinate, radius_km: int, seed: int, noise: Noise) -> Scene:
    buffer = shapely.Point(center).buffer(radius_km * 1000.0, quad_segs=64)
    polygon = polygonal(shapely.intersection(network.polygon, buffer))
    centerlines, half_widths = _clip_roads(network, buffer)

    coarse_noise, fine_noise = noise
    mask, transform = noisy_mask(polygon, seed, coarse_noise=coarse_noise, fine_noise=fine_noise)

    return Scene(
        fields={'radius_km': radius_km},
        label=f'{radius_km} km',
        polygon=polygon,
        mask=mask,
        transform=transform,
        centerlines=centerlines,
        true_dead_ends=true_dead_ends(centerlines, half_widths),
        holes=holes(polygon),
    )
