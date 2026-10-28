import numpy as np
import shapely
from bench.scene import Shape
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely import Geometry
from shapely import LineString
from shapely import MultiLineString
from shapely import Polygon
from shapely import STRtree

ENDPOINT_SNAP_M = 0.5


def as_lines(geometry: Geometry | None) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []

    lines: list[LineString] = []
    for part in shapely.get_parts(geometry):
        if part.geom_type == 'LineString':
            lines.append(part)
        elif part.geom_type in ('MultiLineString', 'GeometryCollection'):
            lines.extend(as_lines(part))

    return [line for line in lines if line.length > 0]


def _noded_segments(lines: list[LineString]) -> NDArray[np.object_]:
    return shapely.get_parts(shapely.node(MultiLineString(lines)))


def _end_keys(lines: list[LineString]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    segments = _noded_segments(lines)
    endpoints = [coord for segment in segments for coord in (segment.coords[0], segment.coords[-1])]
    snapped = np.round(np.array(endpoints) / ENDPOINT_SNAP_M).astype(np.int64)
    return np.unique(snapped, axis=0, return_counts=True)


def dead_ends(lines: list[LineString]) -> int:
    if not lines:
        return 0

    _, occurrences = _end_keys(lines)
    return int((occurrences == 1).sum())


def end_points(lines: list[LineString]) -> NDArray[np.float64]:
    if not lines:
        return np.zeros((0, 2))

    grid_keys, occurrences = _end_keys(lines)
    return grid_keys[occurrences == 1] * ENDPOINT_SNAP_M


def loops(lines: list[LineString]) -> int:
    if not lines:
        return 0

    return len(shapely.get_parts(shapely.polygonize(_noded_segments(lines))))


def holes(polygon: Shape, min_area: float = 50.0) -> int:
    return sum(1 for part in shapely.get_parts(polygon) for ring in part.interiors if Polygon(ring).area >= min_area)


def true_dead_ends(lines: list[LineString], half_widths: list[float] | NDArray[np.float64]) -> int:
    if not lines:
        return 0

    line_array = np.asarray(lines, dtype=object)
    half_width_array = np.asarray(half_widths, dtype=np.float64)
    endpoints = np.concatenate([shapely.get_point(line_array, 0), shapely.get_point(line_array, -1)])
    owner = np.tile(np.arange(len(line_array)), 2)

    reach = half_width_array.max() + 0.5 * half_width_array[owner] + 1.0
    endpoint_index, line_index = STRtree(line_array).query(endpoints, predicate='dwithin', distance=reach)
    is_other = line_index != owner[endpoint_index]
    endpoint_index, line_index = endpoint_index[is_other], line_index[is_other]

    gap = shapely.distance(endpoints[endpoint_index], line_array[line_index])
    is_joined = gap <= half_width_array[line_index] + 0.5 * half_width_array[owner[endpoint_index]]

    is_dead_end = ~shapely.is_closed(line_array)[owner]
    is_dead_end[endpoint_index[is_joined]] = False
    return int(is_dead_end.sum())


def _sample(lines: list[LineString], step: float = 0.5) -> NDArray[np.float64]:
    sampled: list[NDArray[np.float64]] = []
    for line in lines:
        point_count = max(int(np.ceil(line.length / step)), 1) + 1
        distances = np.linspace(0, line.length, point_count)
        sampled.append(shapely.get_coordinates(shapely.line_interpolate_point(line, distances)))

    return np.concatenate(sampled) if sampled else np.zeros((0, 2))


def _f1(precision: float, recall: float) -> float:
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def score(predicted_lines: list[LineString], true_lines: list[LineString]) -> dict[str, float]:
    true_points, predicted_points = _sample(true_lines), _sample(predicted_lines)
    if len(predicted_points) == 0:
        return {'f5': 0.0, 'f2': 0.0, 'offset': float('nan')}

    predicted_to_truth = cKDTree(true_points).query(predicted_points)[0]
    truth_to_predicted = cKDTree(predicted_points).query(true_points)[0]

    metrics: dict[str, float] = {}
    for tolerance in (2, 5):
        precision = (predicted_to_truth <= tolerance).mean()
        recall = (truth_to_predicted <= tolerance).mean()
        metrics[f'f{tolerance}'] = _f1(precision, recall)

    metrics['offset'] = float(predicted_to_truth[predicted_to_truth <= 5].mean())
    return metrics
