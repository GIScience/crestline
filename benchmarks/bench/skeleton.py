from typing import TypeAlias

import numpy as np
import shapely
from affine import Affine
from numba import njit
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from shapely import LineString
from shapely import MultiLineString
from skimage.morphology import skeletonize

Adjacency: TypeAlias = tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]

NEIGHBOR_STEPS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


@njit(cache=True)
def _adjacency(node_count: int, edges: NDArray[np.int64]) -> Adjacency:
    shifted_degree = np.zeros(node_count + 1, np.int64)

    for edge_index in range(len(edges)):
        shifted_degree[edges[edge_index, 0] + 1] += 1
        shifted_degree[edges[edge_index, 1] + 1] += 1

    adjacency_start = np.cumsum(shifted_degree)
    adjacency_node = np.empty(adjacency_start[-1], np.int64)
    adjacency_edge = np.empty(adjacency_start[-1], np.int64)
    insert_position = adjacency_start[:-1].copy()

    for edge_index in range(len(edges)):
        node_a, node_b = edges[edge_index, 0], edges[edge_index, 1]
        for node, neighbor in ((node_a, node_b), (node_b, node_a)):
            adjacency_node[insert_position[node]], adjacency_edge[insert_position[node]] = neighbor, edge_index
            insert_position[node] += 1

    return shifted_degree[1:].copy(), adjacency_start, adjacency_node, adjacency_edge


@njit(cache=True)
def _drop_spurs(
    node_count: int, edges: NDArray[np.int64], edge_lengths: NDArray[np.float64], max_spur_length: float
) -> NDArray[np.bool_]:
    degree, adjacency_start, adjacency_node, adjacency_edge = _adjacency(node_count, edges)
    edge_alive = np.ones(len(edges), np.bool_)
    chain_edges = np.empty(len(edges), np.int64)

    changed = True
    while changed:
        changed = False
        for start_node in range(node_count):
            if degree[start_node] != 1:
                continue

            current_node, previous_edge, chain_length, chain_edge_count = start_node, -1, 0.0, 0
            while True:
                next_node, next_edge = -1, -1
                for slot in range(adjacency_start[current_node], adjacency_start[current_node + 1]):
                    if edge_alive[adjacency_edge[slot]] and adjacency_edge[slot] != previous_edge:
                        next_node, next_edge = adjacency_node[slot], adjacency_edge[slot]
                        break

                if next_node < 0:
                    break

                chain_edges[chain_edge_count] = next_edge
                chain_edge_count += 1
                chain_length += edge_lengths[next_edge]
                current_node, previous_edge = next_node, next_edge

                if degree[current_node] != 2 or chain_length >= max_spur_length:
                    break

            if chain_length < max_spur_length and degree[current_node] != 2:
                for chain_position in range(chain_edge_count):
                    removed_edge = chain_edges[chain_position]
                    edge_alive[removed_edge] = False
                    degree[edges[removed_edge, 0]] -= 1
                    degree[edges[removed_edge, 1]] -= 1

                changed = True

    return edge_alive


def _pixel_edges(
    skeleton_raster: NDArray[np.bool_], pixel_rows: NDArray[np.intp], pixel_cols: NDArray[np.intp]
) -> NDArray[np.int64]:
    pixel_ids = np.full(skeleton_raster.shape, -1, np.int64)
    pixel_ids[pixel_rows, pixel_cols] = np.arange(len(pixel_rows))

    edge_blocks: list[NDArray[np.int64]] = []
    for row_step, col_step in NEIGHBOR_STEPS:
        is_linked = skeleton_raster[pixel_rows + row_step, pixel_cols + col_step]

        if row_step and col_step:
            column_neighbor = skeleton_raster[pixel_rows, pixel_cols + col_step]
            row_neighbor = skeleton_raster[pixel_rows + row_step, pixel_cols]
            is_linked &= ~column_neighbor & ~row_neighbor

        linked_rows, linked_cols = pixel_rows[is_linked], pixel_cols[is_linked]
        source_ids = pixel_ids[linked_rows, linked_cols]
        target_ids = pixel_ids[linked_rows + row_step, linked_cols + col_step]
        edge_blocks.append(np.c_[source_ids, target_ids])

    return np.concatenate(edge_blocks)


def skeleton(mask: NDArray[np.bool_], transform: Affine, prune_factor: float = 2.0) -> LineString | MultiLineString:
    skeleton_raster = np.pad(skeletonize(mask), 1)
    pixel_rows, pixel_cols = np.nonzero(skeleton_raster)

    if len(pixel_rows) < 2:
        return MultiLineString()

    edges = _pixel_edges(skeleton_raster, pixel_rows, pixel_cols)
    row_deltas = pixel_rows[edges[:, 0]] - pixel_rows[edges[:, 1]]
    col_deltas = pixel_cols[edges[:, 0]] - pixel_cols[edges[:, 1]]
    edge_lengths = np.hypot(row_deltas, col_deltas)

    distance_to_edge = distance_transform_edt(mask)[pixel_rows - 1, pixel_cols - 1]
    max_spur_length = prune_factor * np.median(distance_to_edge)
    edges = edges[_drop_spurs(len(pixel_rows), edges, edge_lengths, max_spur_length)]

    if len(edges) == 0:
        return MultiLineString()

    node_coordinates = np.c_[transform * (pixel_cols - 0.5, pixel_rows - 0.5)]
    return shapely.line_merge(shapely.multilinestrings(shapely.linestrings(node_coordinates[edges])))
