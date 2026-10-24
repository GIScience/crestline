from collections.abc import Callable
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
import shapely
from affine import Affine
from bench.environment import logger
from bench.rasters import burn_polygon
from bench.rasters import vectorize
from bench.scene import Shape
from bench.skeleton import skeleton
from centerline.geometry import Centerline
from numpy.typing import NDArray
from pygeoops import centerline as pygeoops_centerline
from shapely import Geometry
from shapely import MultiLineString

from crestline import crestline

CUDA_METHOD = 'crestline (CUDA)'
POLYGON_RESOLUTION_M = 0.5

Extractor: TypeAlias = Callable[[Shape, NDArray[np.bool_], Affine], Geometry]


class Method(NamedTuple):
    on_polygon: Extractor
    on_mask: Extractor


MethodRegistry: TypeAlias = dict[str, Method]


def voronoi_centerline(geometry: Shape) -> Geometry:
    if geometry.geom_type == 'MultiPolygon':
        return shapely.union_all([voronoi_centerline(part) for part in geometry.geoms])

    try:
        return Centerline(geometry, interpolation_distance=1.0).geometry
    except Exception as error:
        logger.debug(f'Voronoi centerline failed on a {geometry.area:.1f} m² part: {error}')
        return MultiLineString()


def methods(with_cuda: bool = False) -> MethodRegistry:
    registry: MethodRegistry = {
        'crestline': Method(
            on_polygon=lambda polygon, mask, transform: crestline(polygon, resolution=POLYGON_RESOLUTION_M),
            on_mask=lambda polygon, mask, transform: crestline(mask, transform=transform),
        ),
        'pygeoops.centerline': Method(
            on_polygon=lambda polygon, mask, transform: pygeoops_centerline(polygon),
            on_mask=lambda polygon, mask, transform: pygeoops_centerline(vectorize(mask, transform)),
        ),
        'skeletonize + prune': Method(
            on_polygon=lambda polygon, mask, transform: skeleton(burn_polygon(polygon, mask, transform), transform),
            on_mask=lambda polygon, mask, transform: skeleton(mask, transform),
        ),
        'centerline (Voronoi)': Method(
            on_polygon=lambda polygon, mask, transform: voronoi_centerline(polygon),
            on_mask=lambda polygon, mask, transform: voronoi_centerline(vectorize(mask, transform)),
        ),
    }

    if with_cuda:
        registry[CUDA_METHOD] = Method(
            on_polygon=lambda polygon, mask, transform: crestline(
                polygon, resolution=POLYGON_RESOLUTION_M, device='cuda'
            ),
            on_mask=lambda polygon, mask, transform: crestline(mask, transform=transform, device='cuda'),
        )

    return registry
