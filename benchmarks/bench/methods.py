from collections.abc import Callable
from typing import Any
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
import shapely
from affine import Affine
from bench.environment import logger
from bench.rasters import rasterize_polygon
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


class Method(NamedTuple):
    on_polygon: Callable[[Shape], Geometry]
    on_mask: Callable[[NDArray[np.bool_], Affine], Geometry]
    single_run: bool = False


MethodRegistry: TypeAlias = dict[str, Method]


def voronoi_centerline(geometry: Shape, interpolation_distance: float) -> Geometry:
    if geometry.geom_type == 'MultiPolygon':
        return shapely.union_all([voronoi_centerline(part, interpolation_distance) for part in geometry.geoms])

    try:
        return Centerline(geometry, interpolation_distance=interpolation_distance).geometry
    except Exception as error:
        logger.debug(f'Voronoi centerline failed on a {geometry.area:.1f} m² part: {error}')
        return MultiLineString()


def _crestline_method(**options: Any) -> Method:
    return Method(
        on_polygon=lambda polygon: crestline(polygon, resolution=POLYGON_RESOLUTION_M, **options),
        on_mask=lambda mask, transform: crestline(mask, transform=transform, **options),
    )


def methods(with_cuda: bool = False) -> MethodRegistry:
    registry: MethodRegistry = {
        'crestline': _crestline_method(),
        'pygeoops.centerline': Method(
            on_polygon=lambda polygon: pygeoops_centerline(polygon, densify_distance=POLYGON_RESOLUTION_M),
            on_mask=lambda mask, transform: pygeoops_centerline(vectorize(mask, transform)),
        ),
        'skeletonize + prune': Method(
            on_polygon=lambda polygon: skeleton(*rasterize_polygon(polygon, POLYGON_RESOLUTION_M)),
            on_mask=skeleton,
        ),
        'centerline (Voronoi)': Method(
            on_polygon=lambda polygon: voronoi_centerline(polygon, POLYGON_RESOLUTION_M),
            on_mask=lambda mask, transform: voronoi_centerline(vectorize(mask, transform), abs(transform.a)),
            single_run=True,
        ),
    }

    if with_cuda:
        registry[CUDA_METHOD] = _crestline_method(device='cuda')

    return registry
