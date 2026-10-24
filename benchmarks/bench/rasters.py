from typing import Any

import numpy as np
import shapely
from affine import Affine
from bench.scene import Shape
from numpy.typing import NDArray
from rasterio.features import rasterize
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from shapely import MultiPolygon
from shapely.geometry.base import BaseGeometry


def burn(polygon: Shape, raster_shape: tuple[int, ...], transform: Affine) -> NDArray[np.bool_]:
    return rasterize(
        [(polygon, 1)],
        out_shape=raster_shape,
        transform=transform,
    ).astype(bool)


def burn_polygon(polygon: Shape, mask: NDArray[np.bool_], transform: Affine) -> NDArray[np.bool_]:
    return burn(polygon, mask.shape, transform)


def _smooth_noise(
    random_generator: np.random.Generator,
    raster_shape: tuple[int, int],
    sigma: float,
    amplitude: float,
) -> NDArray[np.float64]:
    field = gaussian_filter(random_generator.normal(size=raster_shape), sigma)
    return field * amplitude / field.std()


def noisy_mask(
    polygon: Shape,
    seed: int,
    padding: int = 40,
    coarse_noise: float = 2.5,
    fine_noise: float = 0.8,
) -> tuple[NDArray[np.bool_], Affine]:
    random_generator = np.random.default_rng(seed + 1000)
    min_x, min_y, max_x, max_y = polygon.bounds
    raster_shape = (int(np.ceil(max_y - min_y)) + 2 * padding, int(np.ceil(max_x - min_x)) + 2 * padding)
    transform = from_origin(min_x - padding, max_y + padding, 1, 1)

    clean_mask = burn(polygon, raster_shape, transform)
    signed_distance = distance_transform_edt(clean_mask) - distance_transform_edt(~clean_mask)

    coarse_offset = _smooth_noise(random_generator, raster_shape, 4.0, coarse_noise)
    fine_offset = _smooth_noise(random_generator, raster_shape, 1.0, fine_noise)

    return signed_distance + coarse_offset + fine_offset > 0.5, transform


def vectorize(mask: NDArray[np.bool_], transform: Affine) -> Any | BaseGeometry:
    shape_records = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    polygons = [shapely.geometry.shape(geometry) for geometry, _value in shape_records]
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
