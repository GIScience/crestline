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
from skimage.morphology import skeletonize


def burn(polygon: Shape, raster_shape: tuple[int, ...], transform: Affine) -> NDArray[np.bool_]:
    return rasterize([(polygon, 1)], out_shape=raster_shape, transform=transform).astype(bool)


def raster_grid(polygon: Shape, resolution: float, padding: int) -> tuple[tuple[int, int], Affine]:
    min_x, min_y, max_x, max_y = polygon.bounds
    raster_shape = (
        int(np.ceil((max_y - min_y) / resolution)) + 2 * padding,
        int(np.ceil((max_x - min_x) / resolution)) + 2 * padding,
    )
    transform = from_origin(min_x - padding * resolution, max_y + padding * resolution, resolution, resolution)
    return raster_shape, transform


def rasterize_polygon(polygon: Shape, resolution: float, padding: int = 2) -> tuple[NDArray[np.bool_], Affine]:
    raster_shape, transform = raster_grid(polygon, resolution, padding)
    return burn(polygon, raster_shape, transform), transform


def _smooth_noise(
    random_generator: np.random.Generator, raster_shape: tuple[int, int], sigma: float, amplitude: float
) -> NDArray[np.float64]:
    field = gaussian_filter(random_generator.normal(size=raster_shape), sigma)
    return field * amplitude / field.std()


def _local_half_width(clean_mask: NDArray[np.bool_], inside_distance: NDArray[np.float64]) -> NDArray[np.float64]:
    medial_axis = skeletonize(clean_mask)
    nearest_rows, nearest_cols = distance_transform_edt(~medial_axis, return_distances=False, return_indices=True)
    return inside_distance[nearest_rows, nearest_cols]


def noisy_mask(
    polygon: Shape,
    seed: int,
    padding: int = 40,
    coarse_noise: float = 0.15,
    fine_noise: float = 0.05,
    max_shift: float = 0.5,
) -> tuple[NDArray[np.bool_], Affine]:
    random_generator = np.random.default_rng(seed + 1000)
    raster_shape, transform = raster_grid(polygon, 1.0, padding)

    clean_mask = burn(polygon, raster_shape, transform)
    inside_distance = distance_transform_edt(clean_mask)
    signed_distance = inside_distance - distance_transform_edt(~clean_mask)

    coarse_offset = _smooth_noise(random_generator, raster_shape, 4.0, coarse_noise)
    fine_offset = _smooth_noise(random_generator, raster_shape, 1.0, fine_noise)
    relative_offset = np.clip(coarse_offset + fine_offset, -max_shift, max_shift)
    half_width = _local_half_width(clean_mask, inside_distance)

    return signed_distance + relative_offset * half_width > 0.5, transform


def vectorize(mask: NDArray[np.bool_], transform: Affine) -> Shape:
    shape_records = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    polygons = [shapely.geometry.shape(geometry) for geometry, _value in shape_records]
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
