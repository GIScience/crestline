from dataclasses import dataclass
from typing import Any
from typing import TypeAlias

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from shapely import LineString
from shapely import MultiPolygon
from shapely import Polygon

Shape: TypeAlias = Polygon | MultiPolygon
Window: TypeAlias = tuple[float, float, float, float]


@dataclass
class Scene:
    fields: dict[str, Any]
    label: str
    polygon: Shape
    mask: NDArray[np.bool_]
    transform: Affine
    centerlines: list[LineString]
    true_dead_ends: int
    holes: int
