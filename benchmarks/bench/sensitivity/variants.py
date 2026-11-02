from itertools import product
from typing import Any
from typing import NamedTuple


class Variant(NamedTuple):
    name: str
    options: dict[str, Any]
    spur: float
    tip: float
    radius_k: float


def sensitivities() -> dict[str, Variant]:
    spur_values = (1, 1.5, 2, 2.5, 3)
    tip_values = (0.05, 0.25, 0.5, 0.75, 0.95)
    radius_k_values = (0.05, 0.15, 0.35, 0.55, 0.65)

    variants = {}
    for spur, tip, radius_k in product(spur_values, tip_values, radius_k_values):
        name = f's={spur:g}_t={tip:g}_rk={radius_k:g}'
        variants[name] = Variant(
            name=name,
            options={'spur': spur, 'tip': tip, 'radius_k': radius_k},
            spur=spur,
            tip=tip,
            radius_k=radius_k,
        )
    return variants
