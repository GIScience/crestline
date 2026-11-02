from itertools import product
from typing import Any
from typing import NamedTuple


class Variant(NamedTuple):
    name: str
    options: dict[str, Any]
    spur: float
    tip: float
    radius_k: float


def variant_name(spur: float, tip: float, radius_k: float) -> str:
    return f's={spur:g}_t={tip:g}_rk={radius_k:g}'


BASELINE = variant_name(spur=2.0, tip=0.75, radius_k=0.35)


def sensitivities() -> dict[str, Variant]:
    spur_values = (1, 1.5, 2, 2.5, 3)
    tip_values = (0.05, 0.25, 0.5, 0.75, 0.95)
    radius_k_values = (0.05, 0.15, 0.35, 0.55, 0.65)

    variants = {}
    for spur, tip, radius_k in product(spur_values, tip_values, radius_k_values):
        name = variant_name(spur, tip, radius_k)
        variants[name] = Variant(
            name=name,
            options={'spur': spur, 'tip': tip, 'radius_k': radius_k},
            spur=spur,
            tip=tip,
            radius_k=radius_k,
        )
    return variants
