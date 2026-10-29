from collections.abc import Callable
from functools import partial
from typing import Any
from typing import NamedTuple

import numpy as np

from crestline.crestline import _center
from crestline.crestline import _crests
from crestline.crestline import _despur
from crestline.crestline import _flood
from crestline.crestline import _links
from crestline.crestline import _prune
from crestline.crestline import _thin
from crestline.crestline import _tips


class Variant(NamedTuple):
    stage: str
    options: dict[str, Any]
    patches: dict[str, Callable[..., Any]]


def constant_thinning_radius(radius: float) -> Callable[..., Any]:
    def thin(rows, cols, h, w, adaptive):
        return _thin(rows, cols, h, w, np.full_like(adaptive, radius))

    return thin


def constant_link_length(length: float) -> Callable[..., Any]:
    def links(points, heights, near):
        return _links(points, np.full(len(heights), length / 2), near)

    return links


def no_spur_depth_guard(points, pairs, heights, near, spur, tip):
    return _despur(points, pairs, heights, near, spur, np.inf)


def no_tip_extension(points, pairs, heights, crest, depth, tip):
    return _tips(points, pairs, heights, crest, depth, np.inf)


def no_subpixel_refinement(points, heights, crest, depth):
    own = np.zeros_like(crest)
    own[points[:, 0] + 2, points[:, 1] + 2] = True
    return _center(points, heights, own, depth)


def ablations(thinning_radius: float = 3.0, link_length: float = 10.0) -> dict[str, Variant]:
    return {
        'baseline': Variant('', {}, {}),
        '4-neighbour flood': Variant('flood', {}, {'_flood': partial(_flood, neighbourhood=4)}),
        '8-neighbour flood': Variant('flood', {}, {'_flood': partial(_flood, neighbourhood=8)}),
        '3×3 crest window': Variant('crest', {}, {'_crests': partial(_crests, reach=1)}),
        f'constant thinning radius ({thinning_radius:g} px)': Variant(
            'thin', {}, {'_thin': constant_thinning_radius(thinning_radius)}
        ),
        f'constant link length ({link_length:g} px)': Variant(
            'link', {}, {'_links': constant_link_length(link_length)}
        ),
        'no ridge test': Variant('link', {'dip': 0.0}, {}),
        'no shortcut pruning': Variant('prune', {}, {'_prune': partial(_prune, cycle=0)}),
        'triangles only': Variant('prune', {}, {'_prune': partial(_prune, cycle=3)}),
        'no spur pruning': Variant('despur', {'spur': 0.0}, {}),
        'no spur depth guard': Variant('despur', {}, {'_despur': no_spur_depth_guard}),
        'no occluder opening': Variant('occluders', {'hole': 0.0}, {}),
        'no tip extension': Variant('tips', {}, {'_tips': no_tip_extension}),
        'no subpixel refinement': Variant('center', {}, {'_center': no_subpixel_refinement}),
    }
