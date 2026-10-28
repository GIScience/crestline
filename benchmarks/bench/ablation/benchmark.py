import time
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from bench.ablation.figures import ablation_figure
from bench.ablation.variants import Variant
from bench.ablation.variants import ablations
from bench.environment import OUTPUT_DIR
from bench.environment import configure_logging
from bench.environment import logger
from bench.metrics import as_lines
from bench.metrics import dead_ends
from bench.metrics import loops
from bench.metrics import score
from bench.scene import Scene
from bench.synthetic.cases import make_scene
from scipy.stats import wilcoxon
from shapely.errors import GEOSException
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from crestline import crestline
from crestline.crestline import _rasterize

RUNS_PATH = OUTPUT_DIR / 'ablation_benchmark_runs.csv'
TABLE_PATH = OUTPUT_DIR / 'ablation_benchmark_results.csv'

DEFAULT_CASES = ('roads', 'mixed', 'river')
DEFAULT_SEEDS = tuple(range(10))
POLYGON_RESOLUTION_M = 0.5
SIGNIFICANCE = 0.05

HIGHER_IS_BETTER = {'f5': True, 'f2': True, 'offset': False, 'dead_end_error': False, 'loop_error': False}
METRICS = [*HIGHER_IS_BETTER, 'runtime_ms']


def sources(scene: Scene) -> dict[str, tuple[np.ndarray, Affine]]:
    mask, (left, top) = _rasterize(scene.polygon, POLYGON_RESOLUTION_M)
    transform = Affine(POLYGON_RESOLUTION_M, 0, left, 0, -POLYGON_RESOLUTION_M, top)
    return {'polygon': (mask, transform), 'noisy mask': (scene.mask, scene.transform)}


def trace(variant: Variant, mask: np.ndarray, transform: Affine):
    patches = patch.multiple('crestline.crestline', **variant.patches) if variant.patches else nullcontext()
    with patches:
        return crestline(mask, transform=transform, **variant.options)


def timed_trace(variant: Variant, mask: np.ndarray, transform: Affine, repeat: int):
    output, best = trace(variant, mask, transform), np.inf
    for _ in range(repeat):
        started = time.perf_counter()
        output = trace(variant, mask, transform)
        best = min(best, time.perf_counter() - started)

    return output, best


def topology(predicted: list) -> tuple[int, int]:
    try:
        return dead_ends(predicted), loops(predicted)
    except GEOSException:
        noded = as_lines(shapely.union_all(predicted, grid_size=0.01))
        return dead_ends(noded), loops(noded)


def measure(scene: Scene, output, seconds: float) -> dict[str, float]:
    predicted = as_lines(output)
    true_length = sum(line.length for line in scene.centerlines)
    dead, found_loops = topology(predicted)
    return {
        **score(predicted, scene.centerlines),
        'dead_end_error': abs(dead - scene.true_dead_ends),
        'loop_error': abs(found_loops - scene.holes),
        'length_error': abs(sum(line.length for line in predicted) / true_length - 1),
        'runtime_ms': 1000 * seconds,
    }


def run(cases: list[str], seeds: list[int], repeat: int, thinning_radius: float, link_length: float) -> pd.DataFrame:
    variants = ablations(thinning_radius, link_length)
    rows: list[dict[str, Any]] = []

    with tqdm(total=len(cases) * len(seeds) * 2 * len(variants), unit='run') as progress:
        for case_kind in cases:
            for seed in seeds:
                scene = make_scene(case_kind, seed)

                for source, (mask, transform) in sources(scene).items():
                    for name, variant in variants.items():
                        progress.set_description(f'{scene.label} {source} {name}')
                        output, seconds = timed_trace(variant, mask, transform, repeat)
                        rows.append(
                            {
                                'case': case_kind,
                                'seed': seed,
                                'source': source,
                                'variant': name,
                                'stage': variant.stage,
                                **measure(scene, output, seconds),
                            }
                        )
                        progress.update()

    return pd.DataFrame(rows)


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ['case', 'seed', 'source']
    full = runs[runs.variant == 'baseline'].set_index(keys)
    rows: list[dict[str, Any]] = []

    for (source, variant), group in runs.groupby(['source', 'variant'], sort=False):
        reference = full.loc[group.set_index(keys).index]
        for metric in METRICS:
            ablated, baseline = group[metric].to_numpy(float), reference[metric].to_numpy(float)
            valid = ~(np.isnan(ablated) | np.isnan(baseline))
            if not valid.any():
                continue

            sign = 1 if HIGHER_IS_BETTER.get(metric, False) else -1
            gain = sign * (ablated[valid] - baseline[valid])
            tolerance = 1e-9 * max(1.0, np.abs(baseline[valid]).max())
            changed = np.abs(gain) > tolerance

            rows.append(
                {
                    'source': source,
                    'variant': variant,
                    'metric': metric,
                    'full': baseline[valid].mean(),
                    'ablated': ablated[valid].mean(),
                    'delta': ablated[valid].mean() - baseline[valid].mean(),
                    'worse': int((gain < -tolerance).sum()),
                    'better': int((gain > tolerance).sum()),
                    'same': int((~changed).sum()),
                    'scenes': int(valid.sum()),
                    'p_value': wilcoxon(gain[changed]).pvalue if changed.sum() >= 6 else 1.0,
                    'direction': 'worse' if np.median(gain) < 0 or gain.mean() < 0 else 'better',
                }
            )

    table = pd.DataFrame(rows)
    family = [table['source'], table['metric'] == 'runtime_ms']
    table['p_holm'] = table.groupby(family, group_keys=False)['p_value'].transform(holm)
    table['verdict'] = np.select(
        [table['worse'] + table['better'] == 0, table['p_holm'] >= SIGNIFICANCE],
        ['identical', 'no significant change'],
        table['direction'],
    )
    return table.drop(columns='direction')


def holm(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    ranked = p_values.to_numpy()[order] * (len(p_values) - np.arange(len(p_values)))
    adjusted = np.empty(len(p_values))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ranked))
    return pd.Series(adjusted, index=p_values.index)


def main(
    cases: list[str] | None = None,
    seeds: list[int] | None = None,
    repeat: int = 3,
    thinning_radius: float = 3.0,
    link_length: float = 10.0,
    log_level: str = 'INFO',
) -> None:
    """Ablate every stage of crestline on the synthetic benchmark.

    :param cases: Case kinds to run; defaults to 'roads', 'mixed' and 'river'
    :param seeds: Seeds per case kind; defaults to 0 through 9
    :param repeat: Timed runs per ablation and input after one warm-up run; the fastest one is kept
    :param thinning_radius: Radius in pixels for the constant thinning radius ablation
    :param link_length: Length in pixels for the constant link length ablation
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """

    configure_logging(log_level)

    cases, seeds = list(cases or DEFAULT_CASES), list(DEFAULT_SEEDS if seeds is None else seeds)
    logger.info(f'Ablating {len(cases) * len(seeds)} scenes over {", ".join(cases)}')

    started = time.perf_counter()
    with logging_redirect_tqdm():
        runs = run(cases, seeds, repeat, thinning_radius, link_length)

    logger.info(f'Ablation finished in {time.perf_counter() - started:.1f} s with {len(runs)} runs')
    runs.to_csv(RUNS_PATH, index=False, lineterminator='\n')

    table = paired(runs)
    table.round(4).to_csv(TABLE_PATH, index=False, lineterminator='\n')
    logger.info(f'Wrote {len(table)} comparisons to {TABLE_PATH}')

    ablation_figure(table)
