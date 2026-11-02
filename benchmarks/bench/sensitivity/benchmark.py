import time
from functools import partial
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from bench.environment import OUTPUT_DIR
from bench.environment import configure_logging
from bench.environment import logger
from bench.methods import POLYGON_RESOLUTION_M
from bench.metrics import as_lines
from bench.metrics import dead_ends
from bench.metrics import loops
from bench.metrics import score
from bench.runner import timed
from bench.scene import Scene
from bench.sensitivity.figures import sensitivity_figure
from bench.sensitivity.variants import BASELINE
from bench.sensitivity.variants import Variant
from bench.sensitivity.variants import sensitivities
from bench.synthetic.cases import make_scene
from scipy.stats import wilcoxon
from shapely import Geometry
from shapely import LineString
from shapely.errors import GEOSException
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from crestline import crestline
from crestline.crestline import _rasterize

RUNS_PATH = OUTPUT_DIR / 'sensitivity_benchmark_runs.csv'
TABLE_PATH = OUTPUT_DIR / 'sensitivity_benchmark_results.csv'

DEFAULT_CASES = ('roads', 'mixed', 'river')
DEFAULT_SEEDS = tuple(range(10))
SIGNIFICANCE = 0.05
SCENE_KEYS = ['case', 'seed', 'source']
HIGHER_IS_BETTER = {
    'f5': True,
    'f2': True,
    'offset': False,
    'dead_end_error': False,
    'loop_error': False,
    'runtime_ms': False,
}


def sources(scene: Scene) -> dict[str, tuple[np.ndarray, Affine]]:
    mask, (left, top) = _rasterize(scene.polygon, POLYGON_RESOLUTION_M)
    transform = Affine(POLYGON_RESOLUTION_M, 0, left, 0, -POLYGON_RESOLUTION_M, top)
    return {'polygon': (mask, transform), 'noisy mask': (scene.mask, scene.transform)}


def trace(variant: Variant, mask: np.ndarray, transform: Affine) -> Geometry:
    return crestline(mask, transform=transform, **variant.options)


def topology(predicted: list[LineString]) -> tuple[int, int]:
    try:
        return dead_ends(predicted), loops(predicted)
    except GEOSException:
        noded = as_lines(shapely.union_all(predicted, grid_size=0.01))
        return dead_ends(noded), loops(noded)


def measure(scene: Scene, output: Geometry, seconds: float) -> dict[str, float]:
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


def run(cases: list[str], seeds: list[int], repeat: int) -> pd.DataFrame:
    variants = sensitivities()
    rows: list[dict[str, Any]] = []

    with tqdm(total=len(cases) * len(seeds) * 2 * len(variants), unit='run') as progress:
        for case_kind, seed in product(cases, seeds):
            scene = make_scene(case_kind, seed)

            for source, (mask, transform) in sources(scene).items():
                for name, variant in variants.items():
                    progress.set_description(f'{scene.label} {source} {name}')
                    output, seconds, _ = timed(partial(trace, variant, mask, transform), repeat)
                    rows.append(
                        {
                            'case': case_kind,
                            'seed': seed,
                            'source': source,
                            'variant': name,
                            'spur': variant.spur,
                            'tip': variant.tip,
                            'radius_k': variant.radius_k,
                            **measure(scene, output, seconds),
                        }
                    )
                    progress.update()

    return pd.DataFrame(rows)


def compare(variant_vals: np.ndarray, full: np.ndarray, higher_is_better: bool) -> dict[str, Any]:
    gain = (1 if higher_is_better else -1) * (variant_vals - full)
    tolerance = 1e-9 * max(1.0, np.abs(full).max())
    changed = np.abs(gain) > tolerance

    return {
        'full': full.mean(),
        'variant_mean': variant_vals.mean(),
        'delta': variant_vals.mean() - full.mean(),
        'worse': int((gain < -tolerance).sum()),
        'better': int((gain > tolerance).sum()),
        'same': int((~changed).sum()),
        'scenes': len(gain),
        'p_value': wilcoxon(gain[changed]).pvalue if changed.sum() >= 6 else 1.0,
        'direction': 'worse' if np.median(gain) < 0 or gain.mean() < 0 else 'better',
    }


def holm(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    ranked = p_values.to_numpy()[order] * (len(p_values) - np.arange(len(p_values)))
    adjusted = np.empty(len(p_values))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(ranked))
    return pd.Series(adjusted, index=p_values.index)


def paired(runs: pd.DataFrame) -> pd.DataFrame:
    baseline_runs = runs.loc[runs['variant'] == BASELINE, [*SCENE_KEYS, *HIGHER_IS_BETTER]]
    paired_runs = runs.merge(baseline_runs, on=SCENE_KEYS, suffixes=('', '_full'))
    rows: list[dict[str, Any]] = []

    for (source, variant), group in paired_runs.groupby(['source', 'variant'], sort=False):
        for metric, higher_is_better in HIGHER_IS_BETTER.items():
            variant_vals = group[metric].to_numpy(float)
            full = group[f'{metric}_full'].to_numpy(float)
            valid = ~(np.isnan(variant_vals) | np.isnan(full))
            if valid.any():
                comparison = compare(variant_vals[valid], full[valid], higher_is_better)
                rows.append({'source': source, 'variant': variant, 'metric': metric, **comparison})

    table = pd.DataFrame(rows)
    is_runtime = table['metric'] == 'runtime_ms'
    table['p_holm'] = table.groupby([table['source'], is_runtime], group_keys=False)['p_value'].transform(holm)
    table['verdict'] = np.select(
        [table['worse'] + table['better'] == 0, table['p_holm'] >= SIGNIFICANCE],
        ['identical', 'no significant change'],
        table['direction'],
    )
    return table.drop(columns='direction')


def main(
    cases: list[str] | None = None,
    seeds: list[int] | None = None,
    repeat: int = 3,
    log_level: str = 'INFO',
) -> None:
    """Evaluate parameter sensitivity (spur, tip, radius_k) of crestline on the synthetic benchmark.

    :param cases: Case kinds to run; defaults to 'roads', 'mixed' and 'river'
    :param seeds: Seeds per case kind; defaults to 0 through 9
    :param repeat: Timed runs per variant and input after one warm-up run; the fastest one is kept
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    configure_logging(log_level)
    cases = list(DEFAULT_CASES if cases is None else cases)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    logger.info(f'Evaluating sensitivity across {len(cases) * len(seeds)} scenes over {", ".join(cases)}')

    started = time.perf_counter()
    with logging_redirect_tqdm():
        runs = run(cases, seeds, repeat)

    logger.info(f'Sensitivity benchmark finished in {time.perf_counter() - started:.1f} s with {len(runs)} runs')
    runs.to_csv(RUNS_PATH, index=False, lineterminator='\n')

    table = paired(runs)
    table.round(4).to_csv(TABLE_PATH, index=False, lineterminator='\n')
    logger.info(f'Wrote {len(table)} comparisons to {TABLE_PATH}')

    sensitivity_figure(table)
