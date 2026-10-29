import time
from typing import TypeAlias

import pandas as pd
from bench.environment import OUTPUT_DIR
from bench.environment import configure_logging
from bench.environment import detect_cuda
from bench.environment import logger
from bench.methods import MethodRegistry
from bench.methods import methods
from bench.runner import Result
from bench.runner import benchmark_scene
from bench.runner import write_table
from bench.synthetic.cases import make_scene
from bench.synthetic.figures import benchmark_figure
from bench.synthetic.figures import cuda_benchmark_figure
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

Plan: TypeAlias = list[tuple[str, int]]

TABLE_PATH = OUTPUT_DIR / 'synthetic_benchmark_results.csv'
DEFAULT_CASES = ('roads', 'mixed', 'river', 'roads-large')
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_LARGE_SEEDS = (1, 2)


def _plan(cases: list[str], seeds: list[int], large_seeds: list[int]) -> Plan:
    return [
        (case_kind, seed) for case_kind in cases for seed in (large_seeds if case_kind.startswith('roads-') else seeds)
    ]


def run(plan: Plan, method_registry: MethodRegistry, repeat: int) -> list[Result]:
    results: list[Result] = []

    with tqdm(total=2 * len(method_registry) * len(plan), unit='run') as progress:
        for case_kind, seed in plan:
            scene = make_scene(case_kind, seed)
            logger.info(
                f'Case {case_kind} seed {seed}: {len(scene.centerlines)} centerlines, '
                f'{scene.mask.size / 1e6:.2f} Mpx mask, {scene.true_dead_ends} true dead ends, {scene.holes} holes'
            )
            results.extend(benchmark_scene(scene, method_registry, repeat, progress))

    return results


def main(
    cases: list[str] | None = None,
    seeds: list[int] | None = None,
    large_seeds: list[int] | None = None,
    repeat: int = 3,
    log_level: str = 'INFO',
) -> None:
    """Run the synthetic centerline benchmark and write the summary table and figures to ``output/``.

    :param cases: Case kinds to run; defaults to: 'roads', 'mixed', 'river' and 'roads-large'
    :param seeds: Seeds for the 1 km cases; defaults to 0 through 5
    :param large_seeds: Seeds for the 4 km road networks; defaults to 1 and 2
    :param repeat: Timed runs per method and input after one warm-up run; the fastest one is kept
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    configure_logging(log_level)
    cases = list(DEFAULT_CASES if cases is None else cases)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    large_seeds = list(DEFAULT_LARGE_SEEDS if large_seeds is None else large_seeds)

    gpu_name = detect_cuda()
    method_registry = methods(bool(gpu_name))
    logger.info(f'Methods: {", ".join(method_registry)}')

    plan = _plan(cases, seeds, large_seeds)
    logger.info(f'Plan: {len(plan)} cases over {", ".join(cases)}, {repeat} timed repeats per run')

    started = time.perf_counter()
    with logging_redirect_tqdm():
        results = run(plan, method_registry, repeat)

    logger.info(f'Benchmark finished in {time.perf_counter() - started:.1f} s with {len(results)} results')
    write_table(results, ['case', 'source', 'method'], TABLE_PATH)

    runs = pd.DataFrame(results)
    if 'roads' in cases:
        benchmark_figure(runs)

    if gpu_name:
        cuda_benchmark_figure(runs, gpu_name)
