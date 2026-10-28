import time
from collections.abc import Set
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
import pandas as pd
from bench.environment import logger
from bench.environment import release_memory
from bench.methods import CUDA_METHOD
from bench.methods import Extractor
from bench.methods import MethodRegistry
from bench.metrics import as_lines
from bench.metrics import dead_ends
from bench.metrics import loops
from bench.metrics import score
from bench.scene import Scene
from shapely import Geometry
from tqdm import tqdm

Result: TypeAlias = dict[str, Any]


class ResultColumn(NamedTuple):
    name: str
    key: str
    scale: int
    decimals: int


class Timing(NamedTuple):
    output: Geometry
    best_time: float
    warm_up_time: float


RESULT_COLUMNS: list[ResultColumn] = [
    ResultColumn('mpx', 'mpx', 1, 2),
    ResultColumn('f1_5m', 'f5', 1, 4),
    ResultColumn('f1_2m', 'f2', 1, 4),
    ResultColumn('offset_m', 'offset', 1, 3),
    ResultColumn('dead_ends', 'dead', 1, 1),
    ResultColumn('true_dead_ends', 'true_dead', 1, 1),
    ResultColumn('loops', 'loops', 1, 1),
    ResultColumn('holes', 'holes', 1, 1),
    ResultColumn('runtime_ms', 'time', 1000, 2),
]


def _timed_run(
    extract: Extractor,
    scene: Scene,
    repeat: int,
    always_single_run: bool,
    single_run_after: float,
) -> Timing:
    started = time.perf_counter()
    extract(scene.polygon, scene.mask, scene.transform)
    warm_up_time = time.perf_counter() - started

    is_single_run = always_single_run or warm_up_time > single_run_after
    best_time = np.inf

    for _ in range(1 if is_single_run else repeat):
        started = time.perf_counter()
        output = extract(scene.polygon, scene.mask, scene.transform)
        best_time = min(best_time, time.perf_counter() - started)

    return Timing(output, best_time, warm_up_time)


def _result(
    scene: Scene,
    method_name: str,
    source: str,
    output: Geometry,
    best_time: float,
) -> Result:
    predicted_lines = as_lines(output)
    result: Result = {
        **scene.fields,
        'source': source,
        'method': method_name,
        'time': best_time,
        'mpx': scene.mask.size / 1e6,
    }
    result.update(
        score(predicted_lines, scene.centerlines),
        dead=dead_ends(predicted_lines),
        true_dead=scene.true_dead_ends,
        loops=loops(predicted_lines),
        holes=scene.holes,
    )
    return result


def benchmark_scene(
    scene: Scene,
    method_registry: MethodRegistry,
    repeat: int,
    progress: tqdm,
    retired: Set[tuple[str, str]] = frozenset(),
    single_run_after: float = np.inf,
) -> list[Result]:
    results: list[Result] = []
    cpu_outputs: dict[str, Geometry] = {}

    for method_name, method in method_registry.items():
        for source, extract in (('polygon', method.on_polygon), ('noisy mask', method.on_mask)):
            if (method_name, source) in retired:
                progress.update()
                continue

            progress.set_description(f'{scene.label} {method_name} ({source})')
            output, best_time, warm_up_time = _timed_run(
                extract, scene, repeat, method_name.startswith('centerline'), single_run_after
            )

            if method_name == 'crestline':
                cpu_outputs[source] = output

            result = _result(scene, method_name, source, output, best_time)
            if method_name == CUDA_METHOD and source in cpu_outputs:
                result['same'] = bool(output.equals_exact(cpu_outputs[source], 0))
                if not result['same']:
                    logger.warning(f'CUDA and CPU crestline disagree on {scene.label} ({source})')

            logger.debug(
                f'{method_name} on {scene.label} ({source}): F1@5m {result["f5"]:.3f}, '
                f'{result["dead"]}/{scene.true_dead_ends} dead ends, {result["loops"]} loops, '
                f'{1000 * best_time:.1f} ms after a {1000 * warm_up_time:.1f} ms warm-up'
            )

            results.append(result)
            progress.set_postfix(
                f5=f'{result["f5"]:.3f}',
                dead=f'{result["dead"]}/{scene.true_dead_ends}',
                time=f'{best_time:.3f}s',
            )
            progress.update()

            del output
            release_memory()

    return results


def write_table(
    results: list[Result],
    group_columns: list[str],
    table_path: Path,
) -> pd.DataFrame:
    summary = (
        pd.DataFrame(results)
        .groupby(group_columns, sort=False)
        .agg(**{column.name: (column.key, 'mean') for column in RESULT_COLUMNS})
        .reset_index()
    )

    for column in RESULT_COLUMNS:
        if column.scale != 1:
            summary[column.name] *= column.scale

    summary = summary.round({column.name: column.decimals for column in RESULT_COLUMNS})
    summary.to_csv(table_path, index=False, lineterminator='\n')

    logger.info(f'Wrote {len(summary)} summary rows to {table_path}')

    return summary
