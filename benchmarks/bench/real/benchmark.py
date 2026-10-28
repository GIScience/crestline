import time

import pandas as pd
from bench.environment import OUTPUT_DIR
from bench.environment import configure_logging
from bench.environment import detect_cuda
from bench.environment import logger
from bench.environment import release_memory
from bench.methods import MethodRegistry
from bench.methods import methods
from bench.real.figure import benchmark_figure
from bench.real.network import Coordinate
from bench.real.network import Network
from bench.real.network import Noise
from bench.real.network import load_network
from bench.real.network import make_scene
from bench.real.network import network_center
from bench.runner import Result
from bench.runner import benchmark_scene
from bench.runner import write_table
from bench.scene import Scene
from bench.scene import Window
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

TABLE_PATH = OUTPUT_DIR / 'real_benchmark_results.csv'
RADII_KM: list[int] = list(range(1, 6))
SINGLE_RUN_AFTER_S = 10.0


def _prepare_scene(network: Network, center: Coordinate, radius_km: int, seed: int, noise: Noise) -> Scene:
    started = time.perf_counter()
    scene = make_scene(network, center, radius_km, seed, noise)
    logger.info(
        f'Buffer {radius_km} km: {len(scene.centerlines):,} centerlines over '
        f'{sum(line.length for line in scene.centerlines) / 1000:,.0f} km, {scene.mask.size / 1e6:.1f} Mpx '
        f'mask, {scene.true_dead_ends:,} true dead ends, {scene.holes:,} holes, '
        f'prepared in {time.perf_counter() - started:.1f} s'
    )
    return scene


def _slow_runs(results: list[Result], radius_km: int, time_limit: float) -> set[tuple[str, str]]:
    slow: set[tuple[str, str]] = set()
    for result in results:
        if result['time'] > time_limit:
            slow.add((result['method'], result['source']))
            logger.warning(
                f'{result["method"]} ({result["source"]}) took {result["time"]:,.0f} s on the {radius_km} km '
                f'buffer, over the {time_limit:,.0f} s limit, so it is skipped on larger buffers'
            )

    return slow


def run(
    network: Network,
    center: Coordinate,
    radii: list[int],
    method_registry: MethodRegistry,
    repeat: int,
    seed: int,
    noise: Noise,
    time_limit: float,
) -> pd.DataFrame:
    results: list[Result] = []
    retired: set[tuple[str, str]] = set()
    summary = pd.DataFrame()

    with tqdm(total=2 * len(method_registry) * len(radii), unit='run') as progress:
        for radius_km in radii:
            scene = _prepare_scene(network, center, radius_km, seed, noise)
            scene_results = benchmark_scene(scene, method_registry, repeat, progress, retired, SINGLE_RUN_AFTER_S)
            retired |= _slow_runs(scene_results, radius_km, time_limit)

            results.extend(scene_results)
            summary = write_table(results, ['radius_km', 'source', 'method'], TABLE_PATH)
            del scene
            release_memory()

    return summary


def _square_window(center: Coordinate, offset: Coordinate, size: float) -> Window:
    center_x, center_y = center[0] + offset[0], center[1] + offset[1]
    return center_x - size / 2, center_y - size / 2, center_x + size / 2, center_y + size / 2


def main(
    repeat: int = 3,
    seed: int = 0,
    coarse_noise: float = 0.15,
    fine_noise: float = 0.05,
    time_limit: float = 600.0,
    panel_radius: int = 1,
    window_size: float = 400.0,
    window_offset: tuple[float, float] = (600.0, 350.0),
    log_level: str = 'INFO',
) -> None:
    """Run the Łódź road network benchmark over growing buffers and write the summary table and figure to ``output/``.

    :param repeat: Timed runs per method and input after one warm-up run; the fastest one is kept
    :param seed: Seed of the boundary noise added to the noisy masks
    :param coarse_noise: Standard deviation of the smooth boundary noise as a fraction of the local road half-width
    :param fine_noise: Standard deviation of the fine boundary noise as a fraction of the local road half-width
    :param time_limit: Seconds; a method and input whose best run takes longer is skipped on larger buffers
    :param panel_radius: Buffer radius in km of the scene drawn in the figure's map panels
    :param window_size: Side in metres of the map panels' window
    :param window_offset: Offset in metres of the map panels' window centre from ``center``
    :param log_level: Level for the benchmark's own log messages, such as DEBUG, INFO or WARNING
    """
    configure_logging(log_level)
    noise = (coarse_noise, fine_noise)

    network = load_network()
    center = network_center(network)
    window = _square_window(center, window_offset, window_size)

    method_registry = methods(bool(detect_cuda()))
    logger.info(f'Methods: {", ".join(method_registry)}')
    logger.info(
        f'Plan: buffers of {", ".join(str(radius) for radius in RADII_KM)} km around '
        f'({center[0]:,.0f}, {center[1]:,.0f}), {repeat} timed repeats per run after a warm-up, '
        f'{time_limit:,.0f} s limit per run'
    )

    started = time.perf_counter()
    with logging_redirect_tqdm():
        summary = run(network, center, RADII_KM, method_registry, repeat, seed, noise, time_limit)

    logger.info(f'Benchmark finished in {time.perf_counter() - started:,.1f} s with {len(summary)} summary rows')
    benchmark_figure(summary, network, center, panel_radius, window, seed, noise, time_limit)
