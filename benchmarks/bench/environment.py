import ctypes
import gc
import logging
from contextlib import suppress
from pathlib import Path

from numba import cuda

BENCHMARKS_DIR = Path(__file__).parent.parent
DATA_DIR = BENCHMARKS_DIR / 'data'
OUTPUT_DIR = BENCHMARKS_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger('benchmarks')


def configure_logging(log_level: str) -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(log_level.upper())


def release_memory() -> None:
    gc.collect()
    with suppress(OSError, AttributeError):
        ctypes.CDLL('libc.so.6').malloc_trim(0)


def cuda_device() -> tuple[str, str]:
    try:
        if not cuda.is_available():
            return '', 'no device found'

        device_name: str | bytes = cuda.get_current_device().name
    except Exception as error:
        return '', f'{type(error).__name__}: {error}'

    return (device_name.decode() if isinstance(device_name, bytes) else device_name), ''


def detect_cuda() -> str:
    gpu_name, failure_reason = cuda_device()
    if gpu_name:
        logger.info(f'CUDA: {gpu_name}')
    else:
        logger.info(f'CUDA: off ({failure_reason}), running on the CPU only')

    return gpu_name
