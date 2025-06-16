from contextlib import contextmanager
from time import perf_counter
from tile2net.logger import logger

@contextmanager
def benchmark(step: str):
    t0 = perf_counter()
    logger.debug(f"{step} – start")
    try:
        yield
    finally:
        dt = perf_counter() - t0
        logger.debug(f"{step} – {dt:,.1f}s")
