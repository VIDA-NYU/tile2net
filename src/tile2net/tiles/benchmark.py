from __future__ import annotations
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Literal


from contextlib import contextmanager
from tile2net.tiles.cfg.logger import logger


# Pre-define the textual levels we’ll accept
_LEVELS: dict[Literal[
        "debug",
        "info",
        "warning",
        "error",
        "critical",
    ], int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

@contextmanager
def benchmark(
        step: str,
        level: int | str = logging.DEBUG,
):

    #Convert any textual level to its numeric counterpart
    if isinstance(level, str):
        level = _LEVELS[level.lower()]

    t0 = perf_counter()

    #Log the start message
    logger.log(level, f"{step} – start")

    try:
        yield
    finally:
        dt = perf_counter() - t0

        #Log the elapsed time with the same level
        logger.log(level, f"{step} – {dt:,.1f}s")
