from __future__ import annotations

import logging
import logging.config
from pathlib import Path

logging.config.fileConfig(Path(__file__).with_name("logging.conf"))
logger = logging.getLogger("user")
