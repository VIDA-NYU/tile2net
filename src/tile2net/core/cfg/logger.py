import logging.config
import os

from .cfg import cfg

fname = os.path.join(
    os.path.dirname(__file__),
    'logging.conf',
)
logging.config.fileConfig(fname, disable_existing_loggers=False)
logger = logging.getLogger('debug')
logger.setLevel(getattr(logging, cfg.log_level))
for h in logger.handlers:
    h.setLevel(getattr(logging, cfg.log_level))
