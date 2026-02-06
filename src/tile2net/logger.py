import logging.config
import os


fname = os.path.join(
    os.path.dirname(__file__),
    'logging.conf',
)
logging.config.fileConfig(fname, disable_existing_loggers=False)
logger = logging.getLogger('debug')
