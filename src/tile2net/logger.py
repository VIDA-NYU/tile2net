import logging.config
import os

fname = os.path.join(
    os.path.dirname(__file__),
    'logging.conf',
)
logging.config.fileConfig(fname)
logger = logging.getLogger('debug')
