import sys
import os
import logging.config

import tqdm
from toolz import pipe
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
print(path)
print(__file__)
conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
print(f'{conf} exists: {os.path.exists(conf)}')

pipe(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'),
    logging.config.fileConfig
)
# todo: when release, set to USER
logger = logging.getLogger('debug')

# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         # import tqdm
#         super().__init__(level)
#         # self.tqdm = tqdm.tqdm(total=1, unit="log", leave=False)
#
#     def emit(self, record):
#         try:
#             # msg = self.format(record)
#             # self.tqdm.write(msg, file=sys.stderr)
#             # self.tqdm.update()
#             # tqdm.tqdm.write(msg, file=sys.stderr)
#             self.flush()
#         except (KeyboardInterrupt, SystemExit):
#             raise
#         except Exception:
#             self.handleError(record)

# logger.handlers
pass
# pipe(
#     sys.stderr,
#     logging.StreamHandler,
#     # logger.addHandler
#     logger.addHandler
# )
