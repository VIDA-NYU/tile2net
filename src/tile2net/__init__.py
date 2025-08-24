import warnings


warnings.filterwarnings('ignore', '.*initial implementation of Parquet.*', FutureWarning)
warnings.filterwarnings("ignore", '.*Shapely GEOS version.*', UserWarning)
warnings.filterwarnings('ignore', '.*Shapely 2.0.*', UserWarning)

from tile2net.logger import logger