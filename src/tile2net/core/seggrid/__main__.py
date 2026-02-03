from tile2net.core import Grid
from tile2net.core.cfg import Cfg

cfg = Cfg.from_json()
with cfg:
    grid = Grid.from_cfg(cfg)
    grayscale = grid.seggrid.file.pred
