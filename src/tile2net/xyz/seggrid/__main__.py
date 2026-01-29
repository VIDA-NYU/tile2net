from tile2net.xyz import Grid
from tile2net.xyz.cfg import Cfg

cfg = Cfg.from_json()
with cfg:
    grid = Grid.from_cfg(cfg)
    grayscale = grid.seggrid.file.pred
