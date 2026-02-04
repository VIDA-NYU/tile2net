from tile2net.geo import InGrid
from tile2net.core.cfg import Cfg

cfg = Cfg.from_json()
with cfg:
    grid = InGrid.from_cfg(cfg)
    grayscale = grid.seggrid.file.pred
