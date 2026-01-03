from tile2net.grid import InGrid
from tile2net.grid.cfg import Cfg

cfg = Cfg.from_json()
with cfg:
    ingrid = InGrid.from_cfg(cfg)
    grayscale = ingrid.seggrid.file.pred
