from tile2net.grid.cfg import Cfg

from tile2net.grid import InGrid

from tile2net.grid.cfg import cfg as _cfg

cfg = Cfg.from_parser()

_cfg._context

with cfg:
    assert cfg.vector.length == _cfg.vector.length
    ingrid = InGrid.from_location(
        location=cfg.location,
        zoom=cfg.zoom
    )
    assert ingrid.cfg._context is not None
    assert cfg.model.bs_val == ingrid.cfg.model.bs_val == 16
    cfg.vector.length
    ingrid.cfg.vector.__dict__
    ingrid.cfg._context
    if cfg.input_dir:
        ingrid = ingrid.set_indir()
    else:
        ingrid = ingrid.set_source()

    ingrid = ingrid.set_segmentation()
    ingrid = ingrid.set_vectorization()
    ingrid.cfg
    cfg

    lines = ingrid.lines


