from tile2net.grid.cfg import Cfg

from tile2net.grid import InGrid

from tile2net.grid.cfg import cfg as _cfg
from tile2net.grid.util import ensure_tempdir_for_indir

# Must be within main to avoid parallelism issues
if __name__ == '__main__':
    cfg = Cfg.from_parser()

    with cfg:
        assert cfg.vector.length == _cfg.vector.length
        ingrid = InGrid.from_location(
            location=cfg.location,
            zoom=cfg.zoom
        )
        if cfg.indir:
            ingrid = ingrid.set_indir()
        else:
            ingrid = ingrid.set_source()
        if cfg.outdir:
            ingrid = ingrid.set_outdir()

        ingrid = ingrid.set_segmentation()
        ingrid = ingrid.set_vectorization()

        if cfg.line.concat:
            lines = ingrid.lines
        if cfg.polygon.concat:
            polygons = ingrid.polygons
