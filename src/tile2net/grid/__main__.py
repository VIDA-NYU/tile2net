from tile2net.grid.cfg import Cfg

from tile2net.grid import InGrid

from tile2net.grid.cfg import cfg as _cfg

# Must be within main to avoid parallelism issues
if __name__ == '__main__':
    cfg = Cfg.from_parser()

    with cfg:
        assert cfg.vector.length == _cfg.vector.length
        ingrid = InGrid.from_location(
            location=cfg.location,
            zoom=cfg.zoom
        )
        if cfg.input_dir:
            ingrid = ingrid.set_indir()
        else:
            ingrid = ingrid.set_source()

        ingrid = ingrid.set_segmentation()
        ingrid = ingrid.set_vectorization()

        if cfg.line.concat:
            lines = ingrid.lines
        if cfg.polygon.convexity:
            polygons = ingrid.polygons
        ingrid.summary()
