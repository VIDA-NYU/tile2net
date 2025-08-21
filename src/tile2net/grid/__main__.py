from tile2net.grid import InGrid
from tile2net.grid.cfg.logger import logger
from tile2net.grid.cfg import Cfg
from tile2net.grid.cfg import cfg as _cfg

# Must be within main to avoid parallelism issues
if __name__ == '__main__':
    cfg = Cfg.from_parser()

    with cfg:

        # instantiate InGrid using a location
        assert cfg.vector.length == _cfg.vector.length
        ingrid = InGrid.from_location(
            location=cfg.location,
            zoom=cfg.zoom
        )

        if cfg.indir:
            # use input imagery
            ingrid = ingrid.set_indir()
        else:
            # set a source if specified or infer from location
            ingrid = ingrid.set_source()

        if cfg.outdir:
            ingrid = ingrid.set_outdir()

        # configure segmentation using cfg parameters
        ingrid = ingrid.set_segmentation()
        # configure vectorization using cfg parameters
        ingrid = ingrid.set_vectorization()

        if cfg.line.concat:
            # concatenate lines into single file and save
            lines = ingrid.lines
        if cfg.polygon.concat:
            # concatenate polygons into single file and save
            polygons = ingrid.polygons

        # save a preview of the lines to file
        if cfg.line.preview:
            msg = (
                f'Saving preview of lines to '
                f'\n\t{ingrid.tempdir.lines.preview}'
            )
            logger.info(msg)
            img = ingrid.lines.plot(
                maxdim=cfg.line.preview,
                show=False
            )
            img.save(ingrid.tempdir.lines.preview)

        # save a preview of the polygons to file
        if cfg.polygon.preview:
            msg = (
                f'Saving preview of polygons to '
                f'\n\t{ingrid.tempdir.polygons.preview}'
            )
            logger.info(msg)
            img = ingrid.polygons.plot(
                maxdim=cfg.polygon.preview,
                show=False
            )
            img.save(ingrid.tempdir.polygons.preview)

        if cfg.segment.to_pkl:
            ingrid.seggrid.to_pickle(ingrid.tempdir.seggrid.pickle)

        # delete empty directories for easier browsing
        ingrid.summary()
