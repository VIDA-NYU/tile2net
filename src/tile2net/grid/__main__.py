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

        if cfg.indir.path:
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


        # ingrid.preview()


        if cfg.line.concat:
            # concatenate lines into single file and save
            lines = ingrid.lines
        if cfg.polygon.concat:
            # concatenate polygons into single file and save
            polygons = ingrid.polygons

        # save a preview of the polygons to file
        if cfg.polygon.preview:
            dest = ingrid.outdir.polygons.preview
            _ = ingrid.polygons
            msg = (
                f'Saving preview of polygons to '
                f'\n\t{dest}'
            )
            logger.info(msg)
            maxdim = cfg.polygon.preview
            img = ingrid.polygons.preview(maxdim=maxdim, show=False)
            img.save(dest)


        # save a preview of the lines to file
        if cfg.line.preview:
            dest = ingrid.outdir.lines.preview
            _ = ingrid.lines
            msg = (
                f'Saving preview of lines to '
                f'\n\t{dest}'
            )
            logger.info(msg)
            maxdim = cfg.line.preview
            img = ingrid.lines.preview(maxdim=maxdim, show=False)
            img.save(dest)

        if cfg.segment.to_pkl:
            ingrid.seggrid.to_pickle(ingrid.outdir.seggrid.pickle)

        # delete empty directories for easier browsing
        ingrid.summary()
