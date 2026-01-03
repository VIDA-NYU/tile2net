import sys

from tile2net.grid import InGrid
from tile2net.grid.cfg.logger import logger
from tile2net.grid.cfg import Cfg
from tile2net.grid.cfg import cfg as _cfg
import subprocess


# Must be within main to avoid parallelism issues
if __name__ == '__main__':

    ingrid = InGrid.from_cfg()
    cfg = ingrid.cfg

    cfg.to_json()

    _ = ingrid.seggrid.file.pred


    with cfg:

        if not cfg.inference:
            msg = 'Skipping inference as per configuration.'
            logger.info(msg)
            exit(0)

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

        if cfg.segmentation.to_pkl:
            ingrid.seggrid.to_pickle(ingrid.outdir.seggrid.pickle)

        # delete empty directories for easier browsing
        ingrid.summary()
