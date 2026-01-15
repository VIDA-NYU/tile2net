import os.path
from dataclasses import dataclass
from typing import *

from tile2net.grid import Grid
from tile2net.grid.cfg.logger import logger


# Must be within main to avoid parallelism issues
# if __name__ == '__main__':
#     raise NotImplementedError
#
#     grid = Grid.from_cfg()
#     cfg = grid.cfg
#
#     cfg.to_json()
#
#     _ = grid.seggrid.file.pred
#
#     with cfg:
#
#         if not cfg.inference:
#             msg = 'Skipping inference as per configuration.'
#             logger.info(msg)
#             exit(0)
#
#         if cfg.line.concat:
#             # concatenate network into single file and save
#             network = grid.network
#         if cfg.polygon.concat:
#             # concatenate polygons into single file and save
#             polygons = grid.polygons
#
#         # save a preview of the polygons to file
#         if cfg.polygon.preview:
#             dest = grid.outdir.polygons.preview
#             _ = grid.polygons
#             msg = (
#                 f'Saving preview of polygons to '
#                 f'\n\t{dest}'
#             )
#             logger.info(msg)
#             maxdim = cfg.polygon.preview
#             img = grid.polygons.preview(maxdim=maxdim, show=False)
#             img.save(dest)
#
#         # save a preview of the network to file
#         if cfg.line.preview:
#             dest = grid.outdir.network.preview
#             _ = grid.network
#             msg = (
#                 f'Saving preview of network to '
#                 f'\n\t{dest}'
#             )
#             logger.info(msg)
#             maxdim = cfg.line.preview
#             img = grid.network.preview(maxdim=maxdim, show=False)
#             img.save(dest)
#
#         if cfg.segmentation.to_pkl:
#             grid.seggrid.to_pickle(grid.outdir.seggrid.pickle)
#
#         # delete empty directories for easier browsing
#         grid.summary()
#

@dataclass
class Process:
    grid: Grid

    @property
    def cfg(self):
        return self.grid.cfg

    def __enter__(self):
        self.before = self.existing

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deletes intermediate files that were not there to start with."""
        to_delete = self.existing - self.before
        for path in to_delete:
            try:
                os.remove(path)
            except Exception as e:
                msg = f'Could not remove file {path}: {e}'
                logger.warning(msg)

    @property
    def arg2dir(self) -> dict[str, str]:
        with self.grid.cfg, self:
            grid = self.grid
            seggrid = grid.seggrid
            vecgrid = grid.vecgrid
            cfg = self.cfg
            outdir = grid.outdir

            mapping: dict[str, str] = {}
            if cfg.static:
                _ = grid.file.static
                mapping['static'] = outdir.grid.static.dir
            if cfg.prob:
                _ = grid.file.prob
                mapping['prob'] = outdir.grid.prob.dir
            if cfg.pred:
                _ = grid.file.pred
                mapping['pred'] = outdir.grid.pred.dir
            if cfg.colorized:
                _ = grid.file.colorized
                mapping['colorized'] = outdir.grid.colorized.dir
            if cfg.intensity:
                _ = grid.file.intensity
                mapping['intensity'] = outdir.grid.intensity.dir
            if cfg.sidebyside:
                _ = grid.file.sidebyside
                mapping['sidebyside'] = outdir.grid.sidebyside.dir
            if cfg.overlay:
                _ = grid.file.overlay
                mapping['overlay'] = outdir.grid.overlay.dir
            if cfg.error:
                _ = grid.file.error
                mapping['error'] = outdir.grid.error.dir
            if cfg.soft:
                _ = grid.file.soft
                mapping['soft'] = outdir.grid.soft.dir
            if cfg.network:
                _ = grid.file.network
                mapping['network'] = outdir.vecgrid.network.dir
            if cfg.polygons:
                _ = grid.file.polygons
                mapping['polygons'] = outdir.vecgrid.polygons.dir
            if cfg.line.preview:
                mapping['line.preview'] = outdir.network.preview
            if cfg.polygon.preview:
                mapping['polygon.preview'] = outdir.polygons.preview

            if cfg.segmentation.static:
                _ = seggrid.file.static
                mapping['segmentation.static'] = outdir.seggrid.static.dir
            if cfg.segmentation.prob:
                _ = seggrid.file.prob
                mapping['segmentation.prob'] = outdir.seggrid.prob.dir
            if cfg.segmentation.pred:
                _ = seggrid.file.pred
                mapping['segmentation.pred'] = outdir.seggrid.pred.dir
            if cfg.segmentation.colorized:
                _ = seggrid.file.colorized
                mapping['segmentation.colorized'] = outdir.seggrid.colorized.dir
            if cfg.segmentation.intensity:
                _ = seggrid.file.intensity
                mapping['segmentation.intensity'] = outdir.seggrid.intensity.dir
            if cfg.segmentation.sidebyside:
                _ = seggrid.file.sidebyside
                mapping['segmentation.sidebyside'] = outdir.seggrid.sidebyside.dir
            if cfg.segmentation.overlay:
                _ = seggrid.file.overlay
                mapping['segmentation.overlay'] = outdir.seggrid.overlay.dir
            if cfg.segmentation.error:
                _ = seggrid.file.error
                mapping['segmentation.error'] = outdir.seggrid.error.dir
            if cfg.segmentation.soft:
                _ = seggrid.file.soft
                mapping['segmentation.soft'] = outdir.seggrid.soft.dir

            if cfg.vectorization.static:
                _ = vecgrid.file.static
                mapping['vectorization.static'] = outdir.vecgrid.static.dir
            if cfg.vectorization.prob:
                _ = vecgrid.file.prob
                mapping['vectorization.prob'] = outdir.vecgrid.prob.dir
            if cfg.vectorization.pred:
                _ = vecgrid.file.pred
                mapping['vectorization.pred'] = outdir.vecgrid.pred.dir
            if cfg.vectorization.colorized:
                _ = vecgrid.file.colorized
                mapping['vectorization.colorized'] = outdir.vecgrid.colorized.dir
            if cfg.vectorization.intensity:
                _ = vecgrid.file.intensity
                mapping['vectorization.intensity'] = outdir.vecgrid.intensity.dir
            if cfg.vectorization.sidebyside:
                _ = vecgrid.file.sidebyside
                mapping['vectorization.sidebyside'] = outdir.vecgrid.sidebyside.dir
            if cfg.vectorization.overlay:
                _ = vecgrid.file.overlay
                mapping['vectorization.overlay'] = outdir.vecgrid.overlay.dir
            if cfg.vectorization.error:
                _ = vecgrid.file.error
                mapping['vectorization.error'] = outdir.vecgrid.error.dir
            if cfg.vectorization.soft:
                _ = vecgrid.file.soft
                mapping['vectorization.soft'] = outdir.vecgrid.soft.dir
            if cfg.vectorization.network:
                _ = vecgrid.file.network
                mapping['vectorization.network'] = outdir.vecgrid.network.dir
            if cfg.vectorization.polygons:
                _ = vecgrid.file.polygons
                mapping['vectorization.polygons'] = outdir.vecgrid.polygons.dir

            return mapping

    @property
    def existing(self) -> set[str]:
        grid = self.grid
        seggrid = grid.seggrid
        vecgrid = grid.vecgrid
        cfg = self.cfg

        paths = []

        def func():

            with grid.file, seggrid.file, vecgrid.file:

                if not cfg.static:
                    paths.append(grid.file.static)
                if not cfg.prob:
                    paths.append(grid.file.prob)
                if not cfg.pred:
                    paths.append(grid.file.pred)
                if not cfg.colorized:
                    paths.append(grid.file.colorized)
                if not cfg.intensity:
                    paths.append(grid.file.intensity)
                if not cfg.sidebyside:
                    paths.append(grid.file.sidebyside)
                if not cfg.overlay:
                    paths.append(grid.file.overlay)
                # if not cfg.error:
                #     paths.append(grid.file.error)
                if not cfg.soft:
                    paths.append(grid.file.soft)
                if not cfg.network:
                    paths.append(vecgrid.file.network)
                if not cfg.polygons:
                    paths.append(vecgrid.file.polygons)
                if not cfg.line.preview:
                    paths.append(grid.outdir.network.preview)
                if not cfg.polygon.preview:
                    paths.append(grid.outdir.polygons.preview)
                if cfg.download.only:
                    return

                if not cfg.segmentation.static:
                    paths.append(seggrid.file.static)
                if not cfg.segmentation.prob:
                    paths.append(seggrid.file.prob)
                if not cfg.segmentation.pred:
                    paths.append(seggrid.file.pred)
                if not cfg.segmentation.colorized:
                    paths.append(seggrid.file.colorized)
                if not cfg.segmentation.intensity:
                    paths.append(seggrid.file.intensity)
                if not cfg.segmentation.sidebyside:
                    paths.append(seggrid.file.sidebyside)
                if not cfg.segmentation.overlay:
                    paths.append(seggrid.file.overlay)
                # if not cfg.segmentation.error:
                #     paths.append(seggrid.file.error)
                if not cfg.segmentation.soft:
                    paths.append(seggrid.file.soft)
                if cfg.segmentation.only:
                    return

                if not cfg.vectorization.static:
                    paths.append(vecgrid.file.static)
                if not cfg.vectorization.prob:
                    paths.append(vecgrid.file.prob)
                if not cfg.vectorization.pred:
                    paths.append(vecgrid.file.pred)
                if not cfg.vectorization.colorized:
                    paths.append(vecgrid.file.colorized)
                if not cfg.vectorization.intensity:
                    paths.append(vecgrid.file.intensity)
                if not cfg.vectorization.sidebyside:
                    paths.append(vecgrid.file.sidebyside)
                if not cfg.vectorization.overlay:
                    paths.append(vecgrid.file.overlay)
                # if not cfg.vectorization.error:
                #     paths.append(vecgrid.file.error)
                if not cfg.vectorization.soft:
                    paths.append(vecgrid.file.soft)
                if not cfg.vectorization.network:
                    paths.append(vecgrid.file.network)
                if not cfg.vectorization.polygons:
                    paths.append(vecgrid.file.polygons)

        func()
        out = {
            string
            for it in paths
            if not isinstance(it, str)
               and isinstance(it, Iterable)
            for string in it
        }
        out.update(
            string
            for string in paths
            if isinstance(string, str)
        )
        return out


if __name__ == '__main__':
    grid = Grid.from_cfg()
    process = Process(grid)
    result = process.arg2dir

    if result:
        max_key_len = max(len(k) for k in result.keys())
        for key, directory in sorted(result.items()):
            logger.info(f"  {key:<{max_key_len}} -> {directory}")
    else:
        result = process.result
