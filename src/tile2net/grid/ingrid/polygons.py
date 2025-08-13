from __future__ import annotations


import os

import geopandas as gpd
import shapely

from tile2net.grid.cfg.logger import logger
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..vecgrid.mask2poly import Mask2Poly

from ..frame.framewrapper import FrameWrapper

if False:
    from .ingrid import InGrid
    import folium




class Polygons(
    FrameWrapper
):
    __name__ = 'polygons'

    def _get(
            self: Polygons,
            instance: InGrid,
            owner: type[InGrid]
    ) -> Polygons:

        self = super()._get(instance, owner)
        if instance is None:
            result = self
        elif self.__name__ in instance.__dict__:
            result = instance.frame.__dict__[self.__name__]
        else:

            file = self.file
            if os.path.exists(file):
                msg = f"loading {instance.__name__}.{self.__name__} from {file}"
                logger.info(msg)
                result = gpd.read_parquet(file).pipe(self.__class__)
            else:

                vecgrid = instance.vecgrid
                n = len(instance.vecgrid.polygons)
                grid_size = max(
                    (affine.a ** 2 + affine.e ** 2)
                    ** .5
                    for affine in vecgrid.affine_params
                )

                n_polygons = (
                    vecgrid.polygons
                    .apply(shapely.get_num_geometries)
                    .to_numpy()
                    .sum()
                )

                n_features = len(vecgrid.polygons.columns)
                msg = (
                    f'Aggregating {n_polygons} polygons from {n} grid and '
                    f'{n_features} features into a single vector with grid '
                    f'size {grid_size:.2e}. This may take a while.'
                )

                with benchmark(msg, level='info'):
                    result = (
                        instance.vecgrid.polygons
                        .stack(future_stack=True)
                        .set_precision(grid_size=grid_size)
                        .to_frame(name='geometry')
                        .pipe(Mask2Poly)
                        .postprocess()
                        .pipe(Polygons)
                    )

                msg = f"Writing {instance.__name__}.{self.__name__} to {file}"
                logger.info(msg)
                result.to_parquet(file)

            instance.frame.__dict__[self.__name__] = result

        result.ingrid = instance

        return result

    locals().update(
        __get__=_get,
    )
    ingrid: InGrid = None

    @property
    def file(self):
        return self.ingrid.outdir.polygons.file

    def unlink(self):
        """Delete the polygons file."""
        file = self.file
        msg = (
            f'Uncaching {self.ingrid.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.ingrid

    def explore(
            self,
            *args,
            grid='cartodbdark_matter',
            m=None,
            dash='5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:

        import folium
        feature2color = cfg.label2color
        it = self.frame.groupby('feature', observed=False)
        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                *args,
                color=color,
                name=feature,
                grid=grid,
                simplify=simplify,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                m=m,
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m

