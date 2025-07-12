from __future__ import annotations

import os

import geopandas as gpd
import shapely

from tile2net.tiles.cfg.logger import logger
from ..benchmark import benchmark
from ..cfg import cfg
from ..explore import explore
from ..fixed import GeoDataFrameFixed
from ..vectiles.mask2poly import Mask2Poly

if False:
    from .intiles import InTiles
    import folium


def __get__(
        self: Polygons,
        instance: InTiles,
        owner: type[InTiles]
) -> Polygons:

    self.intiles = instance
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:

        file = self.file
        if os.path.exists(file):
            msg = f"loading {instance.__name__}.{self.__name__} from {file}"
            logger.info(msg)
            result = gpd.read_parquet(file).pipe(self.__class__)
        else:

            vectiles = instance.vectiles
            n = len(instance.vectiles.polygons)
            grid_size = max(
                (affine.a ** 2 + affine.e ** 2) ** .5
                for affine in vectiles.affine_params
            )

            n_polygons = (
                vectiles.polygons
                .apply(shapely.get_num_geometries)
                .to_numpy()
                .sum()
            )

            n_features = len(vectiles.polygons.columns)
            msg = (
                f'Aggregating {n_polygons} polygons from {n} tiles and '
                f'{n_features} features into a single vector with grid '
                f'size {grid_size:.2e}. This may take a while.'
            )

            with benchmark(msg, level='info'):
                result = (
                    instance.vectiles.polygons
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

        instance.__dict__[self.__name__] = result

    result.intiles = instance

    return result


class Polygons(
    GeoDataFrameFixed
):
    __name__ = 'polygons'
    locals().update(
        __get__=__get__,
    )
    intiles: InTiles = None

    @property
    def file(self):
        return self.intiles.outdir.polygons.file

    def unlink(self):
        """Delete the polygons file."""
        file = self.file
        msg = (
            f'Uncaching {self.intiles.__name__}.{self.__name__} and '
            f'deleting file:\n\t{file}'
        )
        logger.info(msg)
        if os.path.exists(file):
            os.remove(file)
        del self.intiles

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            dash='5, 20',
            simplify=None,
            **kwargs,
    ) -> folium.Map:

        import folium
        feature2color = cfg.label2color
        it = self.groupby('feature', observed=False)
        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                *args,
                color=color,
                name=feature,
                tiles=tiles,
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

