from __future__ import annotations

from functools import *

import shapely.wkt
from centerline.geometry import Centerline
from tqdm import tqdm

from tile2net.logger import logger
from .standalone import  Lines
from . import mintrees, stubs
_ = mintrees, stubs
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs
from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from ..fixed import GeoDataFrameFixed

if False:
    from .pednet import PedNet
    import folium


def __get__(
        self,
        instance: PedNet,
        owner: type[PedNet]
) -> Center:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        union = instance.union
        geometry = union.geometry

        warn = (
            f'High variance in polygon areas may cause progress-rate '
            f'fluctuations for centerline computation. '
        )
        logger.debug(warn)

        centers = []
        for i, poly in tqdm(
                enumerate(geometry),
                total=len(geometry),
                desc='Centerlines',
                leave=False
        ):
            try:
                centers.append(Centerline(poly).geometry)
            except Exception as e:
                err = f'Centerline computation failed for index {i}: {e}'
                tqdm.write(err)

        multilines = np.asarray(centers, dtype=object)
        repeat = shapely.get_num_geometries(multilines)
        lines = shapely.get_parts(multilines)
        iloc = np.arange(len(repeat)).repeat(repeat)

        result = (
            union
            .iloc[iloc]
            .set_geometry(lines)
            .pipe(Lines.from_frame)
            .drop2nodes
            .pipe(Center)
        )

        lines = shapely.simplify(result.geometry, .01)
        result = result.set_geometry(lines)
        result.index.name = 'icent'
        instance.__dict__[self.__name__] = result

    result.instance = instance
    return result


class Center(
    GeoDataFrameFixed,
):
    locals().update(
        __get__=__get__,
    )

    instance: PedNet = None
    __name__ = 'center'

    @cached_property
    def cleaned(self) -> gpd.GeoDataFrame:
        swntw = self
        geometry = swntw.union_all()
        sw_modif_uni = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(geometry, crs=swntw.crs),
            index=swntw.index,
        )
        sw_modif_uni_met = set_gdf_crs(sw_modif_uni, 3857)
        sw_uni_lines = sw_modif_uni_met.explode()
        sw_uni_lines.reset_index(drop=True, inplace=True)
        sw_uni_lines.geometry = sw_uni_lines.simplify(2)
        sw_uni_lines.dropna(inplace=True)
        sw_uni_line2 = sw_uni_lines.copy()

        try:
            sw_cl1 = clean_deadend_dangles(sw_uni_line2)
            sw_extended = extend_lines(sw_cl1, 10, extension=0)

            sw_cleaned = remove_false_nodes(sw_extended)
            sw_cleaned.reset_index(drop=True, inplace=True)
            sw_cleaned.geometry = sw_cleaned.geometry.set_crs(3857)
        except:
            result = sw_uni_lines
        else:
            result = sw_cleaned
        return result

    @cached_property
    def clipped(self) -> gpd.GeoDataFrame:
        msg = 'Clipping centerlines to the features'
        logger.debug(msg)
        lines = self
        features = self.instance.features
        geometry = features.mutex
        predicate = "intersects"
        ifeat, iline = lines.sindex.query(geometry, predicate)
        features = features.iloc[ifeat]
        lines = lines.iloc[iline]
        geometry = (
            lines
            .intersection(features.mutex, align=False)
            .reset_index(drop=True)
            .geometry
        )
        result = Center({
            'feature': features.feature.values,
            'geometry': geometry,
        })
        result.index.name = 'iclip'
        return result

    @cached_property
    def lines(self) -> Lines:
        center = self
        lines = Lines.from_frame(center)
        lines.pednet = self.instance
        return lines


    @cached_property
    def crosswalk(self) -> gpd.GeoDataFrame:
        loc = self.clipped.feature == 'crosswalk'
        crosswalk = self.clipped.loc[loc].copy()
        return crosswalk

    @cached_property
    def sidewalk(self) -> gpd.GeoDataFrame:
        loc = self.clipped.feature == 'sidewalk'
        sidewalk = self.clipped.loc[loc].copy()
        return sidewalk

    @cached_property
    def result(self):
        """
        Create network from the full polygon dataset
        """
        logger.info('Starting network creation...')

        self.create_sidewalks()
        self.create_crosswalk()

        # connect the crosswalks to the nearest sidewalks
        points = get_line_sepoints(self.crosswalk)

        # query LineString geometry to identify points intersecting 2 geometries
        inp, res = (
            self.crosswalk.sindex
            .query(geo2geodf(points).geometry, predicate="intersects")
        )
        gpd.GeoDataFrame().sindex
        unique, counts = np.unique(inp, return_counts=True)
        ends = np.unique(res[np.isin(inp, unique[counts == 1])])

        new_geoms_s = []
        new_geoms_e = []
        new_geoms_both = []
        all_connections = []
        # iterate over crosswalk segments that are not connected to other crosswalk segments
        # and add the start and end points to the new_geoms
        pgeom = self.crosswalk.geometry.values
        for line in ends:
            l_coords = shapely.get_coordinates(pgeom[line])

            start = Point(l_coords[0])
            end = Point(l_coords[-1])

            first = list(pgeom.sindex.query(start, predicate="intersects"))
            second = list(pgeom.sindex.query(end, predicate="intersects"))
            first.remove(line)
            second.remove(line)

            if first and not second:
                new_geoms_s.append((line, end))

            elif not first and second:
                new_geoms_e.append((line, start))
            if not first and not second:
                new_geoms_both.append((line, start))
                new_geoms_both.append((line, end))

        # create a dataframe of points
        if len(new_geoms_s) > 0:
            ps = [g[1] for g in new_geoms_s]
            # ls = [g[0] for g in new_geoms_s]
            pdfs = gpd.GeoDataFrame(geometry=ps)
            pdfs.set_crs(3857, inplace=True)

            connect_s = get_shortest(self.sidewalk, pdfs, feature='sidewalk_connection')
            all_connections.append(connect_s)

        if len(new_geoms_e) > 0:
            pe = [g[1] for g in new_geoms_e]
            # le = [g[0] for g in new_geoms_e]
            pdfe = gpd.GeoDataFrame(geometry=pe)
            pdfe.set_crs(3857, inplace=True)

            connect_e = get_shortest(self.sidewalk, pdfe, feature='sidewalk_connection')
            all_connections.append(connect_e)

        if len(new_geoms_both) > 0:
            pb = [g[1] for g in new_geoms_both]
            # lb = [g[0] for g in new_geoms_both]  # crosswalk lines where both ends do not intersect
            pdfb = gpd.GeoDataFrame(geometry=pb)
            pdfb.set_crs(3857, inplace=True)

            connect_b = get_shortest(self.sidewalk, pdfb, feature='sidewalk_connection')
            all_connections.append(connect_b)
        if len(all_connections) > 1:
            connect = pd.concat(all_connections)
        elif len(all_connections) == 1:
            connect = all_connections[0]
        else:
            connect = []

        if len(all_connections) > 0:
            # manage median islands
            combined = pd.concat([self.crosswalk, connect, self.sidewalk])
        else:
            combined = pd.concat([self.crosswalk, self.sidewalk])

        combined.dropna(inplace=True)
        combined.geometry = combined.geometry.set_crs(3857)
        combined.geometry = combined.geometry.to_crs(4326)
        combined = combined[~combined.geometry.isna()]
        combined.drop_duplicates(subset='geometry', inplace=True)
        combined.reset_index(drop=True, inplace=True)

        return combined

    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line_color='grey',
            node_color='red',
            simplify: float = None,
            dash='5, 20',
            attr: str = None,
            **kwargs,
    ) -> folium.Map:
        import folium
        features = self.instance.features
        feature2color = features.color.to_dict()
        _ = features.mutex
        it = features.groupby(level='feature', observed=False)

        lines = self
        if attr:
            lines = getattr(self, attr)
        for feature, frame in it:
            color = feature2color[feature]
            m = explore(
                frame,
                geometry='mutex',
                *args,
                color=color,
                name=f'{feature} polygons',
                tiles=tiles,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                **kwargs,
            )

        if 'feature' in lines.columns:
            it = lines.groupby('feature', observed=False)
            for feature, frame in it:
                color = feature2color[feature]
                m = explore(
                    frame,
                    *args,
                    color=color,
                    name=f'{feature} lines',
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    **kwargs,
                )
        else:
            m = explore(
                lines,
                *args,
                color=line_color,
                name='centerlines',
                tiles=tiles,
                simplify=simplify,
                m=m,
                **kwargs,
            )

        if node_color:
            # nodes = lines.nodes
            # nodes = self.node
            m = explore(
                nodes,
                *args,
                color=node_color,
                name='nodes',
                tiles=tiles,
                m=m,
                # style_kwds=dict(
                #     radius=3,
                #     fill=True,
                #     fill_opacity=1.0,
                #     weight=1.0,
                #     dashArray=dash,
                # ),
                **kwargs,
            )

        folium.LayerControl().add_to(m)
        return m
