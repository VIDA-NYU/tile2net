import logging
import shapely.ops
import datetime
import shutil
import warnings

import pandas as pd
import os

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

pd.options.mode.chained_assignment = None

from tile2net.raster.tile_utils.topology import *
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs, geo2geodf, buffer_union_erode
from tile2net.raster.tile_utils.topology import morpho_atts
from tile2net.raster.project import Project


class PedNet:
    """
    Create network from polygons
=======
class PedNet():
    """

    def __init__(
            self,
            poly: gpd.GeoDataFrame,
            project: Project
    ):

        self.polygons = poly
        self.nodes = []
        self.edges = []
        self.feature_type = -1
        self.island = -1
        self.sidewalk = -1
        self.crosswalk = -1
        self.complete_net = -1
        self.project = project

    def prepare_class_gdf(self, class_name) -> object:
        """
        Filters the polygon :class:`GeoDataFrame` based on the class label

        Parameters
        ----------
        class_name : str
            the class label, i.e. sidewalk, crosswalk, road

        Returns
        -------
        :class:`GeoDataFrame`
            class specific :class:`GeoDataFrame` in metric projection
        """
        nt = self.polygons[self.polygons.f_type == f'{class_name}'].copy()

        nt.geometry = nt.geometry.to_crs(3857)
        return nt

    def find_medianisland(self, swp, cwp, max_area=230):
        """
        Finds the median islands between sidewalks and crosswalks

        Parameters
        ----------
        swp : :class:`GeoDataFrame`
            sidewalks polygons
        cwp : :class:`GeoDataFrame`
            crosswalk polygons
        max_area : int
            maximum island area based to filter

        Returns
        -------
        :class:`GeoDataFrame`
            filtered sidewalk :class:`GeoDataFrame` without islands
        """

        # query to find crosswalks intersecting with two or more sidewalks
        inp, res = swp.geometry.values.sindex.query_bulk(cwp.geometry.values,
                                                         predicate="intersects")
        unique, counts = np.unique(inp, return_counts=True)
        meds = np.unique(res[np.isin(inp, unique[counts >= 2])])

        possible_median = swp[swp.index.isin(meds)]

        # refining the returned query based on the polygon area
        smallers = possible_median[possible_median.area < max_area]
        # remove the medians from sidewalk polygons
        updated_swp = swp[~(swp.index.isin(smallers.index))]
        # get islands/medians centroids to connect crosswalks to
        cent = smallers.centroid
        if len(cent) > 1:
            island_points = gpd.GeoDataFrame(geometry=cent, crs=3857)

            self.island = island_points
            return updated_swp
        else:
            return -1

    def validate_linemerge(self, merged_line):
        # from topojson https://github.com/mattijn/topojson/commit/cdc059bae53f3f5cfe882527e5d34e671f80173e
        """
        Returns
        -------
        list[shapely.geometry.LineString]
            If the linemerge was a MultiLineString, then returns a list of multiple single linestrings
        """

        if not isinstance(merged_line, shapely.geometry.LineString):
            merged_line = [ls for ls in merged_line.geoms]
        else:
            merged_line = [merged_line]
        return merged_line

    def make_longer(self, line, thr):
        """
        Extends a line by a given ratio

        Parameters
        ----------
        line : shapely.LineString
            The LineString to be extended

        thr : float
            ratio of the whole line by which the line is extended (e.g., 2/3)

        Returns
        -------
        shapely.LineString
            The extended line
        """
        lcoord = shapely.get_coordinates(line)
        lcoord = lcoord[-4:] if len(lcoord.shape) == 1 else lcoord[-2:].flatten()
        extended = get_extrapolated_line(lcoord, int(line.length * thr))
        line_checked = self.validate_linemerge(line)
        extended_checked = self.validate_linemerge(extended)
        new_l = MultiLineString(line_checked + extended_checked)
        merged = shapely.ops.linemerge(new_l)
        return merged

    def create_crosswalk(self):
        """
        Create crosswalks centerlines from polygons
        """
        # from .tile_utils.topology import morpho_atts

        nt_cw = self.prepare_class_gdf('crosswalk')
        if len(nt_cw) > 0:
            cw_lin_geom = []
            nt_cw.geometry = nt_cw.simplify(0.6)
            cw_union = buffer_union_erode(nt_cw, 1, -0.95, 0.6, 0.8,
                                          0.8)  # union, erode, simplify before union, simp
            # after union, simpl after erode
            cw_explode = cw_union.explode().reset_index(drop=True)
            cw_explode = cw_explode[cw_explode.geometry.notna()].reset_index(drop=True)
            cw_explode_ = morpho_atts(cw_explode)
            # polak = []
            for c, geom in enumerate(cw_explode_.geometry):
                if geom.area < 5:
                    continue
                    # logging.info(c, 'continue')
                else:
                    # if crosswalks are attached to each other and form a T or U
                    if cw_explode_.iloc[c, cw_explode_.columns.get_loc(cw_explode_.convexity.name)] < 0.8:
                        av_width = 4 * geom.area / geom.length
                        geom_er = geom.buffer(-av_width / 4)
                        # polak.append(geom_er)
                        if geom_er.type == "MultiPolygon":
                            for g in list(geom_er.geoms):  # shapely 2
                                if g.area > 2:
                                    cnl = to_cline(g, 0.3, 1)
                                    tr_line_ = trim_checkempty(cnl, 4.5, 2)
                                    if tr_line_.length < 8:
                                        extended = self.make_longer(tr_line_, 0.8)
                                        extended_line = extend_lines(geo2geodf([extended]),
                                                                     tolerance=8,
                                                                     target=geo2geodf([geom.boundary]), extension=0)
                                        for gi in extended_line.geometry:
                                            cw_lin_geom.append(gi)
                                    else:
                                        cw_lin_geom.append(tr_line_)
                                else:
                                    continue
                        elif geom_er.type == "Polygon":
                            if geom_er.area > 2:
                                cnl = to_cline(geom_er, 0.2, 1)
                                tr_line_ = trim_checkempty(cnl, 4.5, 2)
                                if tr_line_.length < 8:
                                    extended = self.make_longer(tr_line_, 0.8)
                                    extended_line = extend_lines(geo2geodf([extended]),
                                                                 target=geo2geodf([geom.boundary]), tolerance=8,
                                                                 extension=0)
                                    for g in extended_line.geometry:
                                        cw_lin_geom.append(g)
                                else:
                                    cw_lin_geom.append(tr_line_)
                            else:
                                continue
                        else:
                            continue
                    else:
                        # polak.append(geom)
                        line = get_crosswalk_cnl(geom)
                        if line.length < 8:
                            extended = self.make_longer(line, 0.8)
                            extended_line = extend_lines(geo2geodf([extended]),
                                                         target=geo2geodf([geom.boundary]), tolerance=8, extension=0)
                            for g in extended_line.geometry:
                                cw_lin_geom.append(g)
                        else:
                            cw_lin_geom.append(line)

            cw_ntw = geo2geodf(cw_lin_geom)
            cw_ntw['f_type'] = 'crosswalk'
            cw_ntw.geometry = cw_ntw.geometry.set_crs(3857)
            smoothed = wrinkle_remover(cw_ntw, 1.3)
            self.crosswalk = smoothed
        else:
            warnings.warn('No crosswalks found')
            self.crosswalk = gpd.GeoDataFrame({
                'geometry': [],
            }, crs=3857)

    def create_lines(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create centerlines from polygons

        Parameters
        ----------
        gdf : :class:`GeoDataFrame`
            :class:`GeoDataFrame` of polygons

        Returns
        -------
        :class:`GeoDataFrame` | None
            Centerline polygons based on the given polygons, or None if it is not possible

        """
        lin_geom = []
        gdf_atts = morpho_atts(gdf)
        for c, geom in enumerate(gdf.geometry):
            corners = gdf_atts.iloc[c, gdf_atts.columns.get_loc(gdf_atts.corners.name)]
            minpr = 2 * math.sqrt(math.pi * abs(geom.area))
            trim1 = 20
            trim2 = 6
            if geom.area <= 20:  # 45 DC #10 others
                continue
            elif minpr / geom.length > 0.8:
                # it is close to a circle
                # the interpolation distance to be 2/3rd of the minimum circle perimeter
                cl_arg = math.sqrt(geom.area / math.pi) / 4
            else:
                cl_arg = 0.2

            line = to_cline(geom, cl_arg, 1)
            if not line.is_empty:
                tr_line_ = trim_checkempty(line, trim1, trim2)
                if corners > 100:
                    tr_line = trim_checkempty(tr_line_, trim1, trim2)
                else:
                    tr_line = tr_line_

            else:
                line_clh = to_cline(geom, cl_arg / 2, 1)
                if not line_clh.is_empty:
                    tr_line_ = trim_checkempty(line_clh, trim1, trim2)
                    if corners > 100:
                        tr_line = trim_checkempty(tr_line_, trim1, trim2)
                    else:
                        tr_line = tr_line_
                else:
                    new_line = to_cline(geom, 0.1, 0.5)
                    tr_line_ = trim_checkempty(new_line, trim1, trim2)
                    if corners > 100:
                        tr_line = trim_checkempty(tr_line_, trim1, trim2)
                    else:
                        tr_line = tr_line_
            if tr_line.is_empty:
                logging.debug('empty line')
                continue
            else:
                line_tr = tr_line.simplify(1)
                extended_line = extend_lines(geo2geodf([line_tr]),
                                             target=geo2geodf([geom.boundary]), tolerance=6, extension=0)
                lin_geom.append(extended_line)

        if len(lin_geom) > 0:
            ntw = pd.concat(lin_geom)
            smoothed = wrinkle_remover(ntw, 1.5)
            return smoothed

    def create_sidewalks(self):
        """
        Create sidewalk network

        Returns
        -------
        None
        """
        sw_all = self.prepare_class_gdf('sidewalk')

        if len(sw_all) > 0:
            swntw = self.create_lines(sw_all)
            logging.info('..... creating the processed sidewalk network')
            #
            swntw.geometry = swntw.simplify(0.6)
            sw_modif_uni = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([geom for geom in swntw.unary_union.geoms]))
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

                self.sidewalk = sw_cleaned
            except:
                # logging.info('cannot save modified')
                self.sidewalk = sw_uni_lines
        else:
            warnings.warn('No sidewalk polygons found')
            self.sidewalk = gpd.GeoDataFrame({
                'geometry': [],
            }, crs=3857)

        self.sidewalk['f_type'] = 'sidewalk'

    def convert_whole_poly2line(self):
        """
        Create network from the full polygon dataset
        """
        logging.info('Starting network creation...')

        self.create_sidewalks()
        self.create_crosswalk()

        # connect the crosswalks to the nearest sidewalks
        points = get_line_sepoints(self.crosswalk)

        # query LineString geometry to identify points intersecting 2 geometries
        # inp, res = self.crosswalk.sindex.query(geo2geodf(points).geometry,
        #                                        predicate="intersects")
        inp, res = (
            self.crosswalk.sindex
            .query(geo2geodf(points).geometry, predicate="intersects")
        )
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

            connect_s = get_shortest(self.sidewalk, pdfs, f_type='sidewalk_connection')
            all_connections.append(connect_s)

        if len(new_geoms_e) > 0:
            pe = [g[1] for g in new_geoms_e]
            # le = [g[0] for g in new_geoms_e]
            pdfe = gpd.GeoDataFrame(geometry=pe)
            pdfe.set_crs(3857, inplace=True)

            connect_e = get_shortest(self.sidewalk, pdfe, f_type='sidewalk_connection')
            all_connections.append(connect_e)

        if len(new_geoms_both) > 0:
            pb = [g[1] for g in new_geoms_both]
            # lb = [g[0] for g in new_geoms_both]  # crosswalk lines where both ends do not intersect
            pdfb = gpd.GeoDataFrame(geometry=pb)
            pdfb.set_crs(3857, inplace=True)

            connect_b = get_shortest(self.sidewalk, pdfb, f_type='sidewalk_connection')
            all_connections.append(connect_b)
        if len(all_connections) > 1:
            connect = pd.concat(all_connections)
        elif len(all_connections) == 1:
            connect = all_connections[0]
        else:
            connect = []

        if len(all_connections) > 0:
            # manage median islands
            if not isinstance(self.island, int):
                nearest_cw = self.island.sindex.nearest(pdfb.geometry, max_distance=7)
                indcwnear = list(zip(nearest_cw[0], nearest_cw[1]))
                # island lines
                # iterate over the returned indices and draw the shortest line between the two geometries
                island_lines = []

                for k, v in indcwnear:
                    island_lines.append(
                        shapely.shortest_line(self.island.geometry.values[v],
                                              pdfb.geometry.values[k]))

                island = gpd.GeoDataFrame(geometry=island_lines)

                island.geometry = island.geometry.set_crs(3857)
                island['f_type'] = 'medians'
                combined = pd.concat([self.crosswalk, connect, self.sidewalk, island])
            else:
                combined = pd.concat([self.crosswalk, connect, self.sidewalk])

        else:
            combined = pd.concat([self.crosswalk, self.sidewalk])

        combined.dropna(inplace=True)
        combined.geometry = combined.geometry.set_crs(3857)
        combined.geometry = combined.geometry.to_crs(4326)
        combined = combined[~combined.geometry.isna()]
        combined.drop_duplicates(subset='geometry', inplace=True)
        combined.reset_index(drop=True, inplace=True)
        path = self.project.network.path

        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath(f'{self.project.name}-Network-{datetime.datetime.now().strftime("%d-%m-%Y_%H_%M")}')
        if os.path.exists(path):
            shutil.rmtree(path)
        combined.to_file(path)

        self.complete_net = combined
