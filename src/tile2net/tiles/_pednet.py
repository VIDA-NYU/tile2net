from __future__ import annotations
import logging
import datetime
from functools import cached_property

import pandas as pd
import os

import shapely

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

pd.options.mode.chained_assignment = None

from tile2net.raster.tile_utils.topology import *
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs, geo2geodf, buffer_union_erode
from tile2net.raster.tile_utils.topology import morpho_atts
from tile2net.raster.project import Project

METRIC_CRS = 3857


class NodeProcess:
    # def __init__(self, node: Node, project: Project):
    #     self.node = node
    #     self.project = project
    #     self.sidewalk = -1
    #     self.crosswalk = -1
    #     self.island = -1
    #     self.feature_type = -1
    #     self.complete_net = -1

    def __repr__(self):
        return f'NodeProcess({self.node})'
class  PedPoly:
    def __init__(self, pols: gpd.GeoDataFrame):
        self.pols = pols

    @cached_property
    def metric_polygons(self, metric_crs=METRIC_CRS):
        """Converts polygons to metric projection
        Parameters
        ----------
        metric_crs :
            metric projection
        Returns
        -------
        GeoDataFrame
            metric projected GeoDataFrame
        """
        metric = self.pols.copy()
        metric.geometry = metric.geometry.to_crs(metric_crs)
        return metric

    @cached_property
    def polygon_shapeinfo(self):
        return morpho_atts(self.metric_polygons)

    @cached_property
    def crosswalk_polygons(self):
        return self.polygon_shapeinfo[(self.polygon_shapeinfo.f_type == 'crosswalk')
                                      & (self.polygon_shapeinfo.area > 10)]

    @cached_property
    def sidewalk_polygons(self):
        return self.polygon_shapeinfo[(self.polygon_shapeinfo.f_type == 'sidewalk')
                                      & (self.polygon_shapeinfo.area > 15)]

    @cached_property
    def road_polygons(self):
        return self.polygon_shapeinfo[(self.polygon_shapeinfo.f_type == 'road')
                                      & (self.polygon_shapeinfo.area > 20)]

    def get_compact_shapes(self, class_name, metric_crs=METRIC_CRS):
        ...

    def get_elongated_shapes(self, class_name, metric_crs=METRIC_CRS):
        ...

    def get_rectangular_shapes(self, class_name, metric_crs=METRIC_CRS):
        ...

    def get_circular_shapes(self, class_name: str, metric_crs=METRIC_CRS):
        ...

    def handle_complex_shapes(self, class_name: str, metric_crs=METRIC_CRS):
        ...

    @cached_property
    def connected_crosswalks(self):
        return self.crosswalk_polygons[self.crosswalk_polygons.convexity <= 0.85]




class PedNet:
    """Create network from polygons
=======
class PedNet():
    """

    def __init__(
            self,
            poly: gpd.GeoDataFrame,
            project: Project
    ):

        self.polygons = PedPoly(poly)
        self.nodes: gpd.GeoDataFrame
        self.edges: gpd.GeoDataFrame
        self.feature_type: gpd.GeoDataFrame
        self.island: gpd.GeoDataFrame
        self.sidewalk: gpd.GeoDataFrame
        self.crosswalk: gpd.GeoDataFrame
        self.complete_net: gpd.GeoDataFrame
        self.project = project

    def __repr__(self):
        ...
    """
    perpare the data for the network
    """
    @staticmethod
    def get_complex_lines(gdf: gpd.GeoDataFrame):
        """Get complex lines from a GeoDataFrame of LineStrings."""

        p_per_line = shapely.get_num_coordinates(gdf.geometry.array)

        complex_lind = np.where(p_per_line > 2)
        return gdf.loc[complex_lind]
    @staticmethod
    def get_simple_lines(gdf: gpd.GeoDataFrame):
        """Get simple lines from a GeoDataFrame of LineStrings."""
        p_per_line = shapely.get_num_coordinates(gdf.geometry.array)
        return gdf.loc[np.where(p_per_line == 2)]

    @staticmethod
    def _2simple_lines(line: LineString) -> list:
        """Split a complex LineString into multiple LineStrings with only two points each."""
        coords = np.array(line.coords)

        # If the LineString is already simple, just return it
        if coords.shape[0] == 2:
            return [line]

        # Get start and end points for simple LineStrings
        start_points = coords[:-1]
        end_points = coords[1:]

        # Pair up start and end points
        simple_coords = np.stack((start_points, end_points), axis=1)

        # Convert each pair into a LineString
        simple_lines = shapely.linestrings(simple_coords)

        return simple_lines

    def make_simple_lines(self, gdf: gpd.GeoDataFrame):
        """Vectorized Split complex LineStrings into multiple LineStrings with only two points each."""

        simple_lines = np.vectorize(self._2simple_lines)
        return np.concatenate(simple_lines(self.get_complex_lines(gdf).geometry.array))

    def all_simpled_lines(self, gdf: gpd.GeoDataFrame):
        """Get all simple lines from a GeoDataFrame of LineStrings. (complex and simple)"""
        return np.concatenate([self.make_simple_lines(gdf), self.get_simple_lines(gdf)])


    @staticmethod
    def filter_lneighbors(line: LineString, parray: np.ndarray):
        """Filter points that fall within the line's x bounding box."""

        # Get the bounding box of the line
        minx, _, maxx, _ = line.envelope.bounds

        # Create a boolean mask based on the condition
        mask = (parray[:, 0] >= minx) & (parray[:, 0] <= maxx)

        # Use the mask to filter the points
        return parray[mask]

    @staticmethod
    def search_box(pcoords, thr: float):
        return np.array([pcoords + thr, pcoords - thr]).flatten()

    @staticmethod
    def filter_coords_by_bbox(coords, box):
        """Filter coordinates based on a bounding box.

        Args:
            coords (np.array): Array of shape (n, 2) containing coordinates.
            box (np.array): Array of shape (4,) containing the maxx, maxy, minx,miny coords


        Returns:
            np.array: Filtered coordinates.
        """
        # Filtering using logical indexing
        maxx, maxy, minx, miny = box
        filtered_coords = coords[(coords[:, 0] >= minx) &
                                 (coords[:, 0] <= maxx) &
                                 (coords[:, 1] >= miny) &
                                 (coords[:, 1] <= maxy)]

        return filtered_coords



    def prepare_class_gdf(self, class_name) -> object:
        """Filters the polygon geodataframe based on the class label
        Parameters
        ----------
        class_name :
            the class label, i.e. sidewalk, crosswalk, road
        Returns
        -------
        GeoDataFrame
            class specific GeoDataFrame in metric projection
        """

        return self.polygons.polygon_shapeinfo[self.polygons.polygon_shapeinfo.f_type == f'{class_name}'].copy()

    @property
    def sidewlk_polygons(self):
        return self.polygons.sidewalk_polygons

    @property
    def crosswalk_polygons(self):
        return self.polygons.crosswalk_polygons
    @property
    def road_polygons(self):
        return self.polygons.road_polygons

    @staticmethod
    def get_node_stats(network: gpd.GeoDataFrame):
        """Get node statistics from the network
        Parameters`
        ----------
        network :
            network GeoDataFrame with simple lines
        Returns
        -------
        GeoDataFrame
            node statistics
        """
        pcoords = shapely.get_coordinates(network.geometry.array)
        p_per_line = shapely.get_num_points(network.geometry.values)
        line_ind = np.repeat(np.arange(len(p_per_line)), p_per_line)
        upoints, uidx = np.unique(pcoords, axis=0, return_inverse=True)
        _, degree = np.unique(pcoords, axis=0, return_counts=True)
        number_to_count = dict(zip(range(len(upoints)), degree))
        pgdf = pd.DataFrame({'p_ind': range(len(uidx)), 'uniq_ind': uidx, 'line_ind': line_ind})
        pgdf['degree'] = pgdf.uniq_ind.map(number_to_count)

        uni_gdf = gpd.GeoDataFrame({'geometry': shapely.points(upoints), 'degree': degree})
        return pgdf, uni_gdf

    def get_dangling_edges(self, network: gpd.GeoDataFrame[LineString]):
        """Get dangling edges from the network
        Parameters
        ----------
        network :
            network GeoDataFrame
        Returns
        -------
        GeoDataFrame
            dangling edges
        """
        node_stats, _ = self.get_node_stats(network)
        return node_stats[node_stats.degree == 1]

    def filter_edges_degree(self, network: gpd.GeoDataFrame[LineString], degree: int):
        """Filter edges based on degree
        Parameters
        ----------
        network :
            network GeoDataFrame
        degree :
            degree of edges
        Returns
        -------
        GeoDataFrame
            filtered edges
        """
        node_stats, _ = self.get_node_stats(network)
        return node_stats[node_stats.degree == degree]

    @staticmethod
    def round_geom(geom: shapely.geometry, precision: int = 3):
        round_geom = np.vectorize(lambda geomet: shapely.transform(geomet, lambda g: g.round(precision)))
        return round_geom(geom)

    def centerline_trim(self, geom, interpol, simp, trim_length, ):
        ...
    @property
    def vectorized_centerline(self):
        return np.vectorize(to_cline)
    def handle_connected_crosswalks(self, line_length_thresh: int = 3.5):
        cw_lin_geom = []
        pol = self.polygons.connected_crosswalks[self.polygons.connected_crosswalks.geom_type == 'Polygon']
        multipol = self.polygons.connected_crosswalks[self.polygons.connected_crosswalks.geom_type == 'MultiPolygon']
        vec_to_cline = np.vectorize(to_cline)
        vec_trim_checkempty = np.vectorize(trim_checkempty)

        # round = np.vectorize(lambda geom: shapely.apply(geom, lambda g: g.round(3)))

        #
        #
        #
        #
        #
        #         for g in list(geom_er.geoms):  # shapely 2
        #             if g.area > 2:
        #                 cnl = to_cline(g, 0.3, 1)
        #                 tr_line_ = trim_checkempty(cnl, 4.5, 2)
        #                 if tr_line_.length < line_length_thresh:
        #                     cw_lin_geom = self.process_short_lines(geom_er, tr_line_,
        #                                                            geom_list=cw_lin_geom,
        #                                                            make_longer_thresh=3 / 4)
        #                 else:
        #                     cw_lin_geom.append(tr_line_)
        #
        #             else:
        #                 continue
        #     elif geom_er.geom_type == "Polygon":
        #         if geom_er.area > 2:
        #             cnl = to_cline(geom_er, 0.2, 1)
        #             tr_line_ = trim_checkempty(cnl, 4.5, 2)
        #             if tr_line_.length < line_length_thresh:
        #                 cw_lin_geom = self.process_short_lines(geom_er, tr_line_,
        #                                                        geom_list=cw_lin_geom,
        #                                                        make_longer_thresh=3 / 4)
        #
        #             else:
        #                 cw_lin_geom.append(tr_line_)
        #         else:
        #             continue
        #     else:
        #         continue
        #     ,
        # ...

    def filter_polyons(self, base_gdf, mask_gdf, metric_crs=METRIC_CRS):
        return base_gdf[~base_gdf.index.isin(mask_gdf.index)].copy()

    def find_medianisland(self, swp, cwp, max_area=230):
        """
        Finds the median islands between sidewalks and crosswalks
        Args:
            swp: swidewalks polygons
            cwp: crosswalk polygons
            max_area: maximum island area based to filter
        Returns:
            filtered sidewalk geodataframe without islands
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
            island_points = gpd.GeoDataFrame(geometry=cent, crs=METRIC_CRS)

            self.island = island_points
            return updated_swp
        else:
            return -1

    def validate_linemerge(self, merged_line):
        # from topojson https://github.com/mattijn/topojson/commit/cdc059bae53f3f5cfe882527e5d34e671f80173e
        """
        Return list of linestrings. If the linemerge was a MultiLineString
        then returns a list of multiple single linestrings
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
        line: shapely.LineString

        thr: float
            ratio of the whole line by which the line is extended (e.g., 2/3)

        Returns
        -------
        shapely.LineString
                The extended line
        """
        lcoord = shapely.get_coordinates(line)
        lcoord = lcoord[-4:] if len(lcoord.shape) == 1 else lcoord[-2:].flatten()
        extended = get_extended_line(lcoord, int(line.length * thr))
        line_checked = self.validate_linemerge(line)
        extended_checked = self.validate_linemerge(extended)
        new_l = MultiLineString(line_checked + extended_checked)
        merged = shapely.ops.linemerge(new_l)
        return merged

    def update_sw_intersecting(self, cw_pol):
        """
        Cleans up sidewalk lines extending over crosswalks polygons and cut them
        Parameters
        ----------
        cw_pol: gpd.GeoDataFrame
            geodataframe of crosswalks
        """
        # finding sidewalk lines that extends to the crosswalks polygons and cut them
        swl = self.sidewalk.copy()
        swl = swl.reset_index(drop=True).explode().reset_index(drop=True)
        # print(swl.head(10))
        cwrect = [i.minimum_rotated_rectangle for i in cw_pol.geometry]
        cwrectgdf = geo2geodf(cwrect)

        cwin, swin = swl.geometry.values.sindex.query_bulk(cwrectgdf.geometry.values,
                                                           predicate="intersects")
        indsect = list(zip(cwin, swin))
        inds = []
        updated_geom = []
        for v, k in indsect:
            if swl.iloc[k, 1].intersects(cwrectgdf.iloc[v, 0]):
                inds.append(k)
                updated_geom.append(
                    swl.iloc[k, swl.columns.get_loc(swl.geometry.name)].difference(
                        cwrectgdf.iloc[v, 0]))
        swl.iloc[inds, swl.columns.get_loc(swl.geometry.name)] = updated_geom
        smoothed = wrinkle_remover(swl, 1)
        self.sidewalk = smoothed

    def process_short_lines(self, pol_geom, line, geom_list: list, **kwargs) -> list:
        """
        extend the short lines created as the artifact of simplifaction or Voronoi
        and extend them to reach the polygon boundary

        Parameters
        ----------
        geom_list : list
        line: shapely.LineString
        pol_geom: shapely.Polygon
        line_length_thresh: float
            threshold of
        Returns
        -------

        """
        extended = self.make_longer(line, kwargs['make_longer_thresh'])
        extended_line = extend_lines(geo2geodf([extended]),
                                     tolerance=5,
                                     target=geo2geodf([pol_geom.boundary]), extension=0)
        for g in extended_line.geometry:
            geom_list.append(g)
        return geom_list

    def process_crosswalk_geometry(self, geom_area_thresh: float = 10, line_length_thresh=6) -> \
            tuple[list[tuple, tuple], list[tuple, tuple]]:
        """Processes the geometry of the crosswalk GeoDataFrame, removing small areas and handling complex shapes.
        Parameters
        ----------
        gdf : GeoDataFrame
            the GeoDataFrame representing the crosswalks
        geom_area_thresh : float
            the minimum area threshold for the crosswalk geometry
        Returns
        -------
        List
            a list of processed geometries
        """
        gdf = self.crosswalk_polygons.copy()
        polak = []
        cw_lin_geom = []
        filtered_geom = gdf[gdf.geometry.area > geom_area_thresh]

        for c, geom in enumerate(filtered_geom.geometry):
            # if crosswalks are attached to each other and form a T or U
            if gdf.iloc[
                c, gdf.columns.get_loc(gdf.convexity.name)] < 0.81:
                av_width = 1.5 * geom.area / geom.length
                geom_er = geom.buffer(-av_width / 2)
                polak.append(geom_er)
                if geom_er.geom_type == "MultiPolygon":
                    for g in list(geom_er.geoms):  # shapely 2
                        if g.area > 2:
                            cnl = to_cline(g, 0.3, 1)
                            tr_line_ = trim_checkempty(cnl, 4.5, 2)
                            if tr_line_.length < line_length_thresh:
                                cw_lin_geom = self.process_short_lines(geom_er, tr_line_,
                                                                       geom_list=cw_lin_geom,
                                                                       make_longer_thresh=3 / 4)
                            else:
                                cw_lin_geom.append(tr_line_)

                        else:
                            continue
                elif geom_er.geom_type == "Polygon":
                    if geom_er.area > 2:
                        cnl = to_cline(geom_er, 0.2, 1)
                        tr_line_ = trim_checkempty(cnl, 4.5, 2)
                        if tr_line_.length < line_length_thresh:
                            cw_lin_geom = self.process_short_lines(geom_er, tr_line_,
                                                                   geom_list=cw_lin_geom,
                                                                   make_longer_thresh=3 / 4)

                        else:
                            cw_lin_geom.append(tr_line_)
                    else:
                        continue
                else:
                    continue
            else:
                polak.append(geom)
                line = get_crossing_lines(geom)
                if line.length < line_length_thresh:
                    if line.length < line_length_thresh:
                        cw_lin_geom = self.process_short_lines(geom, line,
                                                               geom_list=cw_lin_geom,
                                                               make_longer_thresh=1 / 2)
                else:
                    cw_lin_geom.append(line)
        return cw_lin_geom, polak

    def create_lines(self, gdf: gpd.GeoDataFrame, trim1: int = 20, trim2: int = 15) -> gpd.GeoDataFrame:
        """
        Create centerlines from polygons
        Parameters
        ----------
        gdf: gpd.GeoDataFrame
            geodataframes of polygons
        trim1, trim2: the triming parameters
        -------

        """
        lin_geom = []
        gdf_atts = morpho_atts(gdf)
        for c, geom in enumerate(gdf.geometry):
            corners = gdf_atts.iloc[c, gdf_atts.columns.get_loc(gdf_atts.corners.name)]
            minpr = 2 * math.sqrt(math.pi * abs(geom.area))
            if geom.area <= 20:  # 45 DC
                continue
            elif minpr / geom.length > 0.8:
                # it is close to a circle
                # the interpolation distance to be 2/3rd of the minimum circle perimeter
                cl_arg = math.sqrt(geom.area / math.pi) / 3
            else:
                cl_arg = 0.2

            # Process the line with the initial parameters
            tr_line = process_line(geom, cl_arg, corners, trim1, trim2)
            if tr_line is None:
                # Process the line with half the interpolation distance
                tr_line = process_line(geom, cl_arg / 2, corners, trim1, trim2)
            if tr_line is None:
                # Process the line with fixed interpolation distance and minimum distance
                tr_line = process_line(geom, 0.1, corners, trim1, trim2)

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

    def create_crosswalk(self):
        """
        Create crosswalks centerlines from polygons
        """
        # from .tile_utils.topology import morpho_atts

        # nt_cw = self.prepare_class_gdf('crosswalk')
        if len(self.crosswalk_polygons) > 0:
            lines, polys = self.process_crosswalk_geometry()

            cw_ntw = geo2geodf(lines)
            cw_ntw['f_type'] = 'crosswalk'
            cw_ntw.geometry = cw_ntw.geometry.set_crs(METRIC_CRS)
            smoothed = wrinkle_remover(cw_ntw, 1.2)
            self.crosswalk = smoothed
        else:
            print('No crosswalks found')

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
            sw_modif_uni_met = set_gdf_crs(sw_modif_uni, METRIC_CRS)
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
                sw_cleaned.geometry = sw_cleaned.geometry.set_crs(METRIC_CRS)

                self.sidewalk = sw_cleaned
            except:
                # logging.info('cannot save modified')
                self.sidewalk = sw_uni_lines

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
        inp, res = self.crosswalk.sindex.query(geo2geodf(points).geometry,
                                               predicate="intersects")
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
            pdfs.set_crs(METRIC_CRS, inplace=True)

            connect_s = get_shortest(self.sidewalk, pdfs, f_type='sidewalk_connection')
            all_connections.append(connect_s)

        if len(new_geoms_e) > 0:
            pe = [g[1] for g in new_geoms_e]
            # le = [g[0] for g in new_geoms_e]
            pdfe = gpd.GeoDataFrame(geometry=pe)
            pdfe.set_crs(METRIC_CRS, inplace=True)

            connect_e = get_shortest(self.sidewalk, pdfe, f_type='sidewalk_connection')
            all_connections.append(connect_e)

        if len(new_geoms_both) > 0:
            pb = [g[1] for g in new_geoms_both]
            # lb = [g[0] for g in new_geoms_both]  # crosswalk lines where both ends do not intersect
            pdfb = gpd.GeoDataFrame(geometry=pb)
            pdfb.set_crs(METRIC_CRS, inplace=True)

            connect_b = get_shortest(self.sidewalk, pdfb, f_type='sidewalk_connection')
            all_connections.append(connect_b)

        combined = pd.concat([self.crosswalk, self.sidewalk])
        if all_connections:  # list is not empty
            connect = pd.concat(all_connections)
            combined = pd.concat([combined, connect])
        combined.dropna(inplace=True)
        combined.geometry = combined.geometry.set_crs(METRIC_CRS)
        combined.geometry = combined.geometry.to_crs(4326)
        combined = combined[(~combined.geometry.is_empty) & (combined.geometry.notna())]
        combined.reset_index(drop=True, inplace=True)
        path = self.project.network.path

        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath(
            f'{self.project.name}-Network-{datetime.datetime.now().strftime("%d-%m-%Y_%H")}'
        )
        combined.to_file(path)

        self.complete_net = combined
