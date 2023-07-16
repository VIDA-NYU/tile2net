import math
import os
import numpy as np
# import momepy
import pandas as pd

pd.options.mode.chained_assignment = None
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point, MultiLineString, Polygon
import operator
from tile2net.raster.tile_utils.geodata_utils import geo2geodf
import collections
from functools import reduce
from tile2net.raster.tile_utils.momepy_shapes import *


def morpho_atts(gdf):
    """Using momepy but removed the import since it was slow and depends on pygeos
    Create shape descriptor for the polygon geodataframe.

    """

    gdf['ari'] = gdf.area
    gdf['peri'] = gdf.length
    gdf['ari_peri'] = gdf.area / gdf.length
    gdf['corners'] = Corners(gdf).series
    gdf['elongation'] = Elongation(gdf).series
    gdf['comp'] = CircularCompactness(gdf, 'ari').series
    gdf['squ_comp'] = SquareCompactness(gdf).series
    gdf['rect'] = Rectangularity(gdf, 'ari').series
    gdf['squareness'] = Squareness(gdf).series
    gdf['convexity'] = Convexity(gdf).series
    return gdf


def replace_straight_polys(gdf, convex=0.8, compact=0.2):
    """

    Args:
        gdf: geopandas geodataframe
        convex:: convexity threshold to filter lines
        compact: circular compactness threshold
    Returns:
        geopandas geodataframe
    """
    # find the straight polygons
    straights = gdf[(gdf.convexity > convex) & (gdf.comp < compact)].copy()
    # replace them with the minimum bounding box
    gdf.loc[straights.index, 'geometry'] = [g.minimum_rotated_rectangle for g in straights.geometry]
    return gdf

def draw_middle(geom):
    """
    given a geotangular polygon, draw the  longest centerline

    geo: shapely rectangular polygon geometry
    """
    # get coordinates of the minumum bounding box vertices
    pos = geom.minimum_rotated_rectangle.exterior.coords[:]
    # create the lines
    lin1 = LineString([Point(pos[0]), Point(pos[3])])
    lin2 = LineString([Point(pos[1]), Point(pos[2])])
    lin3 = LineString([Point(pos[0]), Point(pos[1])])
    lin4 = LineString([Point(pos[3]), Point(pos[2])])
    # find the orientation of the geotangle to find the smaller edge
    # find the centroid of the smaller lines and connect them together to form the centerline
    if lin1.length < lin3.length:
        cp1 = lin1.centroid
        cp2 = lin2.centroid
        cntl = LineString([Point(cp1), Point(cp2)])
        return cntl
    else:
        cp3 = lin3.centroid
        cp4 = lin4.centroid
        cntl = LineString([Point(cp3), Point(cp4)])
        return cntl

def get_line_sepoints(line_gdf):
    '''Get the start and end points of lines
    '''
    # main idea from https://github.com/pysal/momepy/blob/a8475f620ee2611eb1c8432ce16cb17160602918/momepy/preprocessing.py
    linegeom = line_gdf.geometry.values
    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(linegeom)
    # how many geometries are there
    indices = shapely.get_num_coordinates(linegeom)
    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = vectorize_points(np.unique(coords[edges], axis=0))
    return points


def get_start_end(line_gdf):
    lcoords = np.apply_along_axis(shapely.get_coordinates, 0, np.array(line_gdf.geometry))
    return lcoords


def vectorize_points(lst):
    return np.apply_along_axis(Point, 1, lst)


def get_shortest(gdf1, gdf2, f_type: str, max_dist=12):
    # gdf1.sindex.query_bulk(gdf2.buffer(25), predicate="intersects")
    nearest_sw_s = gdf1.sindex.nearest(gdf2.geometry, max_distance=max_dist)
    # First subarray of indices contains input geometry indices. (here pdfS)
    # The second subarray of indices contains tree geometry indices. (here sidewalks)

    # pair the start indices
    # (the nearest sidewalk indice, nearest crosswlk points indice)
    indnear = list(zip(nearest_sw_s[0], nearest_sw_s[1]))

    # iterate over the returned indices and draw the shortest line between the two geometries
    new_lines = []
    for k, v in indnear:
        new_lines.append(shapely.shortest_line(gdf1.iloc[v, gdf1.columns.get_loc(gdf1.geometry.name)],
                                               gdf2.iloc[k, gdf2.columns.get_loc(gdf2.geometry.name)]))

    connect = gpd.GeoDataFrame(geometry=new_lines)
    connect.geometry = connect.geometry.set_crs(3857)
    connect['f_type'] = f_type
    return connect


def find_zigzag_lines(ldf):
    lgeom = ldf.geometry.values
    # how many geometries are there
    indices = shapely.get_num_coordinates(lgeom)
    possible_zigzag = np.where(indices > 8)
    return possible_zigzag


def wrinkle_remover(ldf: gpd.GeoDataFrame, trh: float):
    zigzag_ind = find_zigzag_lines(ldf)
    ldf.loc[ldf.index.isin(zigzag_ind[0]), 'geometry'] = \
        ldf.loc[ldf.index.isin(zigzag_ind[0]), 'geometry'].simplify(trh)
    return ldf


def get_extrapolated_line(coords, tolerance, point=False):
    """From Momepy
    Creates a line extrapoled in p1->p2 direction.
    """
    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] <= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    else:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return shapely.LineString([a, b])


#####


def to_cline(geom: shapely.geometry.Polygon, t: float, simpl: float, **attributes):
    """Create a line from a shapely polygon geometry

    geom: the polygon geometry to create line
    t: the interpolation distance of centerline function,
    this is the maximum distance between two consecutive points on the polygon boundary
    simpl: the threshold to simplify the polygon
    attributes:extra attributes to assign to centerline function
    Return: the shapely geometry of the line
    """

    from centerline.geometry import Centerline

    if geom.area / geom.minimum_rotated_rectangle.area >= 0.68:
        cent = get_crosswalk_cnl(geom)
        return cent
    else:
        cent = Centerline(geom.simplify(simpl), t).geometry
        try:
            line = shapely.ops.linemerge(cent)
            return line
        except ValueError:
            return cent


def create_line(p1: shapely.geometry.Point, p2: shapely.geometry.Point):
    """
    Args:
        p1,p2: shapely points
    Returns:
        shapely linestring

    """
    l = LineString((Point(p1), Point(p2)))
    return l


def get_split(ps: list[shapely.geometry.Point], pol):
    """
    Args:
        ps: the list of 3 shapely points forming the right angle
        pol: the shapely polygon
    Returns:
        the split geometry and its splitting line
    """

    l1 = create_line(ps[0], ps[1])
    l2 = create_line(ps[1], ps[2])

    new_l1 = line_to_line(geo2geodf([l1]), geo2geodf([pol]), 40)
    new_l2 = line_to_line(geo2geodf([l2]), geo2geodf([pol]), 40)

    splited1 = shapely.ops.split(pol, new_l1.iloc[0, 0])
    splited2 = shapely.ops.split(pol, new_l2.iloc[0, 0])

    if max([g.area for g in splited1]) >= max([g.area for g in splited2]):
        return splited1, new_l1
    else:
        return splited2, new_l2


def create_split(ps, pol):
    l1 = create_line(ps[0], ps[1])
    l2 = create_line(ps[1], ps[2])

    new_l1 = line_to_line(geo2geodf([l1]), geo2geodf([pol]), 40)
    new_l2 = line_to_line(geo2geodf([l2]), geo2geodf([pol]), 40)

    splited1 = shapely.ops.split(pol, new_l1.iloc[0, 0])
    splited2 = shapely.ops.split(pol, new_l2.iloc[0, 0])

    return splited1, splited2, new_l1, new_l2



def get_angles(vec_1, vec_2):
    """
    return the angle, in degrees, between two vectors
    """

    dot = np.dot(vec_1, vec_2)
    det = np.cross(vec_1, vec_2)
    angle_in_rad = np.arctan2(det, dot)
    return np.degrees(angle_in_rad)


def simplify_by_angle(poly_in, deg_tol=5):
    """Try to remove persistent coordinate points that remain after
    simplify, convex hull, or something, etc. with some trig instead

    poly_in: shapely Polygon
    deg_tol: degree tolerance for comparison between successive vectors
    """
    ext_poly_coords = poly_in.exterior.coords[:]
    vector_rep = np.diff(ext_poly_coords, axis=0)
    num_vectors = len(vector_rep)
    angles_list = []
    for i in range(0, num_vectors):
        angles_list.append(np.abs(get_angles(vector_rep[i], vector_rep[(i + 1) % num_vectors])))

    #   get mask satisfying tolerance
    thresh_vals_by_deg = np.where(np.array(angles_list) > deg_tol)

    new_idx = list(thresh_vals_by_deg[0] + 1)
    new_vertices = [ext_poly_coords[idx] for idx in new_idx]

    return Polygon(new_vertices)


def put_poly_together(poly, deg_tol=5):
    shell = Polygon(poly.exterior.coords)
    holes = [Polygon(ip.coords) for ip in poly.interiors]
    simple_shell = simplify_by_angle(shell, deg_tol)
    simple_holes = [simplify_by_angle(hole, deg_tol) for hole in holes]
    simple_poly = simple_shell.difference(shapely.ops.unary_union(simple_holes))
    return simple_poly
def fill_holes(gs: gpd.GeoSeries, max_area):
    """
    finds holes in the polygons
    Parameters
    ----------
    gs: gpd.GeoSeries
        the GeoSeries of Shapely Polygons to be filled
    max_area: int
        maximum area of holes to be filled

    Returns
    -------
    newgeom: list[shapely.geometry.Polygon]
        list of polygons with holes filled
    """
    newgeom = None
    rings = [i for i in gs["geometry"].interiors]  # List all interior rings
    if len(rings) > 0:  # If there are any rings
        to_fill = [shapely.geometry.Polygon(ring) for ring in rings if
                   shapely.geometry.Polygon(ring).area < max_area]  # List the ones to fill
        if len(to_fill) > 0:  # If there are any to fill
            print("Filling holes in {}".format(gs.name))
            newgeom = reduce(lambda geom1, geom2: geom1.union(geom2),
                             [gs["geometry"]] + to_fill)  # Union the original geometry with all holes
    if newgeom:

        return newgeom
    else:

        return gs["geometry"]

def calculate_bearing(lat1, lng1, lat2, lng2):
    """
    Calculate the compass bearing(s) between pairs of lat-lng points.
    Vectorized function to calculate (initial) bearings between two points'
    coordinates or between arrays of points' coordinates. Expects coordinates
    in decimal degrees. Bearing represents angle in degrees (clockwise)
    between north and the geodesic line from point 1 to point 2.
    Parameters
    ----------
    lat1 : float or numpy.array of float
        first point's latitude coordinate
    lng1 : float or numpy.array of float
        first point's longitude coordinate
    lat2 : float or numpy.array of float
        second point's latitude coordinate
    lng2 : float or numpy.array of float
        second point's longitude coordinate
    Returns
    -------
    bearing : float or numpy.array of float
        the bearing(s) in decimal degrees
    """
    # get the latitudes and the difference in longitudes, in radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    d_lng = np.radians(lng2 - lng1)

    # calculate initial bearing from -180 degrees to +180 degrees
    y = np.sin(d_lng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lng)
    initial_bearing = np.degrees(np.arctan2(y, x))

    # normalize to 0-360 degrees to get compass bearing
    return initial_bearing % 360


def get_longest_line(line):
    if line.type == 'MultiLineString':
        ind_longest = np.argmax(np.array([l.length for l in list(line.geoms)]))
        longest = [lin for lin in line][ind_longest]
        return longest
    else:
        return line


def simplify_poly(poly, simp_trh):
    poly.geometry = poly.simplify(simp_trh)
    return poly


def set_att_linetype(line, geom, cl_arg):
    cw_lin_geom = []
    if line.type == 'LineString':
        extended_line = line_to_line(geo2geodf([line]), geo2geodf([geom.boundary]), 10)
        extended_line.reset_index(inplace=True, drop=True)
        cw_lin_geom.append(extended_line)
    elif line.type == 'MultiLineString':
        if len(list(line.geoms)) <= 3:
            longest = get_longest_line(line)
            extended_line = line_to_line(geo2geodf([longest]), geo2geodf([geom.boundary]), 10)
            extended_line.reset_index(inplace=True, drop=True)
            cw_lin_geom.append(extended_line)
        else:
            line2 = to_cline(geom, cl_arg + 0.5, 1)
            line2 = trim_lines(line2, 2, 2)
            extended_line = line_to_line(geo2geodf([line2]), geo2geodf([geom.boundary]), 10)
            extended_line.reset_index(inplace=True, drop=True)


def extend_longest(line, target, dist, ):
    # fixme check the type of line and target and act accordingly
    longest = get_longest_line(line)
    gdf = geo2geodf([longest])
    gtar = geo2geodf([target])
    extended_line = line_to_line(gdf, gtar, dist)
    extended_line.reset_index(inplace=True, drop=True)

    return extended_line


def trim_lines(line, trh1, trh2):
    # adopted from https://github.com/meliharvey/sidewalkwidths-nyc

    if line.type == 'MultiLineString':
        line_list = []
        for c, linestring in enumerate(line.geoms):

            other_lines = MultiLineString([x for j, x in enumerate(line.geoms) if j != c])

            p0 = Point(linestring.coords[0])
            p1 = Point(linestring.coords[-1])

            is_deadend = False

            if p0.disjoint(other_lines): is_deadend = True
            if p1.disjoint(other_lines): is_deadend = True

            if not is_deadend or linestring.length > trh1:
                line_list.append(linestring)
        return MultiLineString(line_list)

    else:
        if line.length > trh2:
            line = line.simplify(0.5)
            return line
        else:
            return LineString()


def trim_checkempty(line, trim1, trim2):
    line_tr = trim_lines(line, trim1, trim2)
    if line_tr.is_empty:
        line_ = trim_lines(line, trim1 / 2, trim2 / 2)
        if line_.is_empty:
            line__ = trim_lines(line, trim1 / 4, trim2 / 4)
            if not line__.is_empty:
                return line__
            else:
                return line
        else:
            return line_
    else:
        return line_tr


def clean_deadend_dangles(gdf, dang_trh=25, dead_trh=18):
    df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    geom = df.geometry.values

    target = df
    itself = True

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = vectorize_points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    # tree = pygeos.STRtree(geom)
    inp, res = df.sindex.query_bulk(geo2geodf(points).geometry, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    dangles = []
    dangind = []
    deads = []
    deadind = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:

        l_coords = shapely.get_coordinates(geom[line])

        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        first = list(geom.sindex.query(start, predicate="intersects"))
        second = list(geom.sindex.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)

        if (first and not second) or (not first and second):
            deads.append(geom[line])
            deadind.append(line)

        if not first and not second:
            dangles.append(geom[line])
            dangind.append(line)

    dang = gpd.GeoDataFrame(dangind, geometry=dangles)
    dea = gpd.GeoDataFrame(deadind, geometry=deads)
    dang.set_crs(3857, inplace=True)
    dea.set_crs(3857, inplace=True)
    drop1 = dang[dang.length < dang_trh][0]
    drop2 = dea[dea.length < dead_trh][0]
    drop_list = set(list(drop1) + list(drop2))

    gdf.drop(drop_list, inplace=True)

    return gdf



def get_crosswalk_cnl(geom):

    """
    Creates the centerline of crosswalk polygons which are almost rectangular

    Parameters
    ----------
    geom: shapely Polygon
        The shapely Polygon of the crosswalk

    Returns
    ----------
        shapely.LineString
        The centerline of the crosswalk, connecting the centroids of the two shortest edges

    """
    # if geom.area/geom.minimum_rotated_rectangle.area <= 0.4:
    #     geom_er = geom.buffer(-0.85)
    #     if geom_er.type == "MultiPolygon":
    #         mylines = []

    rec = list(geom.minimum_rotated_rectangle.exterior.coords)
    lin1 = LineString([Point(rec[0]), Point(rec[3])])
    lin2 = LineString([Point(rec[1]), Point(rec[2])])
    lin3 = LineString([Point(rec[0]), Point(rec[1])])
    lin4 = LineString([Point(rec[3]), Point(rec[2])])
    if lin1.length < lin3.length:
        clin1 = lin1.centroid
        clin2 = lin2.centroid
        longest = LineString([Point(clin1), Point(clin2)])
        return longest
    else:
        clin3 = lin3.centroid
        clin4 = lin4.centroid
        longest = LineString([Point(clin3), Point(clin4)])
        return longest


def find_medianisland(swp, cwp):
    """
    Args:
        swp: swidewalks polygons
        cwp: crosswalk polygons
    Returns:
        geodataframe with island centroids
    """
    # get pygeos geometry
    sw_pg = swp.geometry.values
    cw_pg = cwp.geometry.values

    inp, res = swp.sindex.query_bulk(cwp, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    meds = np.unique(res[np.isin(inp, unique[counts >= 2])])

    possible_median = swp[swp.index.isin(meds)]

    smallers = possible_median[possible_median.area < 100]
    return smallers


def _right_angle(a, b, c):
    # calculate angle between points, return true or false if Right corner as well as the angle value
    # adopted from momepy
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # TODO: add arg to specify these values
    if (np.degrees(angle) >= 80) & (np.degrees(angle) <= 100):
        return True, np.degrees(angle)
    if (np.degrees(angle) >= 260) & (np.degrees(angle) <= 280):
        return True, np.degrees(angle)
    else:
        return False, np.degrees(angle)


def find_right_(gdf):
    from tqdm.auto import tqdm  # progress bar
    # adopted from momepy
    results_list = []
    angles_dict = {}
    for d, geom in enumerate(tqdm(gdf.geometry, total=gdf.shape[0])):
        angles_dict[d] = {}
        if geom.type == "Polygon":
            angles_dict[d]['right'] = {}
            angles_dict[d]['other'] = {}
            corners = 0  # define empty variables
            points = list(geom.exterior.coords)  # get points of a shape
            stop = len(points) - 1  # define where to stop
            for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
                if i == 0:
                    continue
                elif i == stop:
                    a = np.asarray(points[i - 1])
                    b = np.asarray(points[i])
                    c = np.asarray(points[1])
                    bo, an = _right_angle(a, b, c)

                    if bo:
                        corners = corners + 1
                        angles_dict[d]['right'][i] = {'angle': an, 'ps': (a, b, c)}

                    else:
                        angles_dict[d]['other'][i] = {'angle': an, 'ps': (a, b, c)}
                #                         continue

                else:
                    a = np.asarray(points[i - 1])
                    b = np.asarray(points[i])
                    c = np.asarray(points[i + 1])
                    bo, an = _right_angle(a, b, c)
                    if bo:
                        corners = corners + 1
                        angles_dict[d]['right'][i] = {'angle': an, 'ps': (a, b, c)}
                    else:
                        angles_dict[d]['other'][i] = {'angle': an, 'ps': (a, b, c)}

        elif geom.type == "MultiPolygon":
            corners = 0  # define empty variables
            for g in list(geom.geoms):
                points = list(g.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])
                        bo, an = _right_angle(a, b, c)

                        if bo:
                            corners = corners + 1
                            angles_dict[d]['right'][i] = {'angle': an, 'ps': (a, b, c)}
                        else:
                            angles_dict[d]['other'][i] = {'angle': an, 'ps': (a, b, c)}
                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])
                        bo, an = _right_angle(a, b, c)
                        if bo:
                            corners = corners + 1
                            angles_dict[d]['right'][i] = {'angle': an, 'ps': (a, b, c)}
                        else:
                            angles_dict[d]['other'][i] = {'angle': an, 'ps': (a, b, c)}
            else:
                corners = np.nan

        results_list.append(corners)
    return angles_dict, results_list


"""
MOMEPY LIBRARY WRAPPERS 

"""


def remove_false_nodes(gdf):
    """Wrapper around momepy to remove pygeos dependency.
    Clean topology of existing LineString geometry by removal of nodes of degree 2.
    Returns the original gdf if there's no node of degree 2.
    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of pygeos geometries
        (Multi)LineString data of street network
    Returns
    -------
    gdf : GeoDataFrame, GeoSeries
    See also
    --------
    momepy.extend_lines
    momepy.close_gaps
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # reset index due to the bug in GeoPandas explode
        # if GPD_10:
        #     df = gdf.reset_index(drop=True).explode(ignore_index=True)
        # else:
        df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

        # get underlying pygeos geometry
        geom = df.geometry.values
    else:
        geom = gdf
        df = gpd.GeoSeries(gdf)

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = vectorize_points(np.unique(coords[edges], axis=0))
    # query LineString geometry to identify points intersecting 2 geometries
    inp, res = df.sindex.query_bulk(geo2geodf(points).geometry, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    merge = np.unique(res[np.isin(inp, unique[counts == 2])])

    if len(merge) > 0:
        # filter duplications and create a dictionary with indication of components to
        # be merged together
        dups = [item for item, count in collections.Counter(merge).items() if count > 1]
        split = np.split(merge, len(merge) / 2)
        components = {}
        for i, a in enumerate(split):
            if a[0] in dups or a[1] in dups:
                if a[0] in components:
                    i = components[a[0]]
                elif a[1] in components:
                    i = components[a[1]]
            components[a[0]] = i
            components[a[1]] = i

        # iterate through components and create new geometries
        new = []
        for c in set(components.values()):
            keys = []
            for item in components.items():
                if item[1] == c:
                    keys.append(item[0])
            new.append(shapely.line_merge(shapely.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        #  if GPD_10:
        #      final = gpd.GeoSeries(new).explode(ignore_index=True)
        # else:
        final = gpd.GeoSeries(new).explode().reset_index(drop=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            return pd.concat(
                [
                    df,
                    gpd.GeoDataFrame(
                        {df.geometry.name: final}, geometry=df.geometry.name
                    ),
                ],
                ignore_index=True,
            )
        return pd.concat([df, final], ignore_index=True)

    # if there's nothing to fix, return the original dataframe
    return gdf


def extend_lines(gdf, tolerance, target=None, barrier=None, extension=0):
    """Extends lines from gdf to itself or target within a set tolerance
    Extends unjoined ends of LineString segments to join with other segments or
    target. If ``target`` is passed, extend lines to target. Otherwise extend
    lines to itself.
    If ``barrier`` is passed, each extended line is checked for intersection
    with ``barrier``. If they intersect, extended line is not returned. This
    can be useful if you don't want to extend street network segments through
    buildings.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing LineString geometry
    tolerance : float
        tolerance in snapping (by how much could be each segment
        extended).
    target : GeoDataFrame, GeoSeries
        target geometry to which ``gdf`` gets extended. Has to be
        (Multi)LineString geometry.
    barrier : GeoDataFrame, GeoSeries
        extended line is not used if it intersects barrier
    extension : float
        by how much to extend line beyond the snapped geometry. Useful
        when creating enclosures to avoid floating point imprecision.
    Returns
    -------
    GeoDataFrame
        GeoDataFrame of with extended geometry
    See also
    --------
    momepy.close_gaps
    momepy.remove_false_nodes
    """
    # explode to avoid MultiLineStrings
    # reset index due to the bug in GeoPandas explode
    # if GPD_10:
    #     df = gdf.reset_index(drop=True).explode(ignore_index=True)
    # else:
    df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    if target is None:
        target = df
        itself = True
    else:
        itself = False

    # get underlying pygeos geometry
    geom = df.geometry.values

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = vectorize_points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    inp, res = df.sindex.query_bulk(geo2geodf(points).geometry, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:
        l_coords = shapely.get_coordinates(geom[line])

        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        first = list(geom.sindex.query(start, predicate="intersects"))
        second = list(geom.sindex.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)
        if first and not second:
            snapped = _extend_line(l_coords, t, tolerance)
            if (
                    barrier is not None
                    and barrier.sindex.query(
                shapely.linestrings(snapped), predicate="intersects"
            ).size
                    > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and second:
            snapped = _extend_line(np.flip(l_coords, axis=0), t, tolerance)
            if (
                    barrier is not None
                    and barrier.sindex.query(
                shapely.linestrings(snapped), predicate="intersects"
            ).size
                    > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and not second:
            one_side = _extend_line(l_coords, t, tolerance)
            one_side_e = _extend_line(one_side, t, extension, snap=False)
            snapped = _extend_line(np.flip(one_side_e, axis=0), t, tolerance)
            if (
                    barrier is not None
                    and barrier.sindex.query(
                shapely.linestrings(snapped), predicate="intersects"
            ).size
                    > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )

    df.iloc[ends, df.columns.get_loc(df.geometry.name)] = new_geoms
    return df


def _extend_line(coords, target, tolerance, snap=True):
    """
    Extends a line geometry to snap on the target within a tolerance.
    """
    if snap:
        extrapolation = _get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
            tolerance,
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = shapely.intersection(
            target.iloc[int_idx].geometry.values, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = shapely.distance(p, Point(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = shapely.get_coordinates(intersection[minimal])

            else:
                new_point_coords = shapely.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (int(len(coo) / 2), 2))

            return new
        return coords

    extrapolation = _get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def _get_extrapolated_line(coords, tolerance, point=False):
    """
    Creates a pygeos line extrapoled in p1->p2 direction.
    """
    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] <= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    else:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return shapely.LineString([a, b])


def extend_line(coords, target, tolerance, snap=True):
    """warper around momepy to remove Pygeos dependancy
    Extends a line geometry to snap on the target within a tolerance.
    """
    if snap:
        extrapolation = get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(), tolerance
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = shapely.intersection(
            target.iloc[int_idx].geometry.values, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = shapely.distance(p, shapely.points(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = shapely.get_coordinates(intersection[minimal])

            else:
                new_point_coords = shapely.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (int(len(coo) / 2), 2))

            return new
        return coords

    extrapolation = get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def line_to_line(gdf, target, tolerance, extension):
    """ From Momepy
    Extends lines from gdf to target within a set tolerance


    """
    # explode to avoid MultiLineStrings
    # double reset index due to the bug in GeoPandas explode
    df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    # get underlying geometry
    geom = df.geometry.values
    if target is None:
        target = df
        itself = True
    else:
        itself = False

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = vectorize_points(np.unique(coords[edges], axis=0))
    # query LineString geometry to identify points intersecting 2 geometries
    # tree = pygeos.STRtree(geom)
    inp, res = df.sindex.query_bulk(geo2geodf(points).geometry, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:

        l_coords = shapely.get_coordinates(geom[line])

        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        first = list(geom.sindex.query(start, predicate="intersects"))
        second = list(geom.sindex.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)

        if first and not second:
            snapped = extend_line(l_coords, target, tolerance)
            new_geoms.append(
                shapely.LineString(extend_line(snapped, target, 0.00001, snap=False))
            )
            if extension == 0:
                new_geoms.append(shapely.LineString(snapped))
            else:
                new_geoms.append(
                    shapely.LineString(
                        extend_line(snapped, t, extension, snap=False)
                    )
                )
            new_geoms.append(
                shapely.LineString(extend_line(snapped, target, 0.00001, snap=False))
            )
        elif not first and second:
            snapped = extend_line(l_coords,
                                  # np.flip(l_coords,
                                  #         axis=0),
                                  target, tolerance)
            new_geoms.append(
                shapely.LineString(extend_line(snapped, target, 0.00001, snap=False))
            )
            if extension == 0:
                new_geoms.append(shapely.LineString(snapped))
            else:
                new_geoms.append(
                    shapely.LineString(
                        extend_line(snapped, t, extension, snap=False)
                    )
                )
            new_geoms.append(
                shapely.LineString(extend_line(snapped, target, 0.00001, snap=False))
            )
        elif not first and not second:
            one_side = extend_line(l_coords, target, tolerance)
            one_side_e = extend_line(one_side, target, 0.00001, snap=False)
            snapped = extend_line(one_side_e, target, tolerance)
            if extension == 0:
                new_geoms.append(shapely.LineString(snapped))
            else:
                new_geoms.append(
                    shapely.LineString(
                        extend_line(snapped, t, extension, snap=False)
                    )
                )
            new_geoms.append(
                shapely.LineString(extend_line(snapped, target, 0.00001, snap=False))
            )

    df = df.drop(ends)
    final = gpd.GeoSeries(new_geoms).explode().reset_index(drop=True)
    return df.append(
        gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
        ignore_index=True,
    )
