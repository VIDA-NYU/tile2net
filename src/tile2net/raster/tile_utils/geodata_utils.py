import os
from tile2net.logger import logger
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import mapping, shape
import skimage
from affine import Affine


def read_gdf(path):
    """
    Read a GeoDataFrame from a file
    Parameters
    ----------
    path: str
        path to the file

    Returns
    -------
    gdf: GeoDataFrame
    """
    gdf = gpd.read_file(path)
    return gdf


def set_gdf_crs(gdf, crs):
    """
    Set the CRS of a GeoDataFrame
    Parameters
    ----------
    gdf: GeoDataFrame
    crs: int
        coordinate reference system

    Returns
    -------
    gdf: GeoDataFrame
    """
    gdf.geometry = gdf.geometry.set_crs(crs)
    return gdf


def change_crs(gdf, crs):
    """
    Change the CRS of a GeoDataFrame
    Parameters
    ----------
    gdf: GeoDataFrame
    crs: int

    Returns
    -------
    gdf: GeoDataFrame
    """
    gdf.geometry = gdf.geometry.to_crs(crs)
    return gdf


def prepare_spindex(gdf: gpd.GeoDataFrame):
    """
    Prepare a GeoDataFrame for spatial indexing
    Parameters
    ----------
    gdf: GeoDataFrame

    Returns
    -------
    spatial index of a GeoDataFrame
    """
    return gdf.sindex


def _reduce_geom_precision(geom, precision=2):
    """
    Reduce the precision of a geometry to a given number of decimal places.
    Parameters
    ----------
    geom: shapely.geometry
    precision: int
        number of decimal places to round to

    Returns
    -------
    geom: shapely.geometry
    """
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(
        np.array(geojson['coordinates']),
        precision
    )
    return shape(geojson)


def affine_to_list(affine_obj):
    """Convert a :class:`affine.Affine` instance to a list for Shapely."""
    return [affine_obj.a, affine_obj.b,
            affine_obj.d, affine_obj.e,
            affine_obj.xoff, affine_obj.yoff]


def list_to_affine(xform_mat):
    """Create an Affine from a list or array-formatted [a, b, d, e, xoff, yoff]

    Arguments
    ---------
    xform_mat : `list` or :class:`numpy.array`
        A `list` of values to convert to an affine object.

    Returns
    -------
    aff : :class:`affine.Affine`
        An affine transformation object.
    """
    # first make sure it's not in gdal order
    if len(xform_mat) > 6:
        xform_mat = xform_mat[0:6]
    if rasterio.transform.tastes_like_gdal(xform_mat):
        return Affine.from_gdal(*xform_mat)
    else:
        return Affine(*xform_mat)


def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for rasterio.".format(im)
        )


def _check_skimage_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return skimage.io.imread(im)
    elif isinstance(im, np.ndarray):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for scikit-image.".format(im)
        )


def prepare_class_gdf(polys, class_name) -> object:
    """
    separates the polygons of each class, given the keyboard (sidewalk, crosswalk, road)
    Args:
        polys (geodataframe): the dataframe containing the polygons of all classes
        class_name(str): the class label, sidewalk, crosswalk, road

    Returns:
        class specific GeoDataFrame in metric projection
    """

    nt = polys[polys.f_type == f'{class_name}'].copy()
    nt.geometry = nt.geometry.to_crs(3857)
    return nt


def prepare_gdf(gdf, **cols):
    """
    Filter a GeoDataFrame based on a set of columns and values
    Parameters
    ----------
    gdf: GeoDataFrame
    cols: dict
        {column_name: value}

    Returns
    -------
    f_gdf: GeoDataFrame
    """
    # TODO: Add other operations like !
    k = list(cols.keys())[0]
    print(f'k, {k}', f'cols {len(cols[k])}')
    if isinstance(cols[k], list):
        f_gdf = gdf[gdf[k].isin(cols[k])]
    else:
        f_gdf = gdf[gdf[k] == cols[k]]
    return f_gdf


def read_dataframe(src_path, geo=True, cols=None):
    """
    Args:
        src_path:
        geo: if True, will create GeoDataFrame
        cols: optional. Name of specific columns to be read
    Returns:
    """
    if geo:
        if cols:
            df = gpd.read_file(src_path, usecols=cols)
        else:
            df = gpd.read_file(src_path)
    else:
        if cols:
            df = pd.read_csv(src_path, usecols=cols)
        else:
            df = pd.DataFrame(src_path)
    return df


# def unary_multi(gdf):
# 	"""
# 	handles the errors with multipolygon
# 	"""
# 	if gdf.unary_union.type == 'MultiPolygon':
# 		gdf_uni = gpd.GeoDataFrame(geometry=gpd.GeoSeries([geom for geom in gdf.unary_union.geoms]))
# 	else:
# 		gdf_uni = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf.unary_union))
# 	return gdf_uni

def unary_multi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # handles the errors with multipolygon
    loc = ~gdf.is_valid.values
    logger.warning(f'Number of invalid geometries: {loc.sum()} out of {len(gdf)}')
    gdf.geometry.loc[loc] = shapely.make_valid(gdf.geometry.loc[loc])
    if isinstance(gdf, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=gdf)
    result = (
        gdf
        # dissolve overlapping geometries
        .dissolve()
        # explode multipart geometries
        .explode()
    )
    return result


def buffer_union(gdf, buff, simp1, simp2):
    """
    buffer and union the polygons in a GeoDataFrame
    Parameters
    ----------
    gdf: GeoDataFrame
    buff: float
        buffer distance
    simp1: float
        simplification tolerance for the buffer
    simp2: float
        simplification tolerance for the union

    Returns
    -------
    gdf_uni: GeoDataFrame
    """
    gdf.geometry = gdf.geometry.buffer(buff, join_style=2, cap_style=3)
    gdf.geometry = gdf.simplify(simp1)
    gdf_uni = unary_multi(gdf)
    gdf_uni.geometry = gdf_uni.geometry.set_crs(3857)
    gdf_uni.geometry = gdf_uni.geometry.simplify(simp2)
    return gdf_uni


def buffer_union_erode(gdf, buff, erode, simp1, simp2, simp3):
    gdf_buff = buffer_union(gdf, buff, simp1, simp2)
    gdf_erode = gdf_buff.geometry.buffer(erode, join_style=2, cap_style=3)
    gdf_uni = unary_multi(gdf_erode)
    gdf_uni.geometry = gdf_uni.geometry.set_crs(3857)
    gdf_uni.geometry = gdf_uni.geometry.simplify(simp3)
    return gdf_uni


def to_metric(gdf, crs=3857):
    """Converts a GeoDataFrame to metric (3857) coordinate
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame of polygons
    crs : int, optional
        the coordinate system to convert to, by default 3857
    Returns
    -------
    GeoDataFrame
        GeoDataFrame of polygons in metric coordinate system
    """
    gdf.to_crs(crs, inplace=True)
    return gdf


def geo2geodf(geo_lst):
    """
    Converts a list of shapely geometries to a GeoDataFrame
    Parameters
    ----------
    geo_lst: list

    Returns
    -------
    gdf: GeoDataFrame

    """
    gdf = gpd.GeoDataFrame(geometry=geo_lst)
    return gdf


def merge_dfs(gdf1, gdf2, crs=4326):
    """
    merges two dataframes with the results of segmentation (three classes)
    Parameters
    ----------
    gdf1: GeoDataFrame
    gdf2: GeoDataFrame
    crs: int

    Returns
    -------

    """
    if gdf1.crs != gdf2.crs:
        gdf1.to_crs(crs, inplace=True)
        gdf2.to_crs(crs, inplace=True)

    df1sw = prepare_class_gdf(gdf1, 'sidewalk')
    df1cw = prepare_class_gdf(gdf1, 'crosswalk')
    df1rd = prepare_class_gdf(gdf1, 'road')

    df2sw = prepare_class_gdf(gdf2, 'sidewalk')
    df2cw = prepare_class_gdf(gdf2, 'crosswalk')
    df2rd = prepare_class_gdf(gdf2, 'road')

    concsw = pd.concat([df1sw, df2sw])
    conccw = pd.concat([df1cw, df2cw])
    concrd = pd.concat([df1rd, df2rd])

    unionsw = unary_multi(concsw)
    unionsw = unionsw.explode().reset_index(drop=True)
    unionsw.geometry = unionsw.geometry.set_crs(crs)
    unionsw['f_type'] = 'sidewalk'

    unioncw = unary_multi(conccw)

    unioncw.geometry = unioncw.geometry.set_crs(crs)
    unioncw['f_type'] = 'crosswalk'

    unionrd = unary_multi(concrd)

    unionrd.geometry = unionrd.geometry.set_crs(crs)
    unionrd['f_type'] = 'road'

    merged = pd.concat([unionrd, unionsw, unioncw])
    merged.geometry = merged.geometry.set_crs(crs)

    return merged


def create_stats(gdf):
    """

    Parameters
    ----------
    gdf: GeoDataFrame

    Returns
    -------

    """
    cgdf = gdf.copy()
    cgdf['primeter'] = cgdf.length
    cgdf['area'] = cgdf.area
    cgdf['ar_pratio'] = cgdf.area / cgdf.length
    # get the summary statics of the polygons
    ss = cgdf.quantile([0.25, 0.5, 0.75])
    return ss, cgdf


def buff_dfs(gdf):
    """
    union and buffer the polygons of each class separately,
    to create continuous polygons and merge them into one GeoDataFrame.

    Parameters
    ----------
    gdf: GeoDataFrame
        Polygon dataframes with three classes in metric coordinate system
    crs: int

    Returns
    -------
    GeoDataFrame:
        merged GeoDataFrame of the three classes
    """

    gdf.geometry = gdf.simplify(0.2)
    dfsw = prepare_class_gdf(gdf, 'sidewalk')
    dfcw = prepare_class_gdf(gdf, 'crosswalk')
    dfrd = prepare_class_gdf(gdf, 'road')

    buffersw = buffer_union_erode(dfsw, 0.3, -0.3, 0.2, 0.3, 0.3)

    buffersw['f_type'] = 'sidewalk'

    buffercw = buffer_union_erode(dfcw, 0.3, -0.25, 0.2, 0.3, 0.3)

    buffercw['f_type'] = 'crosswalk'

    bufferrd = buffer_union_erode(dfrd, 0.4, -0.4, 0.2, 0.3, 0.3)
    bufferrd['f_type'] = 'road'

    merged = pd.concat([buffercw, buffersw, bufferrd])
    merged.geometry = merged.geometry.set_crs(gdf.crs)

    return merged
