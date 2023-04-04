import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import mapping, shape
import skimage
from affine import Affine


def read_gdf(path):
	gdf = gpd.read_file(path)
	return gdf


def set_gdf_crs(gdf, crs):
	gdf.geometry = gdf.geometry.set_crs(crs)
	return gdf


def change_crs(gdf, crs):
	gdf.geometry = gdf.geometry.to_crs(crs)
	return gdf


def prepare_spindex(gdf: gpd.GeoDataFrame):
	return gdf.sindex


def _reduce_geom_precision(geom, precision=2):
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


def geo2geodf(geo_lst):
	gdf = gpd.GeoDataFrame(geometry=geo_lst)
	return gdf


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
	print('converting to metric')
	nt.geometry = nt.geometry.to_crs(3857)
	return nt


def prepare_gdf(gdf, **cols):
	"""filters a GeoDataFrame based on the passed
	columns
	Args:
		gdf (GeoDataFrame): the dataframe to filter
		**cols (list):
	Returns (GeoDataFrame):
	"""
	# TODO: Add other operations like !
	k = list(cols.keys())[0]
	print(f'k, {k}', f'cols {len(cols[k])}')
	if isinstance(cols[k], list):
		f_gdf = gdf[gdf[k].isin(cols[k])]
	else:
		f_gdf = gdf[gdf[k] == cols[k]]
	return f_gdf


def prepare_spaindex(df):
	spatial_index = df.sindex
	return spatial_index


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


def mycompareplot(gdf1, gdf2, size=(200, 200), *args):
	ax = gdf1.plot(alpha=.5, color='b', figsize=(200, 200))
	gdf2.plot(ax=ax, color='r', alpha=.5, aspect=1)
	ax.set_axis_off()


def createfolder(directory):  # checks and creates folder if not exists
	if not os.path.exists(directory):
		os.makedirs(directory)
		return directory
	else:
		return directory


def unary_multi(gdf):
	"""
	handles the errors with multipolygon
	"""
	if gdf.unary_union.type == 'MultiPolygon':
		gdf_uni = gpd.GeoDataFrame(geometry=gpd.GeoSeries([geom for geom in gdf.unary_union.geoms]))
	else:
		gdf_uni = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf.unary_union))
	return gdf_uni


def buffer_union(gdf, buff, simp1, simp2):
	gdf.geometry = gdf.geometry.buffer(buff, join_style=2, cap_style=3)
	gdf.geometry = gdf.simplify(simp1)
	gdf_uni = unary_multi(gdf)
	gdf_uni.geometry = gdf_uni.geometry.set_crs(3857)
	gdf_uni.geometry = gdf_uni.geometry.simplify(simp2)
	return gdf_uni


def buffer_union_erode(gdf, buff, erode, simp1, simp2, simp3):
	gdf_buff = buffer_union(gdf, buff, simp1, simp2)
	gdf_erode = gdf_buff.geometry.buffer(erode, join_style=2, cap_style=3)
	gdf_uni = unary_multi(gdf)
	gdf_uni.geometry = gdf_uni.geometry.set_crs(3857)
	gdf_uni.geometry = gdf_uni.geometry.simplify(simp3)
	return gdf_uni


def geo2geodf(geo_lst):
	gdf = gpd.GeoDataFrame(geometry=geo_lst)
	return gdf


def merge_dfs(df1, df2):
	df1sw = df1[df1.f_type == 'sidewalk']
	df1cw = df1[df1.f_type == 'crosswalk']
	df1rd = df1[df1.f_type == 'road']

	df2sw = df2[df2.f_type == 'sidewalk']
	df2cw = df2[df2.f_type == 'crosswalk']
	df2rd = df2[df2.f_type == 'road']

	concsw = pd.concat([df1sw, df2sw])
	conccw = pd.concat([df1cw, df2cw])
	concrd = pd.concat([df1rd, df2rd])

	unionsw = unary_multi(concsw)
	unionsw = unionsw.explode().reset_index(drop=True)
	unionsw.geometry = unionsw.geometry.set_crs(4326)
	unionsw['f_type'] = 'sidewalk'

	unioncw = unary_multi(conccw)

	unioncw.geometry = unioncw.geometry.set_crs(4326)
	unioncw['f_type'] = 'crosswalk'

	unionrd = unary_multi(concrd)

	unionrd.geometry = unionrd.geometry.set_crs(4326)
	unionrd['f_type'] = 'road'

	merged = pd.concat([unionrd, unionsw, unioncw])
	merged.geometry = merged.geometry.set_crs(4326)

	return merged


def create_stats(gdf):
	# to line
	cgdf = gdf.copy()
	cgdf['primeter'] = cgdf.length
	cgdf['area'] = cgdf.area
	cgdf['ar_pratio'] = cgdf.area / cgdf.length
	# get the summary statics of the polygons
	ss = cgdf.quantile([0.25, 0.5, 0.75])
	return ss, cgdf


def buff_dfs(df, crs):
	df.geometry = df.geometry.to_crs(3857)
	df.geometry = df.simplify(0.2)
	dfsw = df[df.f_type == 'sidewalk']
	dfcw = df[df.f_type == 'crosswalk']
	dfrd = df[df.f_type == 'road']

	buffersw = buffer_union_erode(dfsw, 0.2, -0.2, 0.1, 0.2, 0.2)

	buffersw['f_type'] = 'sidewalk'

	buffercw = buffer_union_erode(dfcw, 0.3, -0.25, 0.1, 0.2, 0.2)

	#     unioncw.geometry = unioncw.geometry.set_crs(4326)
	buffercw['f_type'] = 'crosswalk'

	bufferrd = buffer_union_erode(dfrd, 0.2, -0.2, 0.1, 0.2, 0.2)
	#     unionrd.geometry = unionrd.geometry.set_crs(4326)
	bufferrd['f_type'] = 'road'

	merged = pd.concat([buffercw, buffersw, bufferrd])
	merged.geometry = merged.geometry.set_crs(3857)
	merged.geometry = merged.geometry.to_crs(crs)

	return merged
