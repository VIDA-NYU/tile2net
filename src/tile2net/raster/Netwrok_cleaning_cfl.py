### Imports
import time

import fiona                    #could reduce imports to load layer?
import pandas as pd             #could reduce imports to concat?
import geopandas as gpd         #could reduce import to GeodataFrane??
import shapely.geometry as geo  # could reduce import to specifically needed geometry types?
from shapely.ops import split
from madina import Zonal        # could reduce import to save map and coloring functions?
from math import degrees, atan2
import numpy as np

save_map = Zonal.save_map
default_geographic_crs = Zonal.default_geographic_crs




# central function to collect all preprocessing, cleaning and formatting to fix issues comin from third party softwa
def _prepare_geometry(geometry_gdf):
    geometry_gdf = geometry_gdf.copy(deep=True)

    # making sure if geometry contains polygons, they are corrected by using the polygon exterior as a line.
    polygon_idxs = geometry_gdf[geometry_gdf["geometry"].geom_type ==
                                "Polygon"].index
    geometry_gdf.loc[polygon_idxs,
    "geometry"] = geometry_gdf.loc[polygon_idxs, "geometry"].exterior


    # if geometries are all multilineString, convert to lineString
    if (geometry_gdf["geometry"].geom_type ==  'MultiLineString').all():
            #and (np.array(list(map(len, geometry_gdf["geometry"].values))) == 1).all():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(lambda x: list(x)[0])

    # making sure if a few geometries contains MultiLineString, they are corrected by using the polygon exterior as a line.
    # TODO : Handle a case where some geometry is Multipolyline, but contain multiple line strings, not just one.
    multipolyline_gdf = geometry_gdf[geometry_gdf["geometry"].geom_type =="MultiLineString"]
    multilinestring_idxs = (multipolyline_gdf.index) #and (np.array(list(map(len, multipolyline_gdf["geometry"].values))) == 1).all()
    geometry_gdf.loc[multilinestring_idxs,"geometry"] = geometry_gdf.loc[multilinestring_idxs, "geometry"].apply(lambda x: x.geoms[0])


    # deleting any Z coordinate if they exist
    from shapely.ops import transform

    def _to_2d(x, y, z):
        return tuple(filter(None, [x, y]))
    if geometry_gdf.has_z.any():
        geometry_gdf["geometry"] = geometry_gdf["geometry"].apply(
            lambda s: transform(_to_2d, s))
    return geometry_gdf

# Static function, wrapper to the numerical and spatial static network node-edge constructors
def _node_edge_builder(geometry_gdf, weight_attribute=None, tolerance=0.0):
    if tolerance == 0:
        # use vectorized implementaytion
        point_xy = GeoPandaExtractor(geometry_gdf.geometry.values.data)
        node_indexer, node_points, node_dgree, edge_start_node, edge_end_node = _vectorized_node_edge_builder(
            point_xy)
    elif tolerance > 0:
        # use spatial index implementation
        # could this gain effeciency by being sorted?
        point_geometries = geometry_gdf["geometry"].boundary.explode(
            index_parts=False).reset_index(drop=True)
        matching = point_geometries.sindex.query_bulk(
            point_geometries.buffer(tolerance))
    else:
        raise ValueError(f"tplerance must be either zero or a positive value")

    # construct node and edge gdfs
    edge_count = geometry_gdf.shape[0]
    index = pd.Index(np.arange(edge_count), name="id")
    length = pd.Series(
        geometry_gdf["geometry"].length.values, fastpath=True, index=index)

    edge_gdf = gpd.GeoDataFrame(
        {
            "length":   length,
            "weight":   length if weight_attribute is None else pd.Series(geometry_gdf[weight_attribute].apply(lambda x: max(1, x)), fastpath=True, index=index),
            "type":     pd.Series(np.repeat(np.array(["street"], dtype=object), repeats=edge_count), fastpath=True, index=index, dtype="category"),
            "parent_street_id": pd.Series(geometry_gdf.index.values, fastpath=True, index=index),
            "start":    pd.Series(edge_start_node, fastpath=True, index=index),
            "end":      pd.Series(edge_end_node, fastpath=True, index=index),
        },
        index=index,
        geometry=geometry_gdf["geometry"].values.data,
        crs=geometry_gdf.crs
    )

    node_count = node_points.shape[0]
    index = pd.Index(np.arange(node_count), name="id")
    node_gdf = gpd.GeoDataFrame(
        {
            "source_layer": pd.Series(np.repeat(np.array(["streets"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "source_id":    pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
            "type":         pd.Series(np.repeat(np.array(["street_node"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "weight":       pd.Series(np.repeat(np.array([0.0], dtype=np.float32), repeats=node_count), fastpath=True, index=index, dtype=np.float32),
            "nearest_street_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
            "nearest_street_node_distance": pd.Series(np.repeat(np.array([{}], dtype=object), repeats=node_count), fastpath=True, index=index),
            "degree":       pd.Series(node_dgree, fastpath=True, index=index),
            # "connected_edges": connected_edges
        },
        index=index,
        # crs=geometry_gdf.crs,
        geometry=gpd.points_from_xy(
            x=point_xy[0][node_indexer], y=point_xy[1][node_indexer], crs=geometry_gdf.crs)
    )

    ### GOOD PLACE TO CONSTRUCT ADJACENCY LIST>>>

    return node_gdf, edge_gdf # Return Adjacency List here...

# Static function for numeric network node-edge construction

# TODO: eliminate functions not compatible with nb.jit
def _vectorized_node_edge_builder(point_xy):
    # the point_xy input is a 2 by n matrix. where the start point and end point of an edge are in alternating order..
    # unique parameter is expected to come from a spatial index, on points that are sorted.

    # input setup
    point_count = point_xy.shape[1]
    edge_count = int(point_count/2)

    point_types = np.repeat(
        np.array([0.0, 1.0], dtype=np.int32), repeats=edge_count)
    edge_ids = np.arange(edge_count, dtype=np.int32)
    point_segment_ids = np.array(
        [edge_ids, edge_ids]).reshape([point_count, 1])

    # sorting points by x then by y. This is to make it effecient to link the similarly sorted unique nodes with each point occurance
    # sort by x then by y ## np.lexsort is not supported with njit
    sorter = np.lexsort((point_xy[1, :], point_xy[0, :]))
    sorted_point_xy = point_xy[:, sorter]
    point_xy_T = np.transpose(sorted_point_xy)

    # Resorting inputs:
    sorted_point_types = point_types[sorter]
    sorted_point_segment_ids = point_segment_ids[sorter]

    # finding unique points (which are from now on, refereed to as nodes.)
    # indices # np.unique is only supported in njit if no arguments are given.
    unique_nodes = np.unique(
        point_xy_T, axis=0, return_index=True, return_counts=True)

    # node data
    node_points = unique_nodes[0]
    node_first_occurance = unique_nodes[1]
    node_indexer = sorter[node_first_occurance]

    node_dgree = np.array(unique_nodes[2], dtype=np.int32)
    # connected_edges = np.empty(node_count, dtype=np.array)
    # edge data

    edge_start_node = np.zeros(edge_count, dtype=np.int32)
    edge_end_node = np.zeros(edge_count, dtype=np.int32)

    # this is to make sure if the last point occurs for the first time, we'll have a proper
    if node_first_occurance[-1] != (point_count-1):
        np.append(node_first_occurance, point_count-1)

    range_start = 0
    i = 0
    node_index = 0
    for range_end in node_first_occurance[1:]:
        # connected_edges[node_index] = sorted_point_segment_ids[range_start:range_end]
        for point in point_xy_T[range_start:range_end, :]:
            edge_index = sorted_point_segment_ids[i]
            if sorted_point_types[i] == 0:
                edge_start_node[edge_index] = node_index
            else:
                edge_end_node[edge_index] = node_index
            i += 1
        range_start = range_end
        node_index += 1

    return node_indexer, node_points, node_dgree, edge_start_node, edge_end_node

## This works when pyGEOS is not available.
import numpy as np
def GeoPandaExtractor(poly_line_data):
    firsts = list(map(lambda x: x.coords[0], poly_line_data))
    lasts =  list(map(lambda x: x.coords[-1], poly_line_data))
    points = np.array([np.array([point[0], point[1]]) for point in firsts + lasts])
    return points.transpose()

# could be improved and timed much better, but works okay and effeciently enough for now
def _effecient_network_nodes_edges(geometry_gdf, weight_attribute=None, tolerance=1.0):
    geometry_gdf["weight"] = geometry_gdf["geometry"].length if weight_attribute is None else geometry_gdf[weight_attribute].apply(
        lambda x: max(1, x))
    geometry_gdf = geometry_gdf.sort_values("weight", ascending=False)

    #geometry_gdf = geometry_gdf[geometry_gdf["weight"] >= tolerance].reset_index(drop=True)
    point_geometries = pd.concat(
        [
            geometry_gdf["geometry"].apply(lambda x: geo.Point(x.coords[0])),
            geometry_gdf["geometry"].apply(lambda x: geo.Point(x.coords[-1]))
        ]
    ).reset_index(drop=True)

    point_types = ["start"] *  geometry_gdf.shape[0] + ["end"] * geometry_gdf.shape[0]
    point_segment_ids = list(geometry_gdf.index) + list(geometry_gdf.index)

    point_count = geometry_gdf.shape[0] * 2
    geometry_collector = np.empty(point_count, dtype=object)
    degree_collector = np.empty(point_count, dtype="int64")

    edge_count = geometry_gdf.shape[0]

    edge_collector = {
        "start": np.zeros(edge_count, dtype="int64") - 1,
        "end": np.zeros(edge_count, dtype="int64") - 1
    }

    matching = point_geometries.sindex.query_bulk(
        point_geometries.buffer(tolerance), predicate="intersects")
    # grouping points..
    unique_points, intersection_count = np.unique(
        matching[0], return_counts=True)


    point_intersection = {}
    current_index = 0
    for point, count in zip(unique_points, intersection_count):
        point_intersection[point] = set(
            matching[1][current_index:current_index+count]) - {point}
        current_index = current_index+count


    new_node_id = -1
    current_point_idx = -1
    for position in ["start", "end"]:
        for edge_idx in geometry_gdf.index:
            current_point_idx += 1
            if edge_collector[position][edge_idx] != -1:
                continue
            current_intersecting_points = point_intersection[current_point_idx]
            degree = 1
            new_node_id += 1
            edge_collector[position][edge_idx] = new_node_id
            for intersecting_point in current_intersecting_points:
                intersecting_point_segment_idx = point_segment_ids[intersecting_point]
                intersecting_point_type = point_types[intersecting_point]
                if edge_collector[intersecting_point_type][intersecting_point_segment_idx] != -1: ## already filled bu another edge..
                    continue
                degree += 1
                edge_collector[intersecting_point_type][intersecting_point_segment_idx] = new_node_id

            degree_collector[new_node_id] = degree
            geometry_collector[new_node_id] = point_geometries[current_point_idx]

    node_gdf = gpd.GeoDataFrame(
        {
            "id":           pd.Series(range(new_node_id), dtype="int64"),
            "source_layer": pd.Series(["streets"] * new_node_id, dtype="category"),
            "source_id":    pd.Series([0] * new_node_id, dtype="int64"),
            "type":         pd.Series(["street_node"] * new_node_id, dtype="category"),
            "weight":       pd.Series([0.0] * new_node_id, dtype="float64"),
            "nearest_street_id": pd.Series([0] * new_node_id, dtype="int64"),
            "nearest_street_node_distance": pd.Series([None] * new_node_id, dtype="object"),
            "degree":       pd.Series(degree_collector[:new_node_id+1], dtype="int64"),
            "geometry":     gpd.GeoSeries(geometry_collector[:new_node_id+1], crs=geometry_gdf.crs)

        }
    ).set_index("id")
    # TODO: edge geometry does not reflect node snapping...  any line that's assigned a node, the intersction centroid should replace the snapped point..
    edge_gdf = gpd.GeoDataFrame(
        {
            "id":       pd.Series(range(edge_count), dtype="int64"),
            "geometry": geometry_gdf["geometry"],
            "length":   geometry_gdf["geometry"].length,
            "weight":   geometry_gdf["weight"],
            "type":     pd.Series(["street"] * edge_count, dtype="category"),
            "parent_street_id": pd.Series(list(geometry_gdf.index), dtype="int64"),
            "start":    pd.Series(edge_collector["start"], dtype="int64"),
            "end":      pd.Series(edge_collector["end"], dtype="int64"),
        }, crs=geometry_gdf.crs
    ).set_index("id")

    return node_gdf, edge_gdf

# maybe just include as an inner function in fuse degree 2 nodes...
def angle_deviation_between_two_lines(point_sequence, raw_angle=False):
    a = point_sequence[0].coords[0]
    b = point_sequence[1].coords[0]
    c = point_sequence[2].coords[0]

    ang = degrees(
        atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
    if raw_angle:
        return ang
    else:
        ang = ang + 360 if ang < 0 else ang
        # how far is this turn from being a 180?
        ang = abs(round(ang) - 180)

        return ang

from shapely import  ops
## needs a com0plete rehaul ##### MAJOR REHAUL>>>>>>>>>>><>><>
def fuse_degree_2_nodes(node_gdf=None, edge_gdf=None, tolerance_angle=10):
    node_gdf = node_gdf.copy()
    edge_gdf = edge_gdf.copy()
    degree_2_nodes = node_gdf[node_gdf["degree"] == 2]
    fused = 0
    for idx in degree_2_nodes.index:
        fused = fused + 1
        edges = edge_gdf[
            (edge_gdf["start"] == idx) |
            (edge_gdf["end"] == idx)
            ]

        node_list = list(pd.concat([edges["start"], edges["end"]]).unique())
        outer_nodes = [value for value in node_list if value != idx]
        if len(outer_nodes) == 1:
            continue
            # these are two parallel lines sharing the same two nodes..
            # They form a 'stray' loop, dropping both is likely a goop option
            # As dropping one would create a stray edge.
            #print(f"{outer_nodes = }")
            #print(f'{idx = }')
            #print (f"{edges = }")
            edge_gdf.drop(index=edges.iloc[0].name, inplace=True)
            if edges.shape[0] > 1:
                edge_gdf.drop(index=edges.iloc[1].name, inplace=True)
            node_gdf.drop(index=idx, inplace=True)
            node_gdf.at[outer_nodes[0], "degree"] -= 2
            # TODO: trace back the source of such 'loops' and see if there is a cause for them
        elif len(outer_nodes) == 2:
            if node_gdf.at[idx, "degree"] != 2:
                #print(f'{node_gdf.at[idx, "degree"] = }')
                #print(f'{idx = }')
                continue

            first_node = node_gdf.loc[outer_nodes[0]]
            last_node = node_gdf.loc[outer_nodes[1]]

            preserved_edge_id = edges.iloc[0].name
            drop_edge_id = edges.iloc[1].name

            start_point = first_node["geometry"]
            center_point = node_gdf.at[idx, 'geometry']
            end_point = last_node["geometry"]
            line = geo.LineString([start_point, end_point])



            # TODO: This should join two linestrings tohether...
            # drop_edge_coords = list(edge_gdf.at[drop_edge_id, "geometry"].coords)
            # preserve_edge_coords = list(edge_gdf.at[preserved_edge_id, "geometry"].coords)
            #if (geo.Point(drop_edge_coords[0]).distance(geo.Point(preserve_edge_coords[0])) <  center_point)  :
            #    multi_line = geo.MultiLineString([drop_edge_coords, preserve_edge_coords])
            angle = angle_deviation_between_two_lines([start_point, node_gdf.at[idx, "geometry"], end_point])
            if angle < tolerance_angle:


                edge_gdf.at[preserved_edge_id, "geometry"] = line

                edge_gdf.at[preserved_edge_id, "length"] = edge_gdf.at[preserved_edge_id, "geometry"].length
                edge_gdf.at[preserved_edge_id, "weight"] = edge_gdf.at[preserved_edge_id, "weight"] + edge_gdf.at[drop_edge_id, "weight"]
                edge_gdf.at[preserved_edge_id, "start"] = first_node.name
                edge_gdf.at[preserved_edge_id, "end"] = last_node.name

                # Drop end edge
                edge_gdf.drop(index=drop_edge_id, inplace=True)

                # drop node
                node_gdf.drop(index=idx, inplace=True)
        else:
            pass
            #print (f"unexpected situation ...{outer_nodes = }")
    # print("Fused segments: " + str(fused))
    return node_gdf, edge_gdf

# Static Function
def _discard_redundant_edges(edge_gdf):
    se = np.vstack([
        edge_gdf["start"].values,
        edge_gdf["end"].values
    ]).transpose()

    uniques = np.unique(se, return_counts=True,
                        return_index=True, return_inverse=True, axis=0)
    sorted_unique = uniques[0]
    unique_indices = uniques[1]
    inverser = uniques[2]
    unique_frequency = uniques[3]
    return edge_gdf.loc[unique_indices]

# Static FUnction, needs improving?
# Strip down needed values and vectorize accordingly, Never use the .apply() function on a dataframe
# inserting data back into a dataframe should use fastpath and should be given a correct index
def _tag_edges(edge_gdf, tolerance=1.0):
    edge_gdf["tag"] = ""
    edge_gdf['tag'] = np.select(
        condlist=[
            edge_gdf["weight"] < tolerance,
            (edge_gdf["weight"] >= tolerance) & (
                    edge_gdf["weight"] < 3.0 * tolerance),
            edge_gdf["start"] == edge_gdf["end"],
            edge_gdf["geometry"].geom_type != "LineString"
        ],
        choicelist=[
            "shorter than tolerance",
            "within 3x tolerance",
            "cycle_edge",
            "bad_geometry_type"
        ],
        default=edge_gdf['tag']
    )
    duplicates = edge_gdf.apply(
        lambda x: {x["start"], x["end"]}, axis=1).duplicated(keep=False)
    edge_gdf.loc[duplicates, "tag"] = 'redundant_edge'
    return edge_gdf

def handle_short_edges(node_gdf, edge_gdf, connected_segment_tolerance, stray_segment_tolerance, looped_edge_tolerance):
    # TODO: This should have been done when edge_gdf was being built, but then would need to be maintained upon any changes?
    edge_candidates = edge_gdf.join(node_gdf['degree'], on='start', how='left').rename(columns={'degree': 'start_degree'}).join(node_gdf['degree'], on='end', how='left').rename(columns={'degree': 'end_degree'})
    edge_candidates["start_neighbors"] = edge_candidates.apply(lambda x: set(edge_gdf[(edge_gdf["start"] == x['start']) | (edge_gdf["end"] == x['start'])].index) - {x.name}, axis=1)
    edge_candidates["end_neighbors"] = edge_candidates.apply(lambda x: set(edge_gdf[(edge_gdf["start"] == x['end']) | (edge_gdf["end"] == x['end'])].index) - {x.name}, axis=1)



    looped_edges = edge_candidates[
        (edge_candidates["start"] == edge_candidates["end"] ) & ((edge_candidates["weight"] <= looped_edge_tolerance))
        ]

    stray_edges = edge_candidates[
        ((edge_candidates["start_degree"] == 1) | (edge_candidates["end_degree"] == 1))
        & (edge_candidates["weight"] <= stray_segment_tolerance)
        ]

    connected_edges = edge_candidates[
        ((edge_candidates["start_degree"] > 1) & (edge_candidates["end_degree"] > 1))
        & (edge_candidates["weight"] <= connected_segment_tolerance)
        ]
    ## deal with connected segment
    edge_candidates

    drop_edges = set(looped_edges.index).union(set(stray_edges.index))
    reconnect_edges = set(connected_edges.index) - drop_edges

    #print (f"{looped_edges.shape[0] = }\t{stray_edges.shape[0] = }\t{connected_edges.shape[0] = }\t")
    #print (f"{len(drop_edges) = }\t{len(reconnect_edges) = }")
    #print (f"before: \t{node_gdf.shape[0] = }\t{edge_gdf.shape[0] = }")

    for drop_segment_idx  in drop_edges:
        #decrease node degree of start and end, drop node if degree  = 0
        for seg_end in ['start', 'end']:
            node_idx = edge_gdf.at[drop_segment_idx, seg_end]
            if (node_gdf.at[node_idx, 'degree']) == 1:
                node_gdf.drop(index=node_idx, inplace=True) ## maybe put all drop node and edge idxs in lists and drop once for effeciencty?
            else:
                node_gdf.at[node_idx, 'degree'] -= 1
        # drop edge
        edge_gdf.drop(index=drop_segment_idx, inplace=True)
    #print (f"after dropping loops and stray edges: \t{node_gdf.shape[0] = }\t{edge_gdf.shape[0] = }")


    drop_nodes = set()

    node_degrees = []
    node_id = []
    node_geometry = []
    new_node_id = max(list(node_gdf.index))
    for edge_idx in reconnect_edges:
        # find centroid
        # create new node, degree is len(start.union(end))
        centroid_geometry = edge_gdf.at[edge_idx, "geometry"].centroid
        node_neighbors = edge_candidates.at[edge_idx, "start_neighbors"].union(edge_candidates.at[edge_idx, "end_neighbors"])
        node_degree = len(node_neighbors)
        node_degrees.append(node_degree)
        node_geometry.append(centroid_geometry)
        node_id.append(new_node_id)
        new_node_id += 1
        # for each neighbor in start.union(end):
        for neighbor_set, seg_end in zip([edge_candidates.at[edge_idx, "start_neighbors"], edge_candidates.at[edge_idx, "end_neighbors"]], ['start', 'end']):
            for neighbor_edge_idx in neighbor_set:
                for neighbor_seg_end in ['start', 'end']:
                    if neighbor_edge_idx not in list(edge_gdf.index):
                        #print("Investigate this case.."+str(neighbor_edge_idx))
                        continue
                    if edge_gdf.at[neighbor_edge_idx, neighbor_seg_end] == edge_gdf.at[edge_idx, seg_end]:
                        edge_gdf.at[neighbor_edge_idx, neighbor_seg_end] = new_node_id

                        # TODO: reconnect geometry correctly
                        '''
                        centroid = reconnect_edges.at[edge_idx, "geometry"].centroid.coords
                        if neighbor_seg_end == "start":
                            new_line = geo.LineString(edge_gdf.at[neighbor_edge_idx, "geometry"].coords.insert(0, centroid))
                        else:
                            new_line = geo.LineString(edge_gdf.at[neighbor_edge_idx, "geometry"].coords.insert(-1, centroid))

                        edge_gdf.at[neighbor_edge_idx, "geometry"] = new_line
                        '''
                        #edge_gdf.at[neighbor_edge_idx, neighbor_seg_end+'_degree'] = node_degree        # is this necissary?
                        #edge_gdf.at[neighbor_edge_idx, neighbor_seg_end+'_neighbors'] = node_neighbors  # is this necissary?
                        # find the current node (either as start or end)
                        # replace it by current nodes
                        # update degree
                        # update neighbor set
                        # TODO: add node geometry to the edge_linestring by either prepending to the edgers polylines or appending to the polyline based on the start/end (We already know that)


                        # Get the centroid

        # drop old node in placee or add index to list for batched removal
        #node_gdf.drop(index=edge_gdf.at[edge_idx, 'start'], inplace=True) ## maybe put all drop node and edge idxs in lists and drop once for effeciencty?
        #node_gdf.drop(index=edge_gdf.at[edge_idx, 'end'], inplace=True) ## maybe put all drop node and edge idxs in lists and drop once for effeciencty?
        drop_nodes.add(edge_gdf.at[edge_idx, 'start'])
        drop_nodes.add(edge_gdf.at[edge_idx, 'end'])

    # drop current edge
    edge_gdf.drop(index=reconnect_edges, inplace=True)

    node_gdf.drop(
        index=set(node_gdf.index).intersection(drop_nodes),
        inplace=True
    )





    node_count = len(node_degrees)
    index = pd.Index(node_id, name="id")
    new_node_gdf = gpd.GeoDataFrame(
        {
            "source_layer": pd.Series(np.repeat(np.array(["streets"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "source_id":    pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
            "type":         pd.Series(np.repeat(np.array(["street_node"], dtype=object), repeats=node_count), fastpath=True, index=index, dtype="category"),
            "weight":       pd.Series(np.repeat(np.array([0.0], dtype=np.float32), repeats=node_count), fastpath=True, index=index, dtype=np.float32),
            "nearest_street_id": pd.Series(np.repeat(np.array([0], dtype=np.int32), repeats=node_count), fastpath=True, index=index, dtype=np.int32),
            "nearest_street_node_distance": pd.Series(np.repeat(np.array([{}], dtype=object), repeats=node_count), fastpath=True, index=index),
            "degree":       pd.Series(np.array(node_degrees, dtype=np.int32), fastpath=True, index=index),
            # "connected_edges": connected_edges
        },
        index=index,
        geometry=node_geometry,
        crs=node_gdf.crs
    )
    node_gdf = pd.concat([node_gdf, new_node_gdf])
    return node_gdf, edge_gdf



start = time.time()
### Parameters
preferred_projected_coordinates_system = None
node_snapping_tolerance = 3.28


dataset_path = "C:/Users/abdul/Dropbox (MIT)/115_NYCWalks/03_Data/01_Raw/NYC_Planimetrics_SidewalkNetwork/2022_PEDESTRIAN_DATA_RESEARCH_DRAFT.gdb/2022_PEDESTRIAN_DATA_RESEARCH_DRAFT.gdb"
# dataset_path="Planemetric_Sidewalks.shp"
layers = fiona.listlayers(dataset_path)
# print(layers)
#layers[1]

source_geometry_gdf = gpd.read_file(
    filename= dataset_path,    #"./Map Cleaning Script/Planemetric_Sidewalks.shp",
    engine="pyogrio",   # pyogrio or fiona
    read_geometry=True,
    layer=layers[2],
    #bbox=None,
    #mask=None,
    #rows=None,
    ### --- pyogrio settings --- ###
    columns=["OBJECTID"], #"geometry",
    force_2d=True,
    #use_arrow=True,
)#.set_index("OBJECTID")
print ("Data is Loaded")
print (f"Current projection is {source_geometry_gdf.crs}")


# Loading NTA file so we copuld clip
#NTAs = gpd.read_file("./Map Cleaning Script/geo_export_73f5b4b0-c2ca-4646-9e84-d591f2562f1f.shp")
NTAs = gpd.read_file("geo_export_73f5b4b0-c2ca-4646-9e84-d591f2562f1f.shp")


BX_NTAs = NTAs[NTAs["nta2020"].isin(["BX0101", "BX0102"])]
scope = BX_NTAs.to_crs(source_geometry_gdf.crs).dissolve().buffer(804000000000000000000.672)

scopeed_geometry_gdf = gpd.clip(source_geometry_gdf, scope)
#scopeed_geometry_gdf = source_geometry_gdf
print ("Clipped to scope")

scopeed_geometry_gdf.index= pd.Index(np.arange(scopeed_geometry_gdf.shape[0]), name="id")

# Finding the geographic center for visualization
geographic_geometry_gdf = scopeed_geometry_gdf.to_crs(
    default_geographic_crs)
center_x, center_y = geographic_geometry_gdf.dissolve(
).at[0, "geometry"].centroid.coords[0]

prepared_geometry  = _prepare_geometry(scopeed_geometry_gdf)
node_gdf, edge_gdf = _node_edge_builder(prepared_geometry, weight_attribute=None, tolerance=0.0)

pretreatment_node_gdf = node_gdf.copy(deep=True)
pretreatment_edge_gdf = edge_gdf.copy(deep=True)

stats = {
    "time": time.time() - start,
    "Step": "Baseline",
    "Node Count": node_gdf.shape[0],
    "Degree 2": node_gdf[node_gdf['degree'] == 2].shape[0],
    "Degree 1":node_gdf[node_gdf['degree'] == 1].shape[0],
    "Edge Count": edge_gdf.shape[0],
    "Average Edge Length": edge_gdf['geometry'].length.mean(),
    "Edge shorter than tolerance": edge_gdf[edge_gdf['geometry'].length < node_snapping_tolerance].shape[0]
}
print (stats)


node_gdf, edge_gdf = _effecient_network_nodes_edges(
    prepared_geometry,
    weight_attribute=None,
    tolerance=node_snapping_tolerance
)

stats = {
    "time": time.time() - start,
    "Step": "Node Snapping",
    "Node Count": node_gdf.shape[0],
    "Degree 2": node_gdf[node_gdf['degree'] == 2].shape[0],
    "Degree 1":node_gdf[node_gdf['degree'] == 1].shape[0],
    "Edge Count": edge_gdf.shape[0],
    "Average Edge Length": edge_gdf['geometry'].length.mean(),
    "Edge shorter than tolerance": edge_gdf[edge_gdf['geometry'].length < node_snapping_tolerance].shape[0]
}
print (stats)


node_gdf, edge_gdf = fuse_degree_2_nodes(
    node_gdf=node_gdf,
    edge_gdf=edge_gdf,
    tolerance_angle=45
)

stats = {
    "time": time.time() - start,
    "Step": "Fuse Degree-2",
    "Node Count": node_gdf.shape[0],
    "Degree 2": node_gdf[node_gdf['degree'] == 2].shape[0],
    "Degree 1":node_gdf[node_gdf['degree'] == 1].shape[0],
    "Edge Count": edge_gdf.shape[0],
    "Average Edge Length": edge_gdf['geometry'].length.mean(),
    "Edge shorter than tolerance": edge_gdf[edge_gdf['geometry'].length < node_snapping_tolerance].shape[0]
}
print (stats)

#%%
node_gdf, edge_gdf = handle_short_edges(
    node_gdf,
    edge_gdf,
    stray_segment_tolerance = 15 * 3.28,
    connected_segment_tolerance = 5*3.28,
    looped_edge_tolerance = 5*3.28
)


stats = {
    "time": time.time() - start,
    "Step": "Short edge handling",
    "Node Count": node_gdf.shape[0],
    "Degree 2": node_gdf[node_gdf['degree'] == 2].shape[0],
    "Degree 1":node_gdf[node_gdf['degree'] == 1].shape[0],
    "Edge Count": edge_gdf.shape[0],
    "Average Edge Length": edge_gdf['geometry'].length.mean(),
    "Edge shorter than tolerance": edge_gdf[edge_gdf['geometry'].length < node_snapping_tolerance].shape[0]
}

print (stats)
#%%409160614014, 'Step': 'Fuse Degree-2', 'Node Count': 269977, 'Degree 2': 13466, 'Degree 1': 88498, 'Edge Count': 352990, 'Average Edge Length': 189.56347557443746, 'Edge shorter than tolerance': 7337}
