from __future__ import annotations
import heapq


from functools import *

import shapely.wkt
from centerline.geometry import Centerline
from tqdm import tqdm

from tile2net.logger import logger
from tile2net.raster.tile_utils.geodata_utils import set_gdf_crs
from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from ..fixed import GeoDataFrameFixed
INF = float('inf')

if False:
    from .pednet import PedNet
    from .center import Center
    import folium


def __get__(
        self,
        instance: Center,
        owner: type[PedNet]
) -> Stubs:
    if instance is None:
        result = self
    elif self.__name__ in instance.attrs:
        result = instance.attrs[self.__name__]
    else:
        lines = instance.lines
        edges = lines.edges
        nodes = lines.nodes
        _ = edges.threshold, edges.start_degree, edges.stop_degree

        icoord2cost: dict[int, float] = edges.length.to_dict()
        icoord2icoord = edges.stop_end.to_dict()
        icoord2node = edges.tuple.to_dict()

        loc = edges.length.values <= edges.threshold.values
        loc &= edges.start_degree.values == 1
        ends = edges.loc[loc]

        stubs = set(ends.iline.values)

        icoord2iline = pd.Series(edges.iline.values, index=edges.start_end).to_dict()
        icoord2inode = pd.Series(edges.stop_inode.values, index=edges.stop_end).to_dict()
        inode2cost = dict.fromkeys(edges.inode.values, INF)

        it = zip(
            ends.start_end.values,  # ifirst
            ends.stop_end.values,  # ilast
            ends.length.values,  # cost
            ends.iline.values,  # iline
            ends.stop_inode.values,  # inode
            ends.threshold.values,  # stub_length
        )

        for ifirst, ilast, cost, iline, inode, stub_length in it:
            node = icoord2node[ilast]
            queue = [(cost + icoord2cost[ifirst], ifirst) for ifirst in node]
            heapq.heapify(queue)
            visited = {iline}

            while queue:
                cost, ifirst = heapq.heappop(queue)
                iline = icoord2iline[ifirst]
                ilast = icoord2icoord[ifirst]
                inode = icoord2inode[ilast]
                if iline in visited:
                    continue
                visited.add(iline)

                if cost > stub_length or cost >= inode2cost[inode]:
                    continue
                stubs.add(iline)
                inode2cost[inode] = cost
                node = icoord2node[ilast]
                for ifirst in node:
                    iline = icoord2iline[ifirst]
                    if iline in visited:
                        continue
                    heapq.heappush(queue, (icoord2cost[ifirst] + cost, ifirst))


class Stubs(
    GeoDataFrameFixed,
):
    locals().update(
        __get__=__get__,
    )

    instance: PedNet = None
    __name__ = 'center'

