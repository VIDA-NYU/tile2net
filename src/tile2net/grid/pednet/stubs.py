from __future__ import annotations
import logging
from ..benchmark import benchmark
from tqdm.auto import tqdm
from tile2net.grid.cfg.logger import logger

import heapq

from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from ...grid.frame.namespace import namespace
from typing import Self

INF = float('inf')
from .standalone import Lines

if False:
    from .pednet import PedNet
    import folium


class Stubs(
    Lines,
):
    instance: Lines = None
    __name__ = 'stubs'
    
    def _get(
            self,
            instance: Lines,
            owner: type[PedNet]
    ) -> Stubs:
        self: Self = namespace._get(self, instance, owner)
        cache = instance.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result: Self = cache[key]
            # if result.instance is not instance:
            #     loc = result.iline.isin(instance.iline)
            #     result = result.loc[loc].copy()
            #     cache[key] = result

            if result.instance is not instance:
                del cache[key]
                return  self._get(instance, owner)
        else:
            lines = instance
            edges = lines.edges
            nodes = lines.nodes
            _ = edges.threshold, edges.start_degree, edges.stop_degree

            icoord2cost: dict[int, float] = edges.frame.length.to_dict()
            icoord2icoord = edges.stop_iend.to_dict()
            icoord2node = edges.start_tuple.to_dict()

            msg = f'Computing stubs'
            logger.debug(msg)

            loc = edges.frame.length.values <= edges.threshold.values
            loc &= edges.start_degree.values == 1
            ends = edges.loc[loc]

            stubs = set(ends.iline.values)

            icoord2iline = pd.Series(edges.iline.values, index=edges.start_iend).to_dict()
            icoord2inode = pd.Series(edges.stop_inode.values, index=edges.stop_iend).to_dict()
            inode2cost = dict.fromkeys(edges.start_inode.values, INF)

            it = zip(
                ends.start_iend.values,  # ifirst
                ends.stop_iend.values,  # ilast
                ends.frame.length.values,  # cost
                ends.iline.values,  # iline
                ends.stop_inode.values,  # inode
                ends.threshold.values,  # stub_length
            )
            it = tqdm(
                it,
                total=len(ends),
                disable=not logger.isEnabledFor(logging.DEBUG),
            )
            msg = 'Performing depth first search to find stubs'
            with benchmark(msg):
                for ifirst, ilast, cost, iline, inode, stub_length in it:
                    node = icoord2node[ilast]
                    queue = [
                        (cost + icoord2cost[ifirst], ifirst)
                        for ifirst in node
                    ]
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
                            item = icoord2cost[ifirst] + cost, ifirst
                            heapq.heappush(queue, item)

            loc = edges.iline.isin(stubs)
            stub_edges = edges.loc[loc]
            keep_edges = edges.loc[~loc]

            loc = nodes.inode.isin(stub_edges.start_inode)
            loc &= nodes.inode.isin(keep_edges.start_inode)
            connections = nodes.loc[loc]

            loc = nodes.inode.isin(stub_edges.start_inode)
            loc &= nodes.degree.values > 1
            legal_inodes = set(nodes.inode[loc])
            legal_icoords = set(stub_edges.start_iend)

            inode2stub_length = (
                stub_edges.frame
                .groupby("stop_inode")
                ["threshold"]
                .min()
                .to_dict()
            )

            KEEP_ILINES: set[int] = set()
            it = zip(connections.tuple, connections.inode)
            it = tqdm(
                it,
                total=len(connections),
                disable=not logger.isEnabledFor(logging.DEBUG),
            )

            desc = 'Traversing from stub terminals to protect lines from excessive pruning'
            with benchmark(desc):

                for node, inode in it:
                    stub_length = inode2stub_length.get(inode, INF)
                    inode2cost: dict[int, float] = {inode: 0.0}
                    inode2ifirst: dict[int, int] = {}
                    keep_ilines: set[int] = set()

                    queue = [
                        (icoord2cost[ifirst], ifirst, icoord2icoord[ifirst], inode)
                        for ifirst in node
                        if ifirst in legal_icoords
                        if (inode := icoord2inode[icoord2icoord[ifirst]]) in legal_inodes
                    ]
                    heapq.heapify(queue)

                    while queue:
                        cost, ifirst, ilast, inode = heapq.heappop(queue)
                        if inode2cost.get(inode, INF) <= cost:
                            continue
                        inode2ifirst[inode] = ifirst
                        inode2cost[inode] = cost

                        for ifirst in icoord2node[ilast]:
                            ilast = icoord2icoord[ifirst]
                            inode = icoord2inode[ilast]
                            if (
                                    ifirst not in legal_icoords
                                    or inode not in legal_inodes
                            ):
                                continue
                            item = icoord2cost[ifirst] + cost, ifirst, ilast, inode
                            heapq.heappush(queue, item)

                    for inode, cost in inode2cost.items():
                        if cost < stub_length:
                            continue
                        while inode in inode2ifirst:
                            ifirst = inode2ifirst[inode]
                            iline = icoord2iline[ifirst]
                            if iline in keep_ilines:
                                break
                            keep_ilines.add(iline)
                            inode = icoord2inode[ifirst]

                    KEEP_ILINES.update(keep_ilines)

            loc = lines.iline.isin(stubs)
            loc &= ~lines.iline.isin(KEEP_ILINES)
            cols = lines.__keep__
            result = (
                lines
                .loc[loc, cols]
                .frame
                .pipe(self.from_frame, wrapper=self)
            )
            cache[key] = result

        result.instance = instance
        return result
        
    locals().update(
        __get__=_get,
    )

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line_color='grey',
            stub_color='yellow',
            node_color='red',
            polygon_color='grey',
            simplify=None,
            **kwargs,
    ) -> folium.Map:
        import folium
        lines = self.instance
        loc = ~lines.iline.isin(self.iline)
        lines = lines.loc[loc]

        features = self.instance.pednet.features
        feature2color = features.color.to_dict()
        _ = features.mutex

        if 'original' in features:
            it = features.frame.groupby(
                level='feature',
                observed=False,
            )
            for feature, frame in it:
                color = feature2color[feature]
                m = explore(
                    frame,
                    geometry='original',
                    *args,
                    color=color,
                    name=f'{feature} (original)',
                    tiles=tiles,
                    simplify=simplify,
                    m=m,
                    style_kwds=dict(
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.05,
                        weight=0,  # no stroke
                    ),
                    highlight=False,
                    **kwargs,
                )

        if polygon_color:
            m = explore(
                # self.instance.pednet.union,
                self.instance.pednet.union,
                *args,
                color=polygon_color,
                name=f'polygons',
                tiles=tiles,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    dashArray='5, 15',
                    fill=False,
                ),
                **kwargs,
            )

        m = explore(
            lines,
            color=line_color,
            name='lines',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        m = explore(
            self,
            color=stub_color,
            name='stubs',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        nodes = self.nodes
        loc = nodes.index.isin(self.start_inode.values)
        loc |= nodes.index.isin(self.stop_inode.values)
        nodes = nodes.loc[loc]
        m = explore(
            nodes,
            color=node_color,
            name='node',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        folium.LayerControl().add_to(m)
        return m


Lines.stubs = Stubs()
