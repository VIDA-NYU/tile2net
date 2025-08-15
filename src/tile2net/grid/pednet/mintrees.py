from __future__ import annotations
import logging
import tqdm.auto
import tqdm

import heapq

from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from tile2net.grid.cfg.logger import logger
from ...grid.frame.namespace import namespace
from typing import Self

INF = float('inf')
from .standalone import Lines

if False:
    from .pednet import PedNet
    import folium


class Mintrees(
    Lines,
):
    instance: Lines = None
    __name__ = 'mintrees'
    
    def _get(
            self,
            instance: Lines,
            owner: type[PedNet]
    ) -> Mintrees:
        self: Self = namespace._get(self, instance, owner)
        cache = instance.__dict__
        key = self.__name__
        if instance is None:
            return self
        if key in cache:
            result = cache[key]
            if result.instance is not instance:
                del cache[key]
                return  self._get(instance, owner)
        else:
            stubs = instance.stubs
            edges = stubs.edges
            nodes = stubs.nodes
            INF = float('inf')
            msg = f'Computing mintrees to preserve connectivity during pruning'
            logger.debug(msg)

            assert edges.iline.isin(stubs.iline).all()
            assert edges.start_iend.isin(edges.stop_iend.values).all()
            assert stubs.start_inode.isin(nodes.inode.values).all()
            assert edges.start_inode.isin(nodes.inode.values).all()
            assert nodes.inode.isin(edges.start_inode.values).all()
            _ = nodes.tuple

            icoord2cost: dict[int, float] = edges.frame.length.to_dict()
            icoord2icoord: dict[int, int] = edges.stop_iend.to_dict()
            icoord2inode = edges.start_inode.to_dict()
            icoord2node: dict[int, tuple[int, ...]] = (
                edges.start_tuple
                .to_dict()
            )

            edges = instance.edges
            nodes = instance.nodes
            loc = ~edges.iline.isin(stubs.iline)
            edges = edges.loc[loc]
            cols = 'start_x start_y'.split()
            haystack = pd.MultiIndex.from_frame(edges[cols])
            needles = pd.MultiIndex.from_frame(stubs.nodes['x y'.split()])
            loc = needles.isin(haystack)
            cols = 'stop_x stop_y'.split()
            haystack = pd.MultiIndex.from_frame(edges[cols])
            loc |= needles.isin(haystack)
            icoord2iline = stubs.edges.iline.to_dict()
            is_terminal = pd.Series(
                loc,
                index=stubs.nodes.inode,
                dtype=bool
            )


            nodes = stubs.nodes

            inode2is_terminal: dict[int, bool] = is_terminal.to_dict()
            assert edges.stop_iend.isin(edges.start_iend.values).all()
            terminals = nodes.loc[is_terminal]

            result_icoords: set[int] = set()
            result_inodes: set[int] = set()

            it = tqdm.auto.tqdm(
                zip(terminals.tuple, terminals.inode),
                total=len(terminals),
                desc='Iterating across terminal nodes',
                disable=not logger.isEnabledFor(logging.DEBUG),
            )

            for node, inode in it:
                INODE = inode
                inode2cost: dict[int, float] = {}
                inode2ifirst: dict[int, int] = {}
                inode2cost[inode] = 0.0
                queue = [
                    (icoord2cost[ifirst], ifirst)
                    for ifirst in node
                ]
                heapq.heapify(queue)
                while queue:
                    cost, ifirst = heapq.heappop(queue)
                    iline = icoord2iline[ifirst]
                    ilast = icoord2icoord[ifirst]
                    inode = icoord2inode[ilast]
                    # todo: what if the cost lower, but it's a terminal?

                    if inode2cost.get(inode, INF) <= cost:
                        continue
                    inode2ifirst[inode] = ifirst
                    inode2cost[inode] = cost

                    if (
                            inode in result_inodes
                            or inode2is_terminal[inode]
                    ):
                        while (
                                inode in inode2ifirst
                                # and inode not in result_inodes
                        ):
                            ifirst = inode2ifirst[inode]
                            result_icoords.add(ifirst)
                            result_inodes.add(inode)
                            inode = icoord2inode[ifirst]
                            if inode == INODE:
                                break
                        # break


                    node = icoord2node[ilast]
                    for ifirst in node:
                        item = icoord2cost[ifirst] + cost, ifirst
                        heapq.heappush(queue, item)

            icoord = np.fromiter(result_icoords, int, len(result_icoords))

            edges = instance.edges
            iline = (
                edges.iline
                .loc[icoord]
                .drop_duplicates()
            )
            result = (
                edges.lines
                .loc[iline]
                .frame
                .sort_index()
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
            grid='cartodbdark_matter',
            m=None,
            line_color='grey',
            stub_color='yellow',
            mintree_color='green',
            node_color='red',
            polygon_color='grey',
            simplify: float = None,
            dash='5, 20',
            **kwargs,
    ) -> folium.Map:
        import folium
        lines = self.instance
        stubs = lines.stubs
        loc = ~lines.iline.isin(stubs.iline)
        lines = lines.loc[loc]

        if polygon_color:
            m = explore(
                self.instance.pednet.union.frame,
                *args,
                color=polygon_color,
                name=f'polygons',
                grid=grid,
                simplify=simplify,
                m=m,
                style_kwds=dict(
                    fill=False,
                    dashArray=dash,
                ),
                **kwargs,
            )

        m = explore(
            lines,
            color=line_color,
            name='lines',
            *args,
            **kwargs,
            grid=grid,
            m=m,
        )
        m = explore(
            stubs.frame,
            color=stub_color,
            name='stubs',
            *args,
            **kwargs,
            grid=grid,
            m=m,
        )
        m = explore(
            self.frame,
            color=mintree_color,
            name='mintrees',
            *args,
            **kwargs,
            grid=grid,
            m=m,
        )
        nodes = self.nodes
        loc = nodes.index.isin(stubs.start_inode.values)
        loc |= nodes.index.isin(stubs.stop_inode.values)
        nodes = nodes.loc[loc]
        m = explore(
            nodes.frame,
            color=node_color,
            name='node',
            *args,
            **kwargs,
            grid=grid,
            m=m,
        )
        folium.LayerControl().add_to(m)
        return m


Lines.mintrees = Mintrees()
