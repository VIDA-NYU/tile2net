from __future__ import annotations

import heapq

import pandas as pd

from tile2net.raster.tile_utils.topology import *
from ..explore import explore

INF = float('inf')
from .standalone import Lines

if False:
    from .pednet import PedNet
    from .standalone import Edges, Nodes
    from .center import Center
    import folium


def __get__(
        self: Mintrees,
        instance: Lines,
        owner: type[PedNet]
) -> Mintrees:
    if instance is None:
        result = self
    elif self.__name__ in instance.__dict__:
        result = instance.__dict__[self.__name__]
    else:
        stubs = instance.stubs
        edges = stubs.edges
        nodes = stubs.nodes
        INF = float('inf')

        edges.start_inode.max()
        edges.stop_inode.max()
        edges.start_end
        edges.stop_end
        stubs.iline.is_monotonic_increasing

        assert edges.iline.isin(stubs.iline).all()
        assert edges.start_end.isin(edges.stop_end.values).all()
        assert stubs.start_inode.isin(nodes.inode.values).all()
        assert edges.start_inode.isin(nodes.inode.values).all()
        assert nodes.inode.isin(edges.start_inode.values).all()
        _ = nodes.tuple

        icoord2cost: dict[int, float] = edges.length.to_dict()
        icoord2icoord: dict[int, int] = edges.stop_end.to_dict()
        # icoord2inode: dict[int, int] = pd.Series(
        #     edges.stop_inode.values,
        #     index=edges.stop_end
        # ).to_dict()
        icoord2inode = edges.stop_inode.to_dict()
        icoord2node: dict[int, tuple[int, ...]] = (
            edges.start_tuple
            .to_dict()
        )

        # edges = instance.edges
        # loc = ~edges.iline.isin(stubs.iline)

        # todo: determine which nodes are terminal
        #   terminal if only 1 non-stub edge in the node
        edges = instance.edges
        nodes = instance.nodes
        data = ~edges.iline.isin(stubs.iline)
        haystack = pd.MultiIndex.from_frame(nodes['x y'.split()])

        needles = pd.MultiIndex.from_frame(stubs.nodes['x y'.split()])
        is_terminal = (
            pd.Series(data)
            .groupby(edges.start_inode)
            .sum()
            .eq(1)
            .loc[nodes.inode]
            .set_axis(haystack)
            .reindex(needles, fill_value=False)
            .set_axis(stubs.nodes.inode)
        )

        nodes = stubs.nodes

        inode2is_terminal: dict[int, bool] = is_terminal.to_dict()
        assert edges.stop_end.isin(edges.start_end.values).all()
        terminals = nodes.loc[is_terminal]

        inode2cost: dict[int, float] = {}
        inode2ifirst: dict[int, int] = {}
        result_icoords: set[int] = set()
        result_inodes: set[int] = set()

        for node, inode in zip(terminals.tuple, terminals.inode):
            inode2cost[inode] = 0.0
            queue = [
                (icoord2cost[ifirst], ifirst)
                for ifirst in node
            ]
            heapq.heapify(queue)
            while queue:
                cost, ifirst = heapq.heappop(queue)
                ilast = icoord2icoord[ifirst]
                inode = icoord2inode[ilast]
                if inode2cost.get(inode, INF) <= cost:
                    continue

                inode2ifirst[inode] = ifirst
                inode2cost[inode] = cost

                if inode in result_inodes or inode2is_terminal[inode]:
                    while inode in inode2ifirst and inode not in result_inodes:
                        ifirst = inode2ifirst[inode]
                        result_icoords.add(ifirst)
                        result_inodes.add(inode)
                        inode = icoord2inode[ifirst]
                    break

                for nxt_ifirst in icoord2node[ilast]:
                    item = icoord2cost[nxt_ifirst] + cost, nxt_ifirst
                    heapq.heappush(queue, item)

        icoord = np.fromiter(result_icoords, int, len(result_icoords))

        #

        result = (
            edges.iline
            .loc[icoord]
            .drop_duplicates()
            .pipe(edges.lines.loc)
            .sort_index()
        )

        instance.__dict__[self.__name__] = result

    result.instance = instance
    return result


class Mintrees(
    Lines,
):
    locals().update(
        __get__=__get__,
    )

    instance: Lines = None
    __name__ = 'mintrees'

    @property
    def nodes(self) -> Nodes:
        nodes = self.instance.nodes
        _ = nodes.tuple, nodes.degree
        loc = nodes.inode.isin(self.start_inode)
        loc |= nodes.inode.isin(self.stop_inode)
        result = nodes.loc[loc].copy()
        result.lines = self.instance
        return result

    @property
    def edges(self) -> Edges:
        edges = self.instance.edges
        _ = edges.start_tuple, edges.stop_tuple, edges.start_degree, edges.stop_degree
        loc = edges.iline.isin(self.iline)
        result = edges.loc[loc].copy()
        result.lines = self.instance
        return result

    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
            m=None,
            line_color='grey',
            stub_color='yellow',
            mintree_color='green',
            node_color='red',
            **kwargs,
    ) -> folium.Map:
        import folium
        lines = self.instance
        stubs = lines.stubs
        loc = ~lines.iline.isin(stubs.iline)
        lines = lines.loc[loc]

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
            stubs,
            color=stub_color,
            name='stubs',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        m = explore(
            mintree_color,
            color=mintree_color,
            name='mintrees',
            *args,
            **kwargs,
            tiles=tiles,
            m=m,
        )
        nodes = self.nodes
        loc = nodes.index.isin(stubs.start_inode.values)
        loc |= nodes.index.isin(stubs.stop_inode.values)
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


Lines.mintrees = Mintrees()
