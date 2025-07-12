from __future__ import annotations
import logging
import tqdm.auto
import tqdm

import heapq

from tile2net.raster.tile_utils.topology import *
from ..explore import explore
from tile2net.tiles.cfg.logger import logger

INF = float('inf')
from .standalone import Lines

if False:
    from .pednet import PedNet
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
        msg = f'Computing mintrees to preserve connectivity during pruning'
        logger.debug(msg)

        assert edges.iline.isin(stubs.iline).all()
        assert edges.start_iend.isin(edges.stop_iend.values).all()
        assert stubs.start_inode.isin(nodes.inode.values).all()
        assert edges.start_inode.isin(nodes.inode.values).all()
        assert nodes.inode.isin(edges.start_inode.values).all()
        _ = nodes.tuple

        icoord2cost: dict[int, float] = edges.length.to_dict()
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
            .sort_index()
            .pipe(self.__class__)
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


    def visualize(
            self,
            *args,
            tiles='cartodbdark_matter',
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
                self.instance.pednet.union,
                *args,
                color=polygon_color,
                name=f'polygons',
                tiles=tiles,
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
            self,
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
