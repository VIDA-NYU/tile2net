from __future__ import annotations

from .lines import Lines

if False:
    pass

"""
without magicpandas,
input: lines
output: nodes, aggregated

one to drop degree=2 noeds
one to extract node information
"""


# Todo: this file is defunct, everything should just import from lines instead of standalone



if __name__ == '__main__':
    import geopandas as gpd

    file = '~/PycharmProjects/kashi/src/tile2net/artifacts/static/brooklyn.feather'
    result = (
        gpd.read_feather(file)
        .pipe(Lines.from_center)
        .drop2nodes
    )
    print(f'{len(result)=}')
    print(f'{len(result)=}')
    result.explore()
