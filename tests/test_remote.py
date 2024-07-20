import abc

import pytest

import tile2net.raster.source
from tile2net.raster.source import SourceNotFound
from tile2net.raster.raster import Raster


def test_small():
    raster = Raster(
        location='Washington Square Park, New York, NY, USA',
        zoom=19,
        # dump_percent=10,
        name='small'
    )
    # for file in raster.project.resources.segmentation.files():
    # assert file.exists()
    # assert os.path.exists(file)
    raster.generate(2)
    raster.inference('--remote', '--debug')


def test_sources():
    import tile2net.raster.source as source
    from tile2net.raster.source import Source
    for key in dir(tile2net.raster.source):
        cls = getattr(tile2net.raster.source, key)
        if (
                not isinstance(cls, type)
                or not issubclass(cls, Source)
                or abc.ABC in cls.__bases__
                or getattr(cls, 'ignore', False)
        ):
            continue
        # assert querying by the polygon returns the same source
        # assert Source[cls.coverage.unary_union] == cls
        assert Source[cls.coverage.union_all()] == cls
        # assert querying by the keyword returns the same source
        if isinstance(cls.keyword, str):
            assert Source[cls.keyword] == cls
        else:
            assert any(
                Source[keyword] == cls
                for keyword in cls.keyword
            )
        # assert querying by the name returns the same source
        assert Source[cls.name] == cls

    assert Source['New York'] in (source.NewYorkCity, source.NewYork)
    assert Source['New York City'] == source.NewYorkCity
    assert Source['New Jersey'] == source.NewJersey
    assert Source['New Brunswick, New Jersey'] == source.NewJersey
    assert Source['Massachusetts'] == source.Massachusetts
    assert Source['King County, Washington'] == source.KingCountyWashington
    # assert Source['Washington, DC'] == source.WashingtonDC
    assert Source['Los Angeles'] == source.LosAngeles
    assert Source['Jersey City'] == source.NewJersey
    assert Source['Hoboken'] == source.NewJersey
    assert Source["Spring Hill, TN"] == source.SpringHillTN
    assert Source['Oregon'] == source.Oregon
    assert Source['Virginia'] == source.Virginia

    assert Source['40.72663613847755, -73.99494276578649'] == source.NewYorkCity
    # assert Source['38.90277706745021, -77.03617656372798'] == source.WashingtonDC
    assert Source['43.05052202494481, -76.19505424681927'] == source.NewYork
    item = '33.97576931943177, -118.19841961122856, 34.116579445776445, -117.97154942950205'
    assert Source[item] == source.LosAngeles
    item = '40.496044, -74.443672, 40.561051, -74.332089'
    assert Source[item] == source.NewJersey

    assert Source[40.72663613847755, -73.99494276578649] == source.NewYorkCity
    # assert Source[38.90277706745021, -77.03617656372798] == source.WashingtonDC
    assert Source[43.05052202494481, -76.19505424681927] == source.NewYork
    item = 33.97576931943177, -118.19841961122856, 34.116579445776445, -117.97154942950205
    assert Source[item] == source.LosAngeles
    item = 40.496044, -74.443672, 40.561051, -74.332089
    assert Source[item] == source.NewJersey

    assert Source['nyc'] == source.NewYorkCity
    assert Source['ny'] == source.NewYork
    assert Source['nj'] == source.NewJersey
    assert Source['new jersey'] == source.NewJersey
    assert Source['la'] == source.LosAngeles
    # just Spring Hill returns Spring Hill, Virgnia
    assert Source['Spring Hill, Tennessee'] == source.SpringHillTN
    assert Source['va'] == source.Virginia

    assert Source['Maywood, California'] == source.LosAngeles
    assert Source['Maywood, CA'] == source.LosAngeles

    # namibia
    assert Source[-15.49933207, 28.203229539, -15.338660813, 28.358324353] == None
    # beijing
    assert Source[39.525834067367256, 116.21383162969653, 39.582584, 116.292915] == None
    # ocean
    assert Source[-37.15612782594927, 64.98947402062927, -37.15612782594927, 64.98947402062927] == None
    # greenland
    assert Source[73.59343881883807, -51.62165778082543, 73.59343881883807, -51.62165778082543] == None
    # russia
    assert Source[54.998172689668486, 36.68930259694381, 55.00000000000001, 36.69112990727614] == None
    # algeria
    assert Source[25.06763435341293, -0.7971811600872423, 25.06763435341293, -0.7971811600872423] == None
    # gulf of mexico
    assert Source[21.82528963135751, -93.76345422053639, 21.82528963135751, -93.76345422053639] == None

if __name__ == '__main__':
    test_small()
    test_sources()
