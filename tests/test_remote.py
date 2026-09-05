import abc

import pytest

pytestmark = pytest.mark.remote


def test_nominatim():
    # Test querying Nominatim actually works
    from tile2net.raster import source
    from tile2net.raster.source import Source
    from tile2net.raster.nominatim import Nominatim
    Nominatim.json.read = False
    Nominatim.json.write = False
    Nominatim.sqlite.read = False
    Nominatim.sqlite.write = False
    assert Source.from_inferred('New York') == source.NewYork
    assert Source.from_inferred('New York City') == source.NewYorkCity
    assert Source.from_inferred('New Jersey') == source.NewJersey



def test_sources():
    import tile2net.raster.source
    from tile2net.raster import source
    from tile2net.raster.source import Source
    from tile2net.raster.nominatim import Nominatim
    Nominatim.json.read = True
    Nominatim.json.write = True
    for key in dir(tile2net.raster.source):
        cls = getattr(tile2net.raster.source, key)
        if (
                not isinstance(cls, type)
                or not issubclass(cls, Source)
                or abc.ABC in cls.__bases__
                or getattr(cls, 'ignore', False)
                or cls.outdated
        ):
            continue
        # assert querying by the polygon returns the same source
        # assert Source.from_inferred(cls.coverage.unary_union) == cls
        assert Source.from_inferred(cls.coverage.union_all()) == cls

        # assert querying by the name returns the same source
        assert Source.from_inferred(cls.name) == cls

    assert Source.from_inferred('New York') in (source.NewYorkCity, source.NewYork)
    assert Source.from_inferred('New York City') == source.NewYorkCity
    assert Source.from_inferred('New Jersey') == source.NewJersey
    assert Source.from_inferred('New Brunswick, New Jersey') == source.NewJersey
    assert Source.from_inferred('Massachusetts') == source.Massachusetts
    assert Source.from_inferred('King County, Washington') == source.KingCountyWashington
    assert Source.from_inferred('Washington, DC') == source.WashingtonDC
    assert Source.from_inferred('Los Angeles') == source.LosAngeles
    assert Source.from_inferred('Jersey City') == source.NewJersey
    assert Source.from_inferred('Hoboken') == source.NewJersey
    assert Source.from_inferred("Spring Hill, TN") == source.SpringHillTN
    # assert Source.from_inferred('Oregon') == source.Oregon
    assert Source.from_inferred('Virginia') == source.Virginia

    assert Source.from_inferred('40.72663613847755, -73.99494276578649') == source.NewYorkCity
    assert Source.from_inferred('38.90277706745021, -77.03617656372798') == source.WashingtonDC
    assert Source.from_inferred('43.05052202494481, -76.19505424681927') == source.NewYork
    item = '33.97576931943177, -118.19841961122856, 34.116579445776445, -117.97154942950205'
    assert Source.from_inferred(item) == source.LosAngeles
    item = '40.496044, -74.443672, 40.561051, -74.332089'
    assert Source.from_inferred(item) == source.NewJersey

    assert Source.from_inferred((40.72663613847755, -73.99494276578649)) == source.NewYorkCity
    assert Source.from_inferred((38.90277706745021, -77.03617656372798)) == source.WashingtonDC
    assert Source.from_inferred((43.05052202494481, -76.19505424681927)) == source.NewYork
    item = 33.97576931943177, -118.19841961122856, 34.116579445776445, -117.97154942950205
    assert Source.from_inferred(item) == source.LosAngeles
    item = 40.496044, -74.443672, 40.561051, -74.332089
    assert Source.from_inferred(item) == source.NewJersey

    assert Source.from_inferred('nyc') == source.NewYorkCity
    assert Source.from_inferred('ny') == source.NewYork
    assert Source.from_inferred('nj') == source.NewJersey
    assert Source.from_inferred('new jersey') == source.NewJersey
    assert Source.from_inferred('la') == source.LosAngeles
    # just Spring Hill returns Spring Hill, Virgnia
    assert Source.from_inferred('Spring Hill, Tennessee') == source.SpringHillTN
    assert Source.from_inferred('va') == source.Virginia

    assert Source.from_inferred('Maywood, California') == source.LosAngeles
    assert Source.from_inferred('Maywood, CA') == source.LosAngeles

    # namibia
    assert Source.from_inferred((-15.49933207, 28.203229539, -15.338660813, 28.358324353)) == None
    # beijing
    assert Source.from_inferred((39.525834067367256, 116.21383162969653, 39.582584, 116.292915)) == None
    # ocean
    assert Source.from_inferred((-37.15612782594927, 64.98947402062927, -37.15612782594927, 64.98947402062927)) == None
    # greenland
    assert Source.from_inferred((73.59343881883807, -51.62165778082543, 73.59343881883807, -51.62165778082543)) == None
    # russia
    assert Source.from_inferred((54.998172689668486, 36.68930259694381, 55.00000000000001, 36.69112990727614)) == None
    # algeria
    assert Source.from_inferred((25.06763435341293, -0.7971811600872423, 25.06763435341293, -0.7971811600872423)) == None
    # gulf of mexico
    assert Source.from_inferred((21.82528963135751, -93.76345422053639, 21.82528963135751, -93.76345422053639)) == None


if __name__ == '__main__':
    test_nominatim()
    test_sources()
