from typing import Optional

import argh
from argh import arg

# from tile2net.raster.util import compose_left
from toolz import compose_left

commandline = compose_left(
    argh.expects_obj,
    arg(
        '--name', '-n', type=str, required=True,
        help='The name of the project; this will define the project subdirectory',
    ),
    arg(
        '--location', '-l', type=str, required=True,
        help='An address, e.g. "Washington Square Park, NYC, NY", '
             'or a bounding box, e.g. "40.730610,-73.935242,40.732582,-73.931057"'
    ),
    arg(
        '--input', '-i', type=str, dest='input_dir',
        help='The path to the input directory, implicating the format of the files, '
             'containing the xtile, ytile, and extension, and possibly containing the zoom level, '
             'e.g. path/to/tiles/z/x/y.ext or path/to/tiles/x_y.ext'
    ),
    arg(
        '--output', '-o', type=str, dest='output_dir',
        help='The path to the output directory; "~/tmp/tile2net" by default'
    ),
    arg(
        '--num_class', help='The number of classes', default=4, type=int
    ),
    arg(
        '--base_tilesize', help='The base tile size', default=256, type=int
    ),
    arg(
        '--zoom', '-z', help='The slippy zoom level', default=19, type=int)
    ,
    arg(
        '--crs', help='The coordinate reference system', default=4326, type=int
    ),
    arg(
        '--boundary_path', default=None, type=str
    ),
    arg(
        '--nopadding', action='store_false', dest='padding'
    ),
    arg(
        '--extension', default='png', type=str
    ),
    arg(
        '--tile_step', default=1, type=int, help=(
            'The integer length, in slippy tiles, of each tile'
        )
    ),
    arg(
        '--stitch_step', '-st',
        help='The amount of tiles that an output file from the semantic segmentation will represent',
        default=4, type=int
    ),
    arg(
        '--quiet', '-q', action='store_true', default=False, dest='quiet'
    ),
    arg(
        '--source', '-s', default=None, type=str,
    ),
    arg(
        '--dump_percent',
        type=int,
        default=0,
        help='The percentage of segmentation results to save. 100 means all, 0 means none.',
    ),
    arg(
        '--debug' '-d', action='store_true', default=False, dest='debug'
    )
)

class Namespace(argh.ArghNamespace):
    name: str
    location: str | list[float]
    input_dir: Optional[str]
    output_dir: Optional[str]
    num_class: int
    base_tilesize: int
    zoom: int
    crs: int
    boundary_path: Optional[str]
    padding: bool
    extension: str
    tile_step: int
    stitch_step: int
    quiet: bool
    source: str
