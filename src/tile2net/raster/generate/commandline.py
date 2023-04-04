import argparse
from typing import Optional

import argh
from toolz import compose_left
from tile2net.raster.project import Project

# arg = argparse.ArgumentParser().add_argument
# globals()['arg'] = argh.arg
import argh
from argh import arg


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
        help='The path to the input directory, if using local tiles.',
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
        '--tile_step', help='The step size of the tiles', default=1, type=int
    ),
    arg(
        '--stitch_step', '-st',
        help='The step size of the tiles stitched for semantic_segmentation',
        default=4, type=int
    ),
    arg(
        '--quiet', '-q', action='store_true', default=False, dest='quiet'
    ),
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
