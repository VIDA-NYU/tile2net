__all__ = ['generate', 'Namespace']

import argh

from tile2net.raster import util
import json
import sys

from tile2net.raster.generate.commandline import commandline, Namespace
from tile2net.raster.raster import Raster


@commandline
def generate(args: Namespace) -> str:
    """Generate a JSON file representing the tile2net project file structure."""
    raster = Raster.from_info(args.__dict__)
    raster.generate(args.stitch_step)
    # raster.save_info_json(new_tstep=args.stitch_step)
    # json.dump(
    #     dict(raster.project.structure),
    #     fp=sys.stdout,
    #     allow_nan=False,
    #     indent=4,
    # )

if __name__ == '__main__':
    argh.dispatch_command(generate)

