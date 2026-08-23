__all__ = ['generate', 'Namespace']

import argh
from typing import Any

from tile2net.raster.generate.commandline import commandline, Namespace
from tile2net.raster.weights import ensure_weights


def _raster_from_info(info: dict[str, Any]) -> Any:
    """Construct Raster lazily so importing the CLI does not load GeoPandas."""
    from tile2net.raster.raster import Raster

    return Raster.from_info(info)


@commandline
def generate(args: Namespace) -> str:
    """Generate a JSON file representing the tile2net project file structure."""
    raster = _raster_from_info(args.__dict__)
    ensure_weights()
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
