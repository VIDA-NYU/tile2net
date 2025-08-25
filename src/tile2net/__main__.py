# from tile2net.tileseg.raster import Raster
# import argh
# from tile2net.tileseg.generate.generate import generate
# from tile2net.tileseg.inference.inference import inference
#
# argh.dispatch_commands([
#     generate,
#     inference,
# ])

# src/tile2net/__main__.py
from runpy import run_module

if __name__ == '__main__':
    # delegate -m tile2net to -m tile2net.grid
    run_module(
        'tile2net.grid',
        run_name='__main__'
    )
