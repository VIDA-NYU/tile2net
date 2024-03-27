from tile2net.raster.raster import Raster
import argh
from tile2net.raster.generate.generate import generate
from tile2net.tileseg.inference.inference import inference

argh.dispatch_commands([
    generate,
    inference,
])
