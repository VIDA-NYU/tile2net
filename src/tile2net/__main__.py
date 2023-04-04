from tile2net.raster.raster import Raster
import argh
from tile2net.raster.generate import generate
from tile2net.tileseg.inference import inference
from tile2net.tileseg.train import train

argh.dispatch_commands([
    generate,
    inference,
    train,
])
