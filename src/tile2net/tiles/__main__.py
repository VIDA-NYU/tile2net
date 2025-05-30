from .tiles import Tiles
parser = Tiles.cfg._parser
args = parser.parse_args()



latlon = [40.722597714156876, -74.00218007670077, 40.75335518538804, -73.97469176766572]
tiles = (
    Tiles
    .from_bounds(latlon=latlon, zoom=19)
    .with_source()
    .stitch.to_mosaic(16)
    .with_cfg(args.)
    # .infer.to_outdir(...)
)
tiles.cfg._parser
