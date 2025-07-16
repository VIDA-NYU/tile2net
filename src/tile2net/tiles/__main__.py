from tile2net.tiles.tiles.tiles import Tiles
from tile2net.tiles.cfg import Cfg

parser = Tiles.cfg._parser
args = parser.parse_args()
cfg = vars(args)
cfg = Cfg(cfg)
