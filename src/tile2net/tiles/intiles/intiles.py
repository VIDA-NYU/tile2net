from .static import static
from .infer import Infer
from .stitch import Stitch
from ..tiles import Tiles

if False:
    import folium
    from .stitched import Stitched


class InTiles(
    Tiles
):
    @Stitch
    def stitch(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:

        # stitch to a target resolution e.g. 2048 ptxels
        self.stitch.to_dimension(...)
        # stitch to a cluster size e.g. 16 tiles
        self.stitch.to_mosaic(...)
        # stitch to an XYZ scale e.g. 17
        self.stitch.to_scale(...)

    @property
    def stitched(self) -> Stitched:
        """If set, will return a Tiles DataFrame with stitched tiles."""
        if 'stitched' not in self.attrs:
            msg = (
                f'Tiles must be stitched using `Tiles.stitch` for '
                f'example `Tiles.stitch.to_resolution(2048)` or '
                f'`Tiles.stitch.to_cluster(16)`'
            )
            raise ValueError(msg)
        result = self.attrs['stitched']
        result.tiles = self
        # result.instance = self
        return result

    @stitched.setter
    def stitched(self, value: Tiles):
        if not isinstance(value, Tiles):
            msg = """Tiles.stitched must be a Tiles object"""
            raise TypeError(msg)
        self.attrs['stitched'] = value
        value.tiles = self

    @Infer
    def infer(self):
        # This code block is just semantic sugar and does not run.
        # Take a look at the following methods which do run:
        result = (
            self.infer
            .with_polygons(
                max_hole_area=dict(
                    road=30,
                    crosswalk=15,
                ),
                grid_size=.001,
            )
            .to_outdir()
        )
        result = self.infer.to_outdir()
