# Data Preparation Guide

<!-- In-depth user guide - including information from DATA_PREPARE.md -->

```{warning}
Under Construction - thank you for your patience!
```

```{contents}
    :local:
```

If you are preparing your own tile images to use with `tile2net`, follow these data requirements.

## Tile Images

Tile images are raster tiles delivered as image files, typically in PNG format, and they serve as the primary input for
Tile2Net.
All subsequent processes rely on these files.

### Tile Image Size

Tile images are typically square-shaped, with the width and height of each tile being identical.
The size of the tile is represented by the number of pixels in the width and height, displayed as "{number of pixels in
width} x {number of pixels in height}". In the Slippy Tile Map system, the base size is 256 x 256 pixels, and Tile2Net
follows this standard.

When geolocating tiles and corresponding polygons and networks, all input tiles must have the same size and cannot be
mixed and matched.
For example, 256x256 tiles cannot be mixed with 512x512 tiles.
However, users have the option to stitch tiles of the same size into larger tiles and save them in a new folder using
the `Raster.stitch` method.

### Tile Image Format

Tile2Net only supports the PNG image format. More formats may be supported in upcoming versions.  
If tile images that you already have are in TIFF format, they need to be converted to XYZ in PNG.
To do that you can use QGIS and the “Generate XYZ Tiles (Directory)” tool to do the conversion.
Once that’s created, you can pass its path implicating the format of the files, e.g. path/to/tiles/z/x/y.ext
The implicit path can contain the zoom level, e.g. path/to/tiles/z/x/y.ext, or it can be like path/to/tiles/x_y.ext. 

To see an example, check out our [jupyter notebook](examples/inference.ipynb). 

<!-- TODO:FIX THIS LINK -->

### Tile Transparency

PNG has different types that some might have Alpha channel (usually PNG 32-bit).
That should be avoided and tiles images should have no transparency when used in the project.
Only the first three channels are loaded from a PNG.

### Tile Image Position

The position of each tile in slippy tile system is represented by the X and Y coordinates of its top left corner in
slippy grid.
X represents the horizontal position, and Y represents the vertical position in that grid. Note: This is **not** the
same grid as used in Tile2Net project. Though, Tile2Net grid is a subset of the slippy grid that matched the area of
interest. (Slippy grid represents all the tiles in the world in that specific zoom level)

### Tile Image Names

Tile names are defined as `{X_tile}_{Y_tile}.{file_format}`.
The X and Y coordinates are the position of the top left corner of the image (tile image position).
A tile image file name could look like this: “154374_197085PNG”. (see tile position for more info).
Although these numbers are based on the zoom level and number of tiles there is a mathematical way to convert them into
geographic coordinates. 