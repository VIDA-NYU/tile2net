# Prepare your Project

To create the sidewalk networks for your area of interest, please ensure the following requirements are met:

1. You have access to at least one CUDA-enabled GPU.
2. You know the address or the latitude and longitude coordinates (EPSG 4326) of the bounding box of your region.
3. You either have your own source of high-resolution orthorectified aerial imagery (zoom level 19 and above is
   preferred, and not less than 18), or your area of interest lies within the regions currently supported by Tile2Net so you can use Tile2Net
   to download the tiles.

**Note:** If you are using your own imagery, please make sure that your data is in accordance with the data requirement
detailed [here](https://github.com/VIDA-NYU/tile2net/blob/main/DATA_PREPARE.md).

Please note that the model may need additional training if the urban fabric/built environment of the region you want to
try it on is very different from the US context. However, the required training set will be significantly smaller than
what would otherwise be needed. We are planning to also provide a detailed tutorial about training the model on your
data, but at this moment, we only support inference.

### Supported Regions 

| Country | State/Province | Entire State/ Province | County Data | City/Town Data  |
|---------|----------------|-----------------------|-------------|------------------|
| USA     | CA             | FALSE                 | Alameda     | LA               |
| USA     | DC             | TRUE                  |             | DC               |
| USA     | MA             | TRUE                  |             |                  |
| USA     | NY             | TRUE                  |             | NYC (higher res) |
| USA     | NJ             | TRUE                  |             |                  |
| USA     | OR             | TRUE                  |             |                  |
| USA     | TN             | FALSE                 |             | Spring Hill      |
| USA     | VA             | TRUE                  |             |                  |
| USA     | WA             | FALSE                 | KING        |                  |


## Basic Concepts:

Loading and working with imagery data all at once is a computationally expensive process.
To avoid this issue, the common practice is to divide the large images into subset of smaller images (tiles) that can be
handled more efficiently (tile images).
When it comes to using tile images in Tile2Net there are two options:

1) Using Tile2Net built-in functions to download the tile images (Recommended)
   It will be handled automatically, and the patterns are as described in the second part for compatibility.
2) Bringing already downloaded tile images into the project. That might require some modification to match the project
   naming convention.

With that in mind, see the main elements of the project below:

### Grid

Put simply, a grid is an area of interest that consists of a number of square cells. Each cell within the grid is
represents an image tile,
with the minimum size of 256x256 pixels being the default. The dimensions of the grid (i.e., its width and height)
are reported in terms of the number of tiles that span across its columns and rows. These dimensions are determined by
the region of interest
that the user specifies. The grid is then automatically generated based on the bounding box specified by the user.

### Bounding Box

A bounding box defines the region of interest by specifying the coordinates of the top-left and bottom-right corners of
a grid box. It can be passed two pairs of latitude and longitude coordinates in the following format: 

```[latitude, longitude, latitude, longitude]```

For example, the bounding box for New York City is:

```[40.91763, -74.258843, 40.476578, -73.700233]```

Alternatively, you can pass a textual address which will be automatically geolocated. The address should follow this
format:

```{region}, {city}, {state}, {country}```

For instance:

```'Washington Square Park, NYC, NY, USA'```

For more information on geolocating, check out the [geopy](https://geopy.readthedocs.io/en/stable/#module-geopy.geocoders) library.

### Tiles

Tile images are raster tiles delivered as image files, typically in PNG format, and they serve as the primary input for
Tile2Net.
All subsequent processes rely on these files. To learn more about the specific tile image requirements, please read
our [data preparation guide](https://github.com/VIDA-NYU/tile2net/blob/main/DATA_PREPARE.md).

### Zoom Level

Basically, you can think of zoom level 0 as fitting the map of the whole earth in a single tile. For each zoom level,
the number of tiles to cover the whole earth could be calculated as 2^2*zoom_level. As of now, zoom levels are defined
as integers from 0 to 23, but all of them might not be available in all the mapping services.

Assuming the number of pixels in the tile Images are always 256x256, the higher the zoom level goes, the more detail can
be seen on the ground. That means when looking at a tile image at zoom level 0, each pixel in it represents
approximately 156,500 meters on the ground, while at zoom level 19, each pixel would be approximately 30 cm on the
ground. The relation between zoom level and number of pixels and resolution should be well understood. For more
information, check out the table below.

| Zoom Level (Z) | Number of Tiles to Cover Entire Earth 2^2*Z | Approximate Resolution (meters/pixel) |
|:--------------:|:-------------------------------------------:|:-------------------------------------:|
|       0        |                      1                      |                156,543                |
|       1        |                      4                      |                78,272                 |
|       2        |                     16                      |                39,136                 |
|       3        |                     64                      |                19,568                 |
|       4        |                     256                     |                 9,784                 |
|       5        |                    1,024                    |                 4,892                 |
|       6        |                    4,096                    |                 2,446                 |
|       7        |                   16,384                    |                 1,223                 |
|       8        |                   65,536                    |                  611                  |
|       9        |                   262,144                   |                  306                  |
|       10       |                  1,048,576                  |                  153                  |
|       11       |                  4,194,304                  |                 76.4                  |
|       12       |                 16,777,216                  |                 38.2                  |
|       13       |                 67,108,864                  |                 19.1                  |
|       14       |                 268,435,456                 |                 9.55                  |
|       15       |                1,073,741,824                |                 4.78                  |
|       16       |                4,294,967,296                |                 2.39                  |
|       17       |               17,179,869,184                |                 1.19                  |
|       18       |               68,719,476,736                |                 0.60                  |
|       19       |               274,877,906,944               |                 0.30                  |
|       20       |              1,099,511,627,776              |                 0.15                  |
|       21       |              4,398,046,511,104              |                 0.07                  |
|       22       |             17,592,186,044,416              |                 0.04                  |
|       23       |             70,368,744,177,664              |                 0.02                  |

**************************************************************************************************************

### Folder Structure

Folder structure gets created automatically and if any changes are made, they Should match the file and folder
structure:
For tiles
`…\{Project_name}\tiles\static\{size_pixels}_{zoom_level}`
`…\NYC_project\tiles\static\256_19`



