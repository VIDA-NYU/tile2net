# Commandline

## Quick Start

Quickly run this small example to download imagery of Boston Common, segment it, and vectorize it:

```bash
python -m tile2net \
  --location "boston common" \
  --model.bs_val 1 \
  --vector.length 1 \
  --outdir ./boston_common
```

Downloading the imagery, performing segmentation, 
and vectorization should take only a few minutes. At the end, you you will see a printout that should look 
something like this:

```
Output directory      
/home/arstneio/PycharmProjects/tile2net/src/boston_common

Temporary directory   
/tmp/tile2net/bRs07FIb0QB5n-8b

Input imagery         
/home/arstneio/PycharmProjects/tile2net/src/boston_common/ingrid/infile

Segmentation (colored)
/home/arstneio/PycharmProjects/tile2net/src/boston_common/seggrid/colored

Polygons              
/home/arstneio/PycharmProjects/tile2net/src/boston_common/polygons/Polygons-875b6ea8d9381c9b.parquet

Network               
/home/arstneio/PycharmProjects/tile2net/src/boston_common/lines/Lines-875b6ea8d9381c9b.parquet

Polygon preview       
/tmp/tile2net/bRs07FIb0QB5n-8b/polygons/preview/Polygons-875b6ea8d9381c9b.png

Network preview       
/tmp/tile2net/bRs07FIb0QB5n-8b/lines/preview/Lines-875b6ea8d9381c9b.png
```

That's it! You can open these paths to view the respective outputs. 

## Core Arguments

These are the core arguments to quickly get Tile2Net up and running.

### `--location` (`-l`)
- Defines the location for which Tile2Net is run
  - A plain-text place name: `"Washington Square Park"`, or `"Cambridge, MA"`  
  - A bounding box string in **lat, lon** order: `"40.729, -73.999, 40.732, -73.995"`

### `--model.bs_val` (`-b`)
- Batch size per GPU during segmentation
- Increase based on available VRAM
- 16 worked best with 24GB VRAM

### `--vector.length` (`-v`)
- Batch size per core during vectorization
- Increase based on available RAM
- 8 worked best with 64GB RAM

### `--outdir` (`-o`)
- Output directory root
- Defaults to `/tile2net` in the system temp directory
- Relative paths like `./cambridge` are created under the current working directory
- Downloaded imagery is stored in outdir

### `--indir` (`-i`)
- Path to input directory containing imagery tiles on disk
- Tile2Net infers the layout from filenames and subdirectories:
  - `path/to/tiles/z/x/y.png` → interpreted as `{zoom}/{xtile}/{ytile}.png` in the `tiles` folder
  - `path/to/tiles/x_y.png` → interpreted as `{xtile}_{ytile}.png` in the `tiles` folder  
- If not provided, imagery will be fetched automatically from the configured remote source

### `--source` (`-s`)
- Imagery source selector or override.  
- Use this to specify a specific imagery source by name.
---

### Examples

#### Cambridge, MA with 24 GB VRAM and 64 GB RAM
This command allows us to leverage the Massachussets tile server and arguments tailored to 24 GB VRAM and 64 GB RAM 
for an end-to-end pipeline downloading imagery, performing inference, and vectorizing the output into polygons and a 
network of lines. With less RAM, you must reduce `-v` and with less VRAM, you must reduce `-b`. Note that we also set 
`--no-cleanup` to keep the downloaded imagery on disk for the next example.

```bash
python -m tile2net \
  -l "Cambridge, MA" \
  -b 16 \
  -o ./cambridge \
  -v 8 \
  --no-cleanup 
```

#### Local Imagery
To demonstrate the use of local imagery, we can reuse the downloaded imagery with the last step, this time using the 
`-i` argument to indicate the input directory. We'll also explicitly pass the zoom level of our images with `-z`. 

```bash
python -m tile2net \
  -i ./cambridge/ingrid/infile/19/x_y \
  -z 19 \ 
  -l "Cambridge, MA" \
  -b 16 \
  -o ./local \
  -v 8 \
```
