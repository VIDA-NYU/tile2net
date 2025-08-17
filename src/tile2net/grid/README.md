# Commandline

## Quick Start

You can likely run this right now:

```bash
python -m tile2net \
  --location "boston common" \
  --model.bs_val 2 \
  --vector.length 4 \
  --output ./boston_common
```

Downloading the weights will take some time. Then, downloading the imagery, performing segmentation, 
and vectorization should take only a few minutes. 

## Basic Functionality

The commandline interface allows you to configure all aspects of training,
segmentation, and vectorization directly via `--namespace.option=value` flags.

For example:


```bash
python -m tile2net \
    --location "Cambridge, MA" \
    --model.bs_val 4 \
    --vector.length 4 \
    --output ./cambridge
```
