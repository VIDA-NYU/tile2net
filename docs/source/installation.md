# Installation

<!-- In-depth installation details - including environment creation and interaction with high-performance computing clusters -->

```{warning}
Under construction - thank you for your patience!
```

```{contents}
    :local:
```

## Requirements

In order to run Tile2Net, you must have the following on your device:

- Hardware: 1 CUDA-enabled GPU for inference
- Operating System: Linux, Windows, or MacOS
- Software:  ***CUDA==11.7, Python==3.10.9, pytorch==2.0.0***

## Step 1: Clone the Repository

You can clone the repository by running the following commands in a command line:
```
git clone git@github.com:VIDA-NYU/tile2net.git
cd tile2net
```

## Step 2: Activate a Virtual Environment

It is highly recommended to create a virtual environment using either pip or conda to install Tile2Net and its
dependencies. The following commands create an environment with conda and python version 3.11

```
conda create --name venv python=3.11
conda activate venv
```

## Step 3: Install Tile2Net Dependencies

```
python -m pip install -e .
```

```{note}
The period at the end is important! This installs every library in the [requirements-dev.txt](https://github.com/VIDA-NYU/tile2net/blob/main/requirements-dev.txt) file
```

## Step 4: Start Generating a Network!


Tile2Net comes with a small shell script, which will prompt the user for a path where the project should be created and saved. It will then download the tiles corresponding to Boston Commons and Public Garden, creates larger tiles (stitched together) for inference, run inference, create the polygon and network of this region. The sample area is small, just so you can test your environment settings and GPU, and see what to look for.

```
bash ./examples/example.sh 
```

```{note}
When the script asks you for an output directory, you **must** provide the absolute path!
```

## What if I don't have a GPU on my computer?

That's ok! You have a few other options for using a GPU, although they come with their own intricacies.

### Connecting to Google Colab

Google Colab offers free and paid Jupyter Notebook services with free access to GPU's and other computing resources. As of December 2023, a free account affords you access to a T4 GPU and a limited number of compute units. 

```{warning}
- If you reach zero compute units, you will no longer be able to access a GPU until you are allotted more units or you buy a Colab Pro account.
- Session storage is connected to an individual runtime, and it does *not* persist across runtimes. If you want to save the outputs of Tile2Net, you will need either mount your google drive to the notebopk or download the outputs locally to your computer
```

For an example of a Tile2Net workflow using Google Colab, see the {doc}`./notebooks/colab_example` page.

### Connecting to a High Performance Cluster (HPC)

## Troubleshooting Tips

### Error 1

### Error 2

### Error 3