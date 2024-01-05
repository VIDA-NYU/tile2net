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
- Session storage is connected to an individual runtime, and it does *not* persist across runtimes. If you want to save the outputs of Tile2Net, you will need either mount your google drive to the notebook or download the outputs locally to your computer
```

For an example of a Tile2Net workflow using Google Colab, see the {doc}`./notebooks/colab_example` page.

### Connecting to a High Performance Cluster (HPC)

HPCs are groups of interconnected computers that work together to perform complex computations, often used for large-scale data analysis and machine learning tasks. They use parallel processing to divide tasks among multiple nodes, significantly speeding up computation time.

Institutions such as universities often offer access to HPCs. To connect to an HPC, you typically use remote access protocols like SSH (Secure Shell) to log in to the cluster's head node, which serves as the entry point. From there, you can submit your computational jobs to the cluster's job scheduler, which manages the allocation of resources and schedules the execution of tasks across the available nodes.

HPC architectures and job schedulers can vary widely, so you should refer to your particular cluster's documentation for a detailed walkthrough. Below is a sample pseudo-workflow running Tile2Net with a linux-based HPC that uses the slurm job scheduler.


```{warning}
- Before starting, make sure your chosen HPC supports these hardware and software [requirements](#requirements)! Some HPCs are not linux-based, and others may not support a high enough CUDA GPU. 
```

#### Step 1: Creating your environment

You may need to load some modules including conda and CUDA before installing Tile2Net. In the login node:

```
module load <anaconda module>
module load <cuda module>
```
```{note}
You can use `module avail` to see which modules you can load!
```

Then, create your conda environment as you normally would, and install all of Tile2Net's dependencies

```
conda create --name testenv python=3.11
python -m pip install -e .
```

#### Step 2: Request a compute node

```{warning}
deactivate your current environment at this point with `conda deactivate` before activating your new one, to reduce the chances of environment issues.
```

Access a compute node using `srun`. The line below uses slurm to request 1 node with 1 gpu called `gpu:volta` from the partition named `partition_name`, and then runs a bash command line in that node

```
srun -N 1 --gres=gpu:volta:1 --partition=partition_name --pty bash
```

you should now be in a compute node. 

#### Step 3: Run Tile2Net!

You're ready to go! Now, just activate your environment and start using the command line Tile2Net tools, as described on the {doc}`./notebooks/quick_start_command_line` page. 
```
source activate testenv
bash ./examples/example.sh
```

```{warning}
Some clusters give their compute nodes internet access, while others don't. If your compute node does not have internet access, you will need to either download the tiles you want to process in the head node, or download them locally and copy them to your HPC environment using `scp`. Ask your HPC administrators about maximum computing resources in the login node to determine which path is more viable for you. See the {doc}`./notebooks/quick_start_command_line` for an example on how to use custom tiles once you have those downloaded. 
```