# cuda

CUDA projects portfolio

## Installation

CUDA can be installed in multiple ways, here we follow [these steps](https://twitter.com/jeremyphoward/status/1697435241152127369):

1. Install GPU drivers (verify with `nvidia-smi`).
2. Install [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) (and optionally mamba `conda install mamba -c conda-forge`)
3. Create a new environment `conda create -n cuda` and activate it `conda activate cuda`.
4. Install CUDA `conda install cuda cudnn -c nvidia/label/cuda-12.1.0`
5. Verify the installation with `nvcc --version`
6. If required, install pytorch with `conda install 'pytorch>2.0.1' torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia/label/cuda-12.1.0`
7. If required to build other packages, cuda path is in `$CONDA_PREFIX`.

## Learning

- [hello_cuda](hello_cuda/README.md): A simple CUDA project to get started with CUDA programming.

## Projects

TODO