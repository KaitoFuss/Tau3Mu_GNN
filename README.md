# SVJ GNN

This package eases the processing of hadronic jets stored in NTuples into graphs and train GNNs of any flavor. Spacial (à la ParticleNET) and Lund graph (usually better) representation of hadronic jets are supported.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Prepare your venv (exchange all instances of `mamba` with `conda` if that is your jam):
```
mamba create -n gnn python=3.11
mamba install numpy, scipy, matplotlib, tqdm

# If you run on CPU
mamba install pytorch torchvision torchaudio cpuonly -c pytorch

# If you have a CUDA-enabled GPU
# mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install pyg -c pyg
mamba install vector, uproot, awkward, networkx
```

Clone this repo:
```
git clone git@github.com:rseidita/SVJ_GNN.git
```

## Usage

A complete example is given in the aptly named "example" notebook. Data loading and processing, model definition and training are covered.

## Contact

roberto.seidita@cern.ch
