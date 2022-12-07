# Dataset for training a CNN to estimate DMI strength from simulated magnetometry experiments

This repository holds a dataset of images of FORCs taken from simulated materials along with the labels of their corresponding material parameters.

## Requirements
Linux version?

The training script, main.py, was run and tested with the following versions of python and modules:
- Python 3.8.10
- cuda 11.4.2
- argparse 1.1
- torch 1.10.1+cu113
- numpy 1.21.2
- scipy 1.7.3

Installation:
pip3 install -r requirements.txt
Please config the CUDA follow the instruction on https://pytorch.org/get-started/locally/
## Description of Data files
- images.bin
  - 20,000 61x61 grayscale images that encode the simulated FORCs with random material parameters
- labels.bin
  - 20,000 8-parameter labels that correspond to the materials that produced the images in images.bin

## Description of code
[Main.py](main.py) will extract the data from the [images](images.bin) file and the [labels](labels.bin) file, create a customized neural network, and train that neural network on a portion the images and labels using tuned hyperparameters. It will save the results of the training after each epoch, the performance of the best version of the CNN encountered during training, and the full set of learnable parameters for the CNN at its peak state.


***All the imported tools need to be described in this readme with version number including python version***

import random

from math import *

import struct

import time
