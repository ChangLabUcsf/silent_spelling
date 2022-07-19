# Generalizable spelling using a speech neuroprosthesis in an individual with severe limb and vocal paralysis

This repository contains code to train and test classification and
speech-detection models that decode silent-speech attempts from cortical activity
recorded from a paralyzed person to enable real-time spelling.
The code for
classification/spelling and speech detection are separated into the `classify_and_spell`
and `speech_detection` folders, each with an additional `README.md` file for
specific notes. 

The code to recreate figures from this paper is in the `figures` folder, with a `README.md` file  with further instructions. 

## Setting up the Python environment

Our code contains several dependencies and we recommend create a virtual 
environment, either through Anaconda (detailed below) or through Python's 
`venv`.

### Installing the base environment

1. Download the Anaconda
   3 [installer](https://www.anaconda.com/products/individual)
   for your OS (we recommend at least version 4.3.3) and follow the
   instructions [here](https://docs.anaconda.com/anaconda/install/windows/).
2. At the command line, create a fresh environment with Python 3.6.6: 

    ```bash
    $ conda create -n silent_spelling python=3.6.6
    ```
   This may take a few minutes (~1 minute on a MacBook Pro).
3. Activate this environment, navigate to the folder containing this README
   file, and install the other dependencies with the following commands
   (`chmod` ensures that the bash script for installing the other
   dependencies is executable):

   ```bash
   $ source activate silent_spelling
   $ cd silent_spelling
   $ chmod +x install_dependencies.sh
   $ ./install_dependencies.sh
   ```

### Installing PyTorch and using GPUs
Although the environment setup may have installed PyTorch, these
steps should be taken to install PyTorch 1.6.0 (which we require).
This installation depends on what operating system is being used and
should overwrite any existing PyTorch installation in your environment
that may have been setup in the preceding steps.

1. Activate the environment:
   ```bash
   $ source activate silent_spelling
   ```
2. Installation options:
   1. CPU-only on a Mac:
      ```bash
      $ pip install torch==1.6.0 torchvision==0.7.0
      ```
   2. CPU-only on Linux
      ```bash
      $ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
      ```
   3. Using CUDA 10.1 on Linux
      ```bash
      $ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
      ```

For other installation options for PyTorch 1.6.0, check the
[PyTorch site](https://pytorch.org/get-started/previous-versions/#v160).

## Running the example code

Refer to the `README.md` file within each sub-folder
for instructions on how to run the provided training and evaluation code (either `classify_and_spell` or `speech_detection`) or the code to generate the figures in the manuscript (`figures`).

## Source data
Source data for re-creating the manuscript figures is in the `source_data` folder
Consult `figures/README.md` for further details.


## Tested systems
This code has been tested on:
* MacBook Pro with macOS 10.15.7, CPU-only
* Linux Ubuntu 20.04, CPU-only
* Linux Ubuntu 20.04, with Tesla V100 GPUs and CUDA 10.1

## Terms of use
All rights reserved. 
Any material contained in this folder should only be used for the purpose of 
reviewing the associated manuscript. All or part of this material should not 
be duplicated or distributed in any form. After a decision is made on this 
manuscript, this material should be completely erased from any machine that 
this material was downloaded to.

&copy; 2019-2022 Sean L. Metzger, Jessie R. Liu, and David A. Moses All 
Rights Reserved.
