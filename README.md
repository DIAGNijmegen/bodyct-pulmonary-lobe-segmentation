## Background

This is the implementation of the paper [Relational Modeling for Robust and Efficient Pulmonary Lobe Segmentation in CT Scans](https://arxiv.org/pdf/2004.07443.pdf) published in IEEE transaction on Medical Imaging. 


## Table Of Contents
- [Usage](#usage)


### Usage
 - Please check `/dockerFile` for the required system and python packages to install.
 - To build a docker image for this algorithm, `cd` into `./`, and run `docker build --tag=gclobe .`.
 - To run this algorithm on a test image, execute `/test.py` by supplying the input and output folders in commandline arguments.
 