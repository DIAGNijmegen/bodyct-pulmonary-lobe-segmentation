## Background

This is the implementation of the paper [Relational Modeling for Robust and Efficient Pulmonary Lobe Segmentation in CT Scans](https://arxiv.org/pdf/2004.07443.pdf) published in IEEE transaction on Medical Imaging. 
We first trained our model on 4000 Inpiration CT scans from COPDGene, and then retrained our model on 104 CT scans with COVID-19. The `best.ckp` is the model after training on 104 CT scans with COVID-19. 


## Table Of Contents
- [Usage](#usage)
- [Notice](#notice)

### Usage
 - Please check `/dockerFile` for the required system and python packages to install.
 - To build a docker image for this algorithm, `cd` into `./`, and run `docker build --tag=gclobe .`.
 - To run this algorithm on a test image, execute `/test.py` by supplying the input and output folders in commandline arguments.
 
### Notice
We resample the input scan at the first stage to 256x256x256 such that large CT scans can be fitted into memory. This is different from that mentioned in the paper to resample input scans to an isotropic spacing at the first stage.
 