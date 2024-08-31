# TMLKD

This is the code of **TMLKD**. 

> Some code of T3S is under organization. We will update the code before publication. 

## Require Packages

Pytorch, Numpy, traj_dist, einops

> For implementing NeuTraj, you may need to install pytorch 1.5.

## Runing Procedures

## Download Files

Due to the restriction of github, please download the [ground truth file](https://drive.google.com/file/d/1XyZNGGjXy-TDDQ6XvbjeXk5tIPCx9Qm7/view?usp=drive_link) and put it into `./features/`.
 
## Create Folders

Please create a folder `model` to place the moel of each training epoch. 

### Training & Evaluating

An example bash file for running the code can be seen in `run_the_code.sh`. 

The parameters can be modified in /tools/config.py
