# TMLKD

This is the code of **TMLKD**. 

## Require Packages

Pytorch, Numpy, traj_dist, einops.

## Running Procedures

### Download Files

Due to the restriction of github platform, please download [the dataset and trained models](https://drive.google.com/file/d/1icbJl0d31T3qqCvHAr4ef9GfiV6iYRQA/view?usp=sharing) and put it into your data-folder, which can be set in `config.py`.

### Create Folders

In your data-folder: 
1. Create a folder `model` to place the model of each training epoch. 
2. Create a folder `feature` to place your input data and ground truth..

### Training & Evaluating

To train teacher models:

1. Check your config in `config.py`.
2. Run `train.py` to train the model.
3. Edit the mode in `config.py` into `output`.
4. Run `test_output` to derive the embeddings.
5. Copy the derived embedding results into the folder of student training project.

To train student models:
1. Check your config in `config.py`. 
2. Run `get_teacher_result.py` to derive enriched labels. 
3. Run `train.py` to train the model.