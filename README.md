# TMLKD

This is the code of **TMLKD**. 

## Require Packages

Pytorch==1.17.0, Numpy, traj_dist, einops.

## Running Procedures

### Download Files

Due to the restriction of github platform, please download the [ground truth file](https://drive.google.com/file/d/1hdRsoCIgyXDcz3oiHHhLVkLhQJqPXHyb/view?usp=sharing) and put it into your data-folder, which can be set in `config.py`.
 
### Create Folders

In your data-folder: 
1. Create a folder `model` to place the model of each training epoch. 
2. Create a folder `feature` to place your input data and ground truth.

In your Teacher and Student folder:
1. Create a folder `teacher_predict_result`. 
2. In the folder you created, create a folder named with your target measure, such as `/Teacher/teacher_predict_result/hausdorff`.

### Training & Evaluating

To train teacher models:

1. Check your config in `config.py`.
2. Run `train.py` to train the model.
3. Edit the mode in `config.py` into `output`.
4. Run `test_output` to derive the embeddings.
5. Copy the derived embedding results into the folder of student training project.

To train student models:
1. Make sure that the embeddings of teacher models has been placed into the folder `/Student/teacher_predict_result/distance_type`.
2. Check your config in `config.py`. 
3. Run `get_teacher_result.py` to derive enriched labels. 
4. Run `train.py` to train the model.