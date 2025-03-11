import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--distance_type", type=str, default='dtw')
parser.add_argument("--data_name", type=str, default='troma')
parser.add_argument("--GPU", type=str, default='0')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--base_model", type=str, default='T3S')

args = parser.parse_args()
distance_type = args.distance_type
data_name = args.data_name
GPU = args.GPU
mode = args.mode
base_model = args.base_model

if mode == 'output':
    data_name = 's' + data_name[1:]

if data_name[0] == 't':
    full_datalength = 15000
else:
    full_datalength = 10000

data_folder = '/hy-tmp'
corrdatapath = f"{data_folder}/features/{data_name}_traj_coord"
griddatapath = f"{data_folder}/features/{data_name}_traj_grid"
real_distancepath = f'{data_folder}/features/{data_name}_{distance_type}_distance_all_{full_datalength}'

save_tag = f"_{base_model}"  # to differentiate different models

# you can select from these measures, also, you can add new teachers.
source_distance = ['discret_frechet', 'hausdorff', 'dtw']
if distance_type in source_distance:
    source_distance.remove(distance_type)
source_distance_paths = {measure: f"{data_folder}/features/{data_name}_{measure}_distance_all_{full_datalength}"
                         for measure in source_distance}

stard_LSTM = (base_model == 'ntnosam')
incell = True

gird_size = [1100, 1100]
val_size = 25
max_len = 250

epochs = 50
if base_model == 'T3S':
    learning_rate = 0.001
    batch_size = 20
elif base_model == 'ntnosam':
    learning_rate = 0.01
    batch_size = 50
else:
    raise NotImplementedError

sampling_num = 30
early_stopping_patience = 10
train_idx = (0, 3000)
val_idx = (3000, 4500)
test_idx = (4500, 15000)

diff_loss_weight = 1
mail_pre_degrees = {
    "dtw": 16,
    "discret_frechet": 16,
    "hausdorff": 16,
}

# Test Config
em_batch = 250
test_num = 1000

# Model Parameters
d = 128

recurrent_unit = 'GRU'  # GRU, LSTM or SimpleRNN
spatial_width = 2

m = 50
grid_size = [1100, 1100]


def config_to_str():
    configs = 'learning_rate = {} '.format(learning_rate) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datapath = {} '.format(corrdatapath) + '\n' + \
              'datatype = {} '.format(data_name) + '\n' + \
              'corrdatapath = {} '.format(corrdatapath) + '\n' + \
              'distancepath = {} '.format(real_distancepath) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'diff_loss_weight = {} '.format(diff_loss_weight) + '\n' + \
              'incell = {}'.format(incell)
    return configs


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format(0, 1))
    print(config_to_str())
    print('=' * 30)
