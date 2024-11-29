# Data path
mode = 'train'  # train(training model) / output(deriving embeddings from trained models)

GPU = '0'
distance_type = 'hausdorff'
# Initial is 's' -> training for target measure(student model)
# Initial is 't' -> training source measure(teacher model)
data_name = 'troma' if mode == 'train' else 'sroma'
base_model = 'T3S'  # T3S or NeuTraj

if data_name[0] == 't':
    full_datalength = 15000
else:
    full_datalength = 10000

data_folder = '/hy-tmp'
corrdatapath = f"{data_folder}/features/{data_name}_traj_coord"
griddatapath = f"{data_folder}/features/{data_name}_traj_grid"
real_distancepath = f'{data_folder}/features/{data_name}_{distance_type}_distance_all_{full_datalength}'

source_distance = ['discret_frechet', 'hausdorff', 'erp', 'dtw']    # you can select from these measures, also, you can add new teachers.
if distance_type in source_distance:
    source_distance.remove(distance_type)
source_distance_paths = {measure: f"{data_folder}/features/{data_name}_{measure}_distance_all_{full_datalength}"
                         for measure in source_distance}

incell = False

gird_size = [1100, 1100]
val_size = 25
max_len = 250

epochs = 50
batch_size = 40
sampling_num = 20
k = 15

learning_rate = 0.0005

mail_pre_degrees = {
    "dtw": 16,
    "erp": 16,
    "discret_frechet": 8,
    "hausdorff": 8
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

train_idx = (0, 3000)
val_idx = (3000, 4500)
test_idx = (4500, 15000)


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
              'incell = {}'.format(incell)
    return configs


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format(0, 1))
    print(config_to_str())
    print('=' * 30)
