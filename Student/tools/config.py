import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--distance_type", type=str, default='dtw')
parser.add_argument("--data_name", type=str, default='sroma')
parser.add_argument('--GPU', type=str, default='0')
parser.add_argument('--base_model', type=str, default='T3S')

args = parser.parse_args()
distance_type = args.distance_type
GPU = args.GPU
data_name = args.data_name
base_model = args.base_model

prob_rho = 7
weak_threshold = 0.01
rank_loss_weight = 0.000001
latent_loss_weight = 1
d = 128
teacher_flag = f"_{base_model}"

# you can select from these measures, also, you can add new teachers.
source_distance = ['discret_frechet', 'hausdorff', 'dtw']

if base_model == 'T3S':
    learning_rate = 0.00005
    batch_size = 5
elif base_model == 'ntnosam':
    learning_rate = 0.01
    batch_size = 20
else:
    raise NotImplementError

full_datalength = 10000

epochs = 1
sampling_num = 5
train_size = 200
val_size = 50
max_len = 250
train_trajectory_size = 3000
incell = True

data_folder = '/hy-tmp'
teacher_folder = '../Teacher'
corrdatapath = f"{data_folder}/features/{data_name}_traj_coord"
griddatapath = f"{data_folder}/features/{data_name}_traj_grid"
real_distancepath = f"{data_folder}/features/{data_name}_{distance_type}_distance_all_{full_datalength}"
distance_info_path = f"{data_folder}/features/{data_name}_{distance_type}_cluster_data_fake"
teacher_rank = f"{teacher_folder}/teacher_predict_result{teacher_flag}/{distance_type}/{data_name}_enriched_ranks_{prob_rho}"
share_embeddings = f"{teacher_folder}/teacher_predict_result{teacher_flag}/{distance_type}/{data_name}_share_embeddings"

# Test Config
em_batch = 1000
test_num = 1000

recurrent_unit = 'GRU'  # GRU, LSTM or SimpleRNN
spatial_width = 2

m = 50
grid_size = [1100, 1100]

teacher_dim = 128

save_tag = f'{teacher_flag}'

def config_to_str():
    configs = 'd = {}'.format(d) + '\n' + \
              'learning_rate = {} '.format(learning_rate) + '\n' + \
              'train_nums = {} '.format(train_size) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datapath = {} '.format(corrdatapath) + '\n' + \
              'datatype = {} '.format(data_name) + '\n' + \
              'corrdatapath = {} '.format(corrdatapath) + '\n' + \
              'distancepath = {} '.format(real_distancepath) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'base_model = {} '.format(base_model) + '\n' + \
              'incell = {}'.format(incell) + '\n' + \
              'teacher_rank = {}'.format(teacher_rank) + '\n' + \
              'prob_rho = {}'.format(prob_rho) + '\n' + \
              'share_embeddings = {}'.format(share_embeddings) + '\n' + \
              'threshold = {}'.format(weak_threshold)
    return configs


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format(0, 1))
    print(config_to_str())
    print('=' * 30)
