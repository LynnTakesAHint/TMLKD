# Data path
distance_type = 'hausdorff'
data_name = 'sroma'
source_distance = ['discret_frechet', 'hausdorff', 'erp', 'dtw']    # you can select from these measures, also, you can add new teachers.
GPU = '0'

prob_rho = 7
weak_threshold = 0.01
d = 128

base_model = 'T3S'  # T3S or NeuTraj
full_datalength = 10000

epochs = 250
batch_size = 5
sampling_num = 5
train_size = 200
val_size = 50
max_len = 250
train_trajectory_size = 3000
learning_rate = 0.00005
rank_loss_weight = 0.000001
incell = False

data_folder = '/hy-tmp'
corrdatapath = f"{data_folder}/features/{data_name}_traj_coord"
griddatapath = f"{data_folder}/features/{data_name}_traj_grid"
real_distancepath = f"{data_folder}/features/{data_name}_{distance_type}_distance_all_{full_datalength}"
distance_info_path = f"{data_folder}/features/{data_name}_{distance_type}_cluster_data_fake"
teacher_rank = f"./teacher_predict_result/{distance_type}/{data_name}_enriched_ranks_{prob_rho}"
share_embeddings = f"./teacher_predict_result/{distance_type}/{data_name}_share_embeddings"

# Test Config
em_batch = 1000
test_num = 1000

recurrent_unit = 'GRU'  # GRU, LSTM or SimpleRNN
spatial_width = 2

m = 50
grid_size = [1100, 1100]

teacher_dim = 128


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
              'prob_rho = {}'.format(prob_rho) + '\n' + \
              'threshold = {}'.format(weak_threshold)
    return configs


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format(0, 1))
    print(config_to_str())
    print('=' * 30)
