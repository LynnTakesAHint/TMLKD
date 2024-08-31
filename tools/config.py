# Data path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--distance_type", type=str)
parser.add_argument('--GPU', type=str)
parser.add_argument('--data_name', type=str)
parser.add_argument('--rho', type=int)
parser.add_argument('--threshold', type=float)
parser.add_argument('--d', type=int)
args = parser.parse_args()
GPU = args.GPU
distance_type = args.distance_type
data_name = args.data_name
prob_rho = args.rho
weak_threshold = args.threshold
d = args.d

base_model = 'neutraj'
full_datalength = 10000
mode = 'train'

data_folder = './'

corrdatapath = data_folder + 'features/%s_traj_coord' % data_name
griddatapath = data_folder + 'features/%s_traj_grid' % data_name
real_distancepath = data_folder + 'features/%s_%s_distance_all_10000' % (data_name, distance_type)  # 原本存地址的
data_type = data_name
distance_info_path = data_folder + 'features/%s_' % data_name + distance_type + '_cluster_data_fake'

incell = False

train_size = 200
val_size = 50
max_len = 250
datalength = 3000

# Training Prarmeters

epochs = 250
batch_size = 10
learning_rate = 0.00008
teacher_rank = ("./teacher_predict/%s/%s_enriched_ranks_%d" % (distance_type, data_type, prob_rho))
share_embeddings = ("./teacher_predict/%s/%s_share_embeddings" % (distance_type, data_type))

sampling_num = 5
if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8

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
              'mail_pre_degree = {} '.format(mail_pre_degree) + '\n' + \
              'train_nums = {} '.format(train_size) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datapath = {} '.format(corrdatapath) + '\n' + \
              'datatype = {} '.format(data_type) + '\n' + \
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
