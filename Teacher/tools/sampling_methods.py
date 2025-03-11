import random

import numpy as np

import tools.config as config

random.seed(2018)
np.random.seed(2018)


def random_sampling(train_seq_len, index):
    sampling_index_list = random.sample(range(train_seq_len), config.sampling_num)
    return sampling_index_list


def distance_sampling(distance, train_seq_len, index, mail_pre_degree):
    index_dis = distance[index]
    pre_sort = [np.exp(-i * mail_pre_degree) for i in index_dis[:train_seq_len]]
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort / np.sum(pre_sort):
        importance.append(t)
        t += i
    importance = np.array(importance)
    # print importance
    while len(sample_index) < config.sampling_num:
        a = np.random.uniform()
        idx = np.where(importance > a)[0]
        if len(idx) == 0:
            sample_index.append(train_seq_len - 1)
        elif ((idx[0] - 1) not in sample_index) & (not ((idx[0] - 1) == index)):
            sample_index.append(idx[0] - 1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]



if __name__ == '__main__':
    pass
