import time
import numpy as np
import torch

import torch.autograd as autograd
import tools.config as config


def pad_sequence(traj_grids, maxlen=250, pad_value=0.0):
    """
    padding
    Args:
        traj_grids: 原始轨迹
        maxlen: 最大长度
        pad_value: 空值

    Returns:

    """
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0]) * pad_value
        while len(traj) < maxlen:
            traj.append(pad_r)  # 在末尾不上同纬度的pad_value填充的矩阵
        paddec_seqs.append(traj)
    return paddec_seqs

def test_comput_embeddings(self, spatial_net, test_batch=900,
                           private_embeddings=None, share_embeddings=None,mode='test', decoder = False):
    embeddings_list = []
    s = time.time()
    j = 0
    if mode == 'val':
        data_length = self.val_idx[1]
    else:
        data_length = config.full_datalength
    embeddings = None
    if data_length % test_batch != 0:
        test_batch = 1000
    if data_length % test_batch != 0:
        test_batch = 500
    assert data_length % test_batch == 0
    if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    if decoder:
        share_embeddings = torch.tensor(share_embeddings).cuda()
        private_embeddings = torch.tensor(private_embeddings).cuda()
        while j < data_length:
            input_s = share_embeddings[j:j+test_batch]
            input_p = private_embeddings[j:j+test_batch]
            out = spatial_net.get_embeddings(
                input_s, input_p
            )
            embeddings = out.data
            j += test_batch
            embeddings_list.append(embeddings)
            if j % 1000 == 0:
                print(j)
    else:
        while j < data_length:
            inputs = torch.tensor(self.padded_trajs[j:j + test_batch], dtype=torch.float32)
            lens = torch.tensor(self.trajs_length[j:j + test_batch], dtype=torch.int)
            if config.base_model == 'T3S':
                coor, grid = inputs[:, :, :2], inputs[:, :, 2:]
                out = spatial_net.get_embeddings(
                    coor, grid, lens, hidden
                )
            elif config.base_model == 'NeuTraj':
                out = spatial_net.get_embeddings(
                    inputs, lens, hidden
                )
            else:
                raise NotImplementedError
            embeddings = out.data
            j += test_batch
            embeddings_list.append(embeddings)
            if j % 1000 == 0:
                print(j)
    embeddings_list = torch.cat(embeddings_list, dim=0)
    e = time.time()
    print('embedding time:', e - s)
    return embeddings_list.cpu().numpy()


def validate_model(self, traj_embeddings, distance_type, similarity = True, print_batch = 500):
    top_10_count = 0
    test_range = range(self.val_idx[0], self.val_idx[1])    # 只测这一部分
    test_num = len(test_range)
    distance = self.distances[distance_type]
    for i in test_range:
        if similarity:
            # This is for the exp similarity
            test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
                             for j, e in enumerate(traj_embeddings)]
            t_similarity = np.exp(-distance[i][:len(traj_embeddings)])
            true_distance = list(enumerate(t_similarity))

            s_test_distance = sorted(
                test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(
                true_distance, key=lambda a: a[1], reverse=True)
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(
                enumerate(distance[i][:len(traj_embeddings)]))
            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
        top10_recall = [l[0] for l in s_test_distance[:11]
                        if l[0] in [j[0] for j in s_true_distance[:11]]]
        top_10_count += len(top10_recall) - 1
        if i % print_batch == 0:
            print(top10_recall)
    print('Test on {} trajs'.format(test_num))
    print('Search Top 10 recall {}'.format(float(top_10_count) / (test_num * 10)))
    return float(top_10_count) / (test_num * 10)

def test_model(self, traj_embeddings, print_batch=1000, similarity=True, r10in50=False):
    test_range = range(self.test_idx[0], self.test_idx[1])
    top_10_count = 0
    top_50_count = 0
    top5_in_top50_count = 0
    top10_in_top50_count = 0
    test_traj_num = 0
    for i in test_range:
        if similarity:
            # This is for the exp similarity
            test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
                             for j, e in enumerate(traj_embeddings)]
            t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)])
            true_distance = list(enumerate(t_similarity))

            s_test_distance = sorted(
                test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(
                true_distance, key=lambda a: a[1], reverse=True)
        else:
            # This is for computing the distance
            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings)]
            true_distance = list(
                enumerate(self.distance[i][:len(traj_embeddings)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])

        top10_recall = [l[0] for l in s_test_distance[:11]
                        if l[0] in [j[0] for j in s_true_distance[:11]]]
        top50_recall = [l[0] for l in s_test_distance[:51]
                        if l[0] in [j[0] for j in s_true_distance[:51]]]
        top5_in_top50 = [l[0] for l in s_test_distance[:6]
                          if l[0] in [j[0] for j in s_true_distance[:51]]]
        top10_in_top50 = [l[0] for l in s_test_distance[:11]
                          if l[0] in [j[0] for j in s_true_distance[:51]]]


        top_10_count += len(top10_recall) - 1
        top_50_count += len(top50_recall) - 1
        top10_in_top50_count += len(top10_in_top50) - 1
        top5_in_top50_count += len(top5_in_top50) - 1

        test_traj_num += 1
        if (i % print_batch) == 0:
            # print test_distance
            print('**----------------------------------**')
            print(s_test_distance[:20])
            print(s_true_distance[:20])
            print(top10_recall)
            print(top50_recall)

    print('need to test:', test_traj_num)
    print('Test on {} trajs'.format(test_traj_num))
    print('Search Top 50 recall {}'.format(
        float(top_50_count) / (test_traj_num * 50)))
    print('Search Top 10 recall {}'.format(
        float(top_10_count) / (test_traj_num * 10)))
    print('Search Top 5 in Top 50 recall {}'.format(
        float(top5_in_top50_count) / (test_traj_num * 5)))
    print('Search Top 10 in Top 50 recall {}'.format(
        float(top10_in_top50_count) / (test_traj_num * 10)))
    return (float(top_10_count) / (test_traj_num * 10),
            float(top_50_count) / (test_traj_num * 50),
            float(top5_in_top50_count) / (test_traj_num * 5),
            float(top10_in_top50_count) / (test_traj_num * 10))