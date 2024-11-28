import time
import numpy as np
import torch
import torch.autograd as autograd
import tools.config as config


def pad_sequence(traj_grids, maxlen=100, pad_value=0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0]) * pad_value
        while len(traj) < maxlen:
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs


def test_comput_embeddings(self, spatial_net, test_batch=1025, mode='val'):
    embeddings_list = []
    j = 0
    s = time.time()
    if mode == 'val':
        data_length = config.train_trajectory_size
    else:
        data_length = config.full_datalength
    embeddings = None
    if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    while j < data_length:
        for i in range(self.batch_size):
            grids = self.padded_grids[j:j + test_batch]
            coors = self.padded_coors[j:j + test_batch]
            lens = self.trajs_length[j:j + test_batch]
            inva = torch.tensor(self.all_pre_embs[j:j + test_batch]).cuda()
            out, _ = spatial_net.get_embeddings(
                grids, coors, lens, inva, hidden
            )
            embeddings = out.data
        j += test_batch
        embeddings_list.append(embeddings)
        if mode != 'val':
            if (j % 1000) == 0:
                print(j)
    print('embedding time of {} trajectories: {}'.format(10000, time.time() - s))
    embeddings_list = torch.cat(embeddings_list, dim=0)
    return embeddings_list.cpu().numpy()


def validate_model(self, traj_embeddings, rank_info):
    top_5_count = 0
    cluster_center = self.val_indice
    data_length = len(cluster_center)
    for anchor_traj in cluster_center:
        start = anchor_traj // 50 * 50
        end = start + 50
        test_distance = [(j + start, float(np.exp(-np.sum(np.square(traj_embeddings[anchor_traj] - e)))))
                         for j, e in enumerate(traj_embeddings[start:end])]
        sorted_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
        sorted_test_id = [i[0] for i in sorted_test_distance][:6]
        real_id = rank_info[anchor_traj]
        top_5_trajs_hit = [i for i in sorted_test_id if i in real_id]
        top_5_count += (len(top_5_trajs_hit) - 1)
    hitting_rate = top_5_count / len(cluster_center) / 5
    print('Test on %d trajs, acc is %f.' % (data_length, hitting_rate))
    return hitting_rate


def test_comput_embeddings_for_time(self, spatial_net, test_batch=1025, rrange=1000):
    embeddings_list = []
    j = 0
    data_length = rrange
    embeddings = None
    if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
        hidden = autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda()
    else:
        hidden = (autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda(),
                  autograd.Variable(torch.zeros(test_batch, self.target_size), requires_grad=False).cuda())
    while j < data_length:
        for i in range(self.batch_size):
            grids = self.padded_grids[j:j + test_batch]
            coors = self.padded_coors[j:j + test_batch]
            lens = self.trajs_length[j:j + test_batch]
            inva = torch.tensor(self.all_pre_embs[j:j + test_batch]).cuda()
            out, _ = spatial_net.get_embeddings(
                grids, coors, lens, inva, hidden
            )
            embeddings = out.data
        j += test_batch
        embeddings_list.append(embeddings)
    embeddings_list = torch.cat(embeddings_list, dim=0)
    return embeddings_list.cpu().numpy()


def final_test_model(self, traj_embeddings, print_batch=10, similarity=True, r10in50=False):
    test_range = range(config.train_trajectory_size, config.full_datalength)
    top_10_count = 0
    top_50_count = 0
    top10_in_top50_count = 0
    test_traj_num = 0
    all_true_distance, all_test_distance = [], []
    for i in test_range:
        if similarity:
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
        top10_in_top50 = [l[0] for l in s_test_distance[:11]
                          if l[0] in [j[0] for j in s_true_distance[:51]]]

        top_10_count += len(top10_recall) - 1
        top_50_count += len(top50_recall) - 1
        top10_in_top50_count += len(top10_in_top50) - 1

        all_true_distance.append(s_true_distance[:50])
        all_test_distance.append(s_test_distance[:50])

        true_top_10_distance = 0.
        for ij in s_true_distance[:11]:
            true_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance = 0.
        for ij in s_test_distance[:11]:
            test_top_10_distance += self.distance[i][ij[0]]
        temp_distance_in_test50 = []
        for ij in s_test_distance[:51]:
            temp_distance_in_test50.append([ij, self.distance[i][ij[0]]])

        test_traj_num += 1
        if (i % print_batch) == 0:
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
    print('Search Top 10 in Top 50 recall {}'.format(
        float(top10_in_top50_count) / (test_traj_num * 10)))
    return (float(top_10_count) / (test_traj_num * 10),
            float(top_50_count) / (test_traj_num * 50),
            float(top10_in_top50_count) / (test_traj_num * 10), 0)
