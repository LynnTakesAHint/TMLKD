import os
import pickle
import time
from typing import List
import datetime

import tools.test_methods as tm
from geo_rnns.wrloss import *
from geo_rnns.neutraj_model import NeuTraj_Network

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


def pad_sequence(traj_grids: List[List[List]], maxlen=100, pad_value=0.0):
    paddec_seqs = []
    for traj in traj_grids:
        traj_copy = traj[::]
        pad_r = np.zeros_like(traj[0]) * pad_value
        while (len(traj_copy) < maxlen):
            traj_copy.append(pad_r)
        paddec_seqs.append(traj_copy)
    return paddec_seqs


def list2tensor(val):
    return torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()


class TMLKD_Trainer_NeuTraj(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate=config.learning_rate):
        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self, griddatapath=config.griddatapath,
                     coordatapath=config.corrdatapath,
                     distancepath=config.distance_info_path,
                     full_distancepath=config.real_distancepath):
        full_length = config.full_datalength
        max_len = config.max_len
        traj_grids = pickle.load(open(griddatapath, 'rb'))[0]
        self.trajs_length = [len(j) for j in traj_grids][:full_length]
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_grids][:full_length]
        self.padded_grids = np.array(pad_sequence(traj_grids, max_len))
        traj_coors = pickle.load(open(coordatapath, 'rb'))[0]
        x, y = [], []
        for traj in traj_coors:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_coors = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_coors][:full_length]
        self.padded_coors = np.array(pad_sequence(traj_coors, max_len))
        self.rank_info = {}
        self.distance_info = {}
        self.tot_info = pickle.load(open(distancepath, 'rb'))
        self.anchor_indice = list(self.tot_info.keys())
        for i in self.anchor_indice:
            self.rank_info[i] = [j[0] for j in self.tot_info[i]]
            self.distance_info[i] = [j[1] for j in self.tot_info[i]]
        distance = pickle.load(open(config.real_distancepath, 'rb'))[:config.full_datalength, :config.full_datalength]
        self.distance = distance
        file = config.share_embeddings
        self.all_pre_embs = pickle.load(open(file, 'rb'))
        self.teacher_predict = pickle.load(open(config.teacher_rank, 'rb'))
        self.train_indice = self.anchor_indice[:config.train_size]
        self.val_indice = self.anchor_indice[config.train_size:]

    def batch_generator(self):
        j = 0
        batch_len = self.batch_size
        while j < config.train_size:
            # Target Measure - Anchor - Input
            anchor_grid_input, anchor_coor_input = [], []
            anchor_len = []
            # Target Measure - Clustering - Input & Distance
            target_cluster_grid_input, target_cluster_coor_input, target_cluster_distance = [], [], []
            target_cluster_len = []
            # Latent Knowledge
            anchor_cluster_invariant_representation = []
            target_cluster_invariant_representation = []
            anchor_LK_invariant_representation = []
            target_LK_invariant_representation = []

            # Neutraj
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [[], []], []
            RK_anchor_coor_input, RK_anchor_grid_input, RK_anchor_len = [], [], []
            RK_other_coor_input, RK_other_grid_input, RK_other_len = [], [], []
            teacher_predicted_res = []
            for i in range(batch_len):
                anchor_idx = self.train_indice[j + i]
                indices_list = []
                if anchor_idx not in batch_trajs_keys:
                    batch_trajs_keys[anchor_idx] = 0
                    batch_trajs_input[0].append(self.padded_coors[anchor_idx])
                    batch_trajs_input[1].append(self.padded_grids[anchor_idx])
                    batch_trajs_len.append(self.trajs_length[anchor_idx])
                for k, traj_idx in enumerate(self.rank_info[anchor_idx]):
                    # Anchor - Anchor - Input
                    anchor_coor_input.append(self.padded_coors[anchor_idx])
                    anchor_grid_input.append(self.padded_grids[anchor_idx])
                    anchor_len.append(self.trajs_length[anchor_idx])
                    # Target Metric - Clustering - Input & Distance
                    target_cluster_grid_input.append(self.padded_grids[traj_idx])
                    target_cluster_coor_input.append(self.padded_coors[traj_idx])
                    target_cluster_len.append(self.trajs_length[traj_idx])
                    target_cluster_distance.append(float(np.exp(-self.distance_info[anchor_idx][k])))
                    indices_list.append(traj_idx % config.m)
                    # ========================================================= #
                    # Latent Knowledge:
                    anchor_cluster_invariant_representation.append(self.all_pre_embs[anchor_idx])
                    target_cluster_invariant_representation.append(self.all_pre_embs[traj_idx])
                    if traj_idx not in batch_trajs_keys:
                        batch_trajs_keys[traj_idx] = 0
                        batch_trajs_input[0].append(self.padded_coors[traj_idx])
                        batch_trajs_input[1].append(self.padded_grids[traj_idx])
                        batch_trajs_len.append(self.trajs_length[traj_idx])
            # Rank-Knowledge: Anchor Traj - Predicted Result
            for i in range(batch_len):
                anchor_idx = self.train_indice[j + i]
                start = anchor_idx // 50 * 50
                end = start + 50
                for k in range(start, end):
                    RK_anchor_coor_input.append(self.padded_coors[anchor_idx])
                    RK_anchor_grid_input.append(self.padded_grids[anchor_idx])
                    RK_anchor_len.append(self.trajs_length[anchor_idx])
                    RK_other_coor_input.append(self.padded_coors[k])
                    RK_other_grid_input.append(self.padded_grids[k])
                    RK_other_len.append(self.trajs_length[k])
                    anchor_LK_invariant_representation.append(self.all_pre_embs[anchor_idx])
                    target_LK_invariant_representation.append(self.all_pre_embs[anchor_idx])
                teacher_predicted_res.append(self.teacher_predict[anchor_idx])
            batch = (
                # Target Metric - Anchor - Anchor - Input
                [np.array(anchor_coor_input),
                 np.array(anchor_grid_input),
                 anchor_len,
                 torch.tensor(anchor_cluster_invariant_representation)],

                # Target Metric - Clustering - Input & Distance
                [np.array(target_cluster_coor_input),
                 np.array(target_cluster_grid_input),
                 target_cluster_len,
                 torch.tensor(target_cluster_invariant_representation),
                 np.array([[i] for i in target_cluster_distance])],

                # Rank_Knowledge
                [np.array(RK_anchor_coor_input),
                 np.array(RK_anchor_grid_input),
                 RK_anchor_len,
                 torch.tensor(anchor_LK_invariant_representation)],

                [np.array(RK_other_coor_input),
                 np.array(RK_other_grid_input),
                 RK_other_len,
                 torch.tensor(target_LK_invariant_representation)],

                teacher_predicted_res
            )
            yield batch
            j = j + batch_len

    def trained_model_eval(self, load_model=None):
        spatial_net = NeuTraj_Network(4, self.target_size, config.grid_size, config.batch_size, config.sampling_num,
                                      stard_LSTM=False, incell=config.incell)
        spatial_net.cuda()
        if load_model is not None:
            m = torch.load(load_model)
            spatial_net.load_state_dict(m)
            embeddings = tm.test_comput_embeddings(
                self, spatial_net, test_batch=1000, mode='test')
            print('len(embeddings): {}'.format(len(embeddings)))
            acc1 = tm.final_test_model(self, embeddings, print_batch=500, similarity=True)
            print(acc1)
            return acc1

    def TMLKD_train(self, save_model=True, load_model=None):
        global end
        spatial_net = NeuTraj_Network(4, self.target_size, config.grid_size, config.batch_size, config.sampling_num,
                                      stard_LSTM=False, incell=config.incell)
        params_to_opt = [list(filter(lambda p: p.requires_grad, spatial_net.parameters()))]
        optimizer = torch.optim.Adam([{'params': params_to_opt[0], 'lr': config.learning_rate}])
        my_loss = Total_Loss()
        spatial_net.cuda()
        my_loss.cuda()
        best_top10_acc = float("-inf")
        if load_model is not None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            embeddings = tm.test_comput_embeddings(
                self, spatial_net, test_batch=config.em_batch)  # 首先获得所有轨迹的Embedding
            acc1 = tm.validate_model(self, embeddings, self.rank_info)
            best_top10_acc = acc1

        print_batch = 20
        for epoch in range(config.epochs):
            spatial_net.train()
            print('=' * 40)
            print("Start training Epochs : {}".format(epoch + 1))
            start = time.time()

            for i, batch in enumerate(self.batch_generator()):
                anchor_coor, anchor_grid, anchor_len, anchor_inva = batch[0]
                target_coor, target_grid, target_len, target_inva, target_dis = batch[1]
                target_distance, _ = \
                    spatial_net((anchor_coor, target_coor),
                                (anchor_grid, target_grid),
                                (anchor_len, target_len),
                                (anchor_inva, target_inva))
                RK_anchor_coor, RK_anchor_grid, RK_anchor_len, RK_anchor_inva = batch[2]
                RK_other_coor, RK_other_grid, RK_other_len, RK_other_inva = batch[3]
                teacher_predicted_res = batch[4]
                RK_distance, target_specific_representation = (
                    spatial_net((RK_anchor_coor, RK_other_coor),
                                (RK_anchor_grid, RK_other_grid),
                                (RK_anchor_len, RK_other_len),
                                (RK_anchor_inva, RK_other_inva),
                                teacher_flag=True))
                loss = my_loss(target_distance, target_dis,
                               teacher_predicted_res, RK_distance,
                               target_specific_representation, RK_other_inva)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optim_time = time.time()
                batch_end = time.time()
                if (i + 1) % print_batch == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Positive_Loss: {},  '
                          'Rank_Loss: {}, Latent_Knowledge_Loss: {},  '
                          'Update_Time_cost: {}, All_Time_cost: {}'.
                          format(epoch + 1, config.epochs, (i + 1) * self.batch_size, config.train_size,
                                 my_loss.trajs_mse_loss.item(),
                                 my_loss.trajs_rank_loss.item(),
                                 my_loss.trajs_latent_loss.item(),
                                 batch_end - optim_time, batch_end - start))
                end = time.time()
            print('Epoch [{}/{}],  Positive_Loss: {}, Rank_Loss: {}, Latent_Knowledge_Loss: {}, '
                  'Time_cost: {}'.format(epoch + 1, config.epochs,
                                         my_loss.trajs_mse_loss.item(),
                                         my_loss.trajs_rank_loss.item(),
                                         my_loss.trajs_latent_loss.item(),
                                         end - start))
            embeddings = tm.test_comput_embeddings(
                self, spatial_net, test_batch=config.em_batch)
            acc1 = tm.validate_model(self, embeddings, self.rank_info)
            print('Acc is :', acc1)
            if save_model and acc1 > best_top10_acc:
                best_top10_acc = acc1
                x = datetime.datetime.now()
                save_model_name = (config.data_folder + "/model/" + str(config.data_name) + "_" + x.strftime("%d") + "_" + x.strftime(
                    "%H") + "_" + str(config.distance_type) + "_" + str(config.base_model) + "_tmlkd_best_model.h5")
                self.best_model = save_model_name
                print(save_model_name)
                torch.save(spatial_net.state_dict(), save_model_name)
            else:
                print("Worse!")
