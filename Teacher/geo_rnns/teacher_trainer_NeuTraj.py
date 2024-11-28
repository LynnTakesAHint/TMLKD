import os
import pickle
import time
from typing import List
import itertools

import tools.sampling_methods as sm
import tools.test_methods as tm
from geo_rnns.wrloss import *
from geo_rnns.neutraj_model import NeuTraj_Network, NeuTraj_Share_Network, NeuTraj_Decoder
from geo_rnns.teacher_trainer_T3S import pad_sequence

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


class NeutrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate=config.learning_rate):
        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self,
                     griddatapath=config.griddatapath,
                     coordatapath=config.corrdatapath):
        self.train_idx = config.train_idx
        self.val_idx = config.val_idx
        self.test_idx = config.test_idx
        dataset_length = config.full_datalength
        traj_grids, useful_grids, max_len = pickle.load(
            open(griddatapath, 'rb'))

        self.trajs_length = [len(j) for j in traj_grids][:dataset_length]
        self.grid_size = config.gird_size
        self.max_length = max_len

        grid_trajs = [[[i[0] + config.spatial_width, i[1] + config.spatial_width] for i in tg]
                      for tg in traj_grids[:dataset_length]]

        traj_grids, useful_grids, max_len = pickle.load(
            open(coordatapath, 'rb'))

        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_grids]
        coor_trajs = traj_grids[:dataset_length]

        grid_train_seqs = grid_trajs[self.train_idx[0]:self.train_idx[1]]
        coor_train_seqs = coor_trajs[self.train_idx[0]:self.train_idx[1]]

        self.grid_trajs = grid_trajs
        self.grid_train_seqs = grid_train_seqs
        self.coor_trajs = coor_trajs
        self.coor_train_seqs = coor_train_seqs
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append(
                    [coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)

        print(f"Padded Trajs shape:{len(pad_trjs)}")
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen=max_len))
        self.train_seqs: List[List[List[float, float, int, int]]] = \
            self.padded_trajs[self.train_idx[0]:self.train_idx[1]]
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen=max_len))
        self.distances = []
        self.train_distances = []
        self.val_distances = []
        for measure, distance_path in config.source_distance_paths.items():
            distance = pickle.load(open(distance_path, 'rb'))
            max_dis = distance.max()
            print(f'max value in {measure} distance matrix : {max_dis}')
            if measure == 'dtw' or measure == 'erp':
                distance = distance / max_dis
            train_distance = distance[self.train_idx[0]:self.train_idx[1], self.train_idx[0]:self.train_idx[1]]
            val_distance = distance[:self.val_idx[1], :self.val_idx[1]]
            self.distances.append(distance)
            self.train_distances.append(train_distance)
            self.val_distances.append(val_distance)

    def batch_generator(self, distance_id=0):
        j = 0
        train_seqs = self.train_seqs
        self.distance = self.distances[distance_id]
        while j < len(train_seqs):
            anchor_input, trajs_input, negative_input = [], [], []
            positive_distance = []
            negative_distance = []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            for i in range(self.batch_size):
                sampling_index_list = sm.distance_sampling(self.distance, len(self.train_seqs), j + i, config.mail_pre_degrees.get(config.source_distance[distance_id], 8))
                negative_sampling_index_list = sm.negative_distance_sampling(self.distance, len(self.train_seqs), j + i, config.mail_pre_degrees.get(config.source_distance[distance_id], 8))
                trajs_input.append(train_seqs[j + i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                if j + i not in batch_trajs_keys:
                    batch_trajs_keys[j + i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])
                positive_distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j + i])
                    trajs_input.append(train_seqs[traj_index])

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
                    positive_distance.append(
                        np.exp(-float(self.train_distances[distance_id][j + i][traj_index])
                               * config.mail_pre_degrees.get(config.source_distance[distance_id], 8)))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(
                        np.exp(-float(self.train_distances[distance_id][j + i][traj_index])
                               * config.mail_pre_degrees.get(config.source_distance[distance_id], 8)))
                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(
                negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            yield (
                [np.array(anchor_input), np.array(trajs_input), np.array(negative_input), np.array(batch_trajs_input)],
                [anchor_input_len, trajs_input_len,
                 negative_input_len, batch_trajs_len],
                [positive_distance, negative_distance])
            j = j + self.batch_size

    def get_optimizer(self, model, model2=None):
        params_to_opt = list(filter(lambda p: p.requires_grad, model.parameters()))
        if model2 is not None:
            params_to_opt = itertools.chain(params_to_opt, filter(lambda p: p.requires_grad, model2.parameters()))
        opt = torch.optim.Adam(params=params_to_opt, lr=config.learning_rate)
        return opt

    def judge_update(self, best_acc, current_acc):
        return sum(best_acc) <= sum(current_acc)

    def train(self, print_batch=10, print_test=300, load_model=None):
        private_spatial_net = [
            NeuTraj_Network(4, self.target_size, config.grid_size, config.batch_size, config.sampling_num,
                            stard_LSTM=False, incell=config.incell) for _ in range(len(self.distances))]
        for p in private_spatial_net:
            p.cuda()
        decoder_net = [NeuTraj_Decoder() for _ in range(len(self.distances))]
        for p in decoder_net:
            p.cuda()
        private_optimizer = [self.get_optimizer(private_spatial_net[i], decoder_net[i]) for i in
                             range(len(private_spatial_net))]

        share_spatial_net = NeuTraj_Share_Network(4, self.target_size, config.grid_size, config.batch_size,
                                                  config.sampling_num,
                                                  stard_LSTM=False, incell=config.incell)
        share_spatial_net.cuda()
        total_criterion = AllTheTeacherLoss(self.batch_size, self.sampling_num)
        total_criterion.cuda()
        share_best_top10_acc_list = [float("-inf") for i in range(len(self.distances))]
        private_best_top10_acc = [float("-inf") for i in range(len(self.distances))]
        decoder_best_top10_acc = [float("-inf") for i in range(len(self.distances))]
        share_optimizer = self.get_optimizer(share_spatial_net)

        if load_model is not None:
            pass
        for epoch in range(config.epochs):
            start = time.time()
            print('=' * 40)
            print("Start training Epochs : {}".format(epoch + 1))
            share_spatial_net.train()
            share_top10_acc_list = [0 for i in range(len(self.distances))]
            for distance_id in range(len(self.distances)):
                private_spatial_net[distance_id].train()
                decoder_net[distance_id].train()
                for i, batch in enumerate(self.batch_generator(distance_id)):
                    inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]
                    share_loss_input, private_loss_input, decoder_loss_input = [], [], []
                    share_embeddings, (share_pos_distance, share_neg_distance) = share_spatial_net(inputs_arrays,
                                                                                                   inputs_len_arrays)
                    private_embeddings, (private_pos_distance, private_neg_distance) = private_spatial_net[distance_id](
                        inputs_arrays, inputs_len_arrays)
                    (final_pos_dis, final_neg_dis) = decoder_net[distance_id](share_embeddings, private_embeddings)

                    positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))
                    negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))
                    private_loss_input.append([private_pos_distance, positive_distance_target,
                                               private_neg_distance, negative_distance_target])
                    decoder_loss_input.append([final_pos_dis, positive_distance_target,
                                               final_neg_dis, negative_distance_target])
                    share_loss_input.append([share_pos_distance, positive_distance_target,
                                             share_neg_distance, negative_distance_target,
                                             share_embeddings[1:],
                                             private_embeddings[1:]])
                    private_optimizer[distance_id].zero_grad()
                    share_optimizer.zero_grad()
                    loss = total_criterion((share_loss_input, private_loss_input, decoder_loss_input))
                    loss.backward()
                    private_optimizer[distance_id].step()
                    share_optimizer.step()
                    batch_end = time.time()
                    if (i + 1) % print_batch == 0:
                        print('Source Distance: {}, Epoch [{}/{}], Step [{}/{}] \n'
                              'Share Loss: {}, Private Loss: {}, Decoder Loss: {}, All_Time_cost: {}'.
                              format(config.source_distance[distance_id],
                                     epoch + 1, config.epochs, i + 1, len(self.train_seqs) // self.batch_size,
                                     total_criterion.share_loss.item(), total_criterion.private_loss[0].item(),
                                     total_criterion.decoder_loss[0].item(), batch_end - start))
                # validate model
                print('Validate Share Net:', end='')
                share_embeddings = tm.test_comput_embeddings(self, share_spatial_net, mode='val')
                for validate_distance_id in range(len(self.distances)):
                    tmp = tm.validate_model(self, share_embeddings, validate_distance_id)
                    share_top10_acc_list[validate_distance_id] = tmp
                if self.judge_update(best_acc=share_best_top10_acc_list, current_acc=share_top10_acc_list):
                    share_best_top10_acc_list = [i for i in share_top10_acc_list]
                    share_save_model_name = \
                        f'{config.data_folder}/model/{config.data_name}_{config.distance_type}_{config.base_model}_share_best_model.h5'
                    print(share_save_model_name)
                    torch.save(share_spatial_net.state_dict(), share_save_model_name)
                print('Validate Private Net:', end='')
                private_embeddings = tm.test_comput_embeddings(self, private_spatial_net[distance_id], mode='val')
                private_top10_acc = tm.validate_model(self, private_embeddings, distance_id)
                if private_top10_acc > private_best_top10_acc[distance_id]:
                    print(
                        f"Better hr on {config.source_distance[distance_id]}: private {private_top10_acc}")
                    private_best_top10_acc[distance_id] = private_top10_acc
                    private_save_model_name = \
                        f'{config.data_folder}/model/{config.data_name}_{config.distance_type}_{config.source_distance[distance_id]}_{config.base_model}_private_best_model.h5'
                    print(private_save_model_name)
                    torch.save(private_spatial_net[distance_id].state_dict(), private_save_model_name)
                print('Validate Decoder Net:', end='')
                decoder_net_embeddings = tm.test_comput_embeddings(self, decoder_net[distance_id],
                                                                   private_embeddings=private_embeddings,
                                                                   share_embeddings=share_embeddings,
                                                                   mode='val', decoder=True)
                decoder_top10_acc = tm.validate_model(self, decoder_net_embeddings, distance_id)
                if decoder_top10_acc > decoder_best_top10_acc[distance_id]:
                    print(
                        f"Better hr on {config.source_distance[distance_id]}: decoder {decoder_top10_acc}")
                    decoder_best_top10_acc[distance_id] = decoder_top10_acc
                    decoder_save_model_name = \
                        f'{config.data_folder}/model/{config.data_name}_{config.distance_type}_{config.source_distance[distance_id]}_{config.base_model}_decoder_best_model.h5'
                    print(decoder_save_model_name)
                    torch.save(decoder_net[distance_id].state_dict(), decoder_save_model_name)
