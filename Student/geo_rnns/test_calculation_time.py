import os
import pickle
import time
from typing import List
import datetime
from tqdm import tqdm

import tools.test_methods as tm
from geo_rnns.wrloss import *
from geo_rnns.t3s_model import T3S_Network
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


class ComputationTimeCompares(object):
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
        self.trajs_length = [len(j) for j in traj_grids]
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_grids]
        self.padded_grids = np.array(pad_sequence(traj_grids, max_len))
        traj_coors = pickle.load(open(coordatapath, 'rb'))[0]
        x, y = [], []
        for traj in traj_coors:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        traj_coors = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_coors]
        self.padded_coors = np.array(pad_sequence(traj_coors, max_len))
        self.all_pre_embs = pickle.load(open(config.share_embeddings, 'rb'))


    def expand_to_target(self, array, target_size):
        """扩展数组到指定大小"""
        current_size = len(array)
        repeat_times = target_size // current_size + 1
        extra_rows = target_size % current_size

        # 重复填充
        expanded_array = np.tile(array, (repeat_times, 1, 1))
        # print(expanded_array.shape)
        return expanded_array[:target_size]

 
    def calculate_time_eval(self, load_model, distance_type, base_model, data_length=5000):
        if data_length > 5000:
            self.padded_grids = self.expand_to_target(self.padded_grids, data_length)
            self.padded_coors = self.expand_to_target(self.padded_coors, data_length)
            tl = self.trajs_length
            for i in range(data_length // 10000):
                self.trajs_length.extend(tl)
            self.trajs_length = self.trajs_length[:data_length]
            self.all_pre_embs = np.tile(self.all_pre_embs, (data_length // 10000 + 1, 1))[:data_length]
            
        start_time = time.time()
        if base_model == 'T3S':
            spatial_net = T3S_Network(target_size=self.target_size, batch_size=self.batch_size, sampling_num=self.sampling_num)
        else:
            spatial_net = NeuTraj_Network(4, self.target_size, config.grid_size, config.batch_size, config.sampling_num,
                                        stard_LSTM=True, incell=config.incell)
        spatial_net.cuda()
        m = torch.load(load_model)
        traj_embeddings = tm.test_comput_embeddings_and_computation(self, spatial_net, test_batch=2500, data_length=data_length)
        all_res = []
        # for i in tqdm(range(data_length)):
        # # for i in range(data_length):
        #     # This is for the exp similarity
        #     diff = traj_embeddings - traj_embeddings[i]
        #     squared_diff = diff ** 2  # 差值平方
        #     sum_squared_diff = np.sum(squared_diff, axis=1)  # 沿维度求和
        #     pred_distances = np.exp(-np.sqrt(sum_squared_diff))  # 开平方
        #     all_res.append(pred_distances)
        # all_res = np.array(all_res)
        # end_time = time.time()
        # print(f"Compute {distance_type} - {data_length} : {end_time - start_time}s")

