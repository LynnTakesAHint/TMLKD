import os
import pickle
from typing import List
import tools.test_methods as tm
import tools.config as config
import torch
import numpy as np
from geo_rnns.neutraj_model import NeuTraj_Network, NeuTraj_Share_Network, NeuTraj_Decoder
from geo_rnns.t3s_model import T3S_Network, T3S_Share_Network, T3S_Decoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


def pad_sequence(traj_grids: List[List[List]], maxlen=250, pad_value=0.0):
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
        traj_copy = traj[::]
        pad_r = np.zeros_like(traj[0]) * pad_value
        while len(traj_copy) < maxlen:
            traj_copy.append(pad_r)  # 在末尾不上同纬度的pad_value填充的矩阵
        paddec_seqs.append(traj_copy)
    return paddec_seqs


class TeacherModelEmbeddingGetter(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num):
        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.share_embeddings = None

    def data_prepare(self,
                     griddatapath=config.griddatapath,
                     coordatapath=config.corrdatapath):
        dataset_length = 10000
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
        self.grid_trajs = grid_trajs
        self.coor_trajs = coor_trajs
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append(
                    [coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)

        print(f"Padded Trajs shape:{len(pad_trjs)}")
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen=max_len))

    def test_models(self, model_name_dict, distance_id):
        share = model_name_dict.get('share', None)
        private = model_name_dict.get('private', None)
        decoder = model_name_dict.get('decoder', None)
        assert share is not None
        assert private is not None
        assert decoder is not None
        if config.base_model == 'NeuTraj':
            private_spatial_net = NeuTraj_Network(4, self.target_size, config.grid_size,
                                                  config.batch_size, config.sampling_num,
                                                  stard_LSTM=False, incell=config.incell).cuda()
            share_spatial_net = NeuTraj_Share_Network(4, self.target_size, config.grid_size, config.batch_size,
                                                      config.sampling_num,
                                                      stard_LSTM=False, incell=config.incell).cuda()
            decoder_net = NeuTraj_Decoder().cuda()

        else:
            private_spatial_net = T3S_Network(self.target_size, config.batch_size, config.sampling_num).cuda()
            share_spatial_net = T3S_Share_Network(self.target_size, config.batch_size, config.sampling_num).cuda()
            decoder_net = T3S_Decoder().cuda()
        pm = torch.load(private)
        private_spatial_net.load_state_dict(pm)
        sm = torch.load(share)
        share_spatial_net.load_state_dict(sm)
        dm = torch.load(decoder)
        decoder_net.load_state_dict(dm)
        private_embeddings = tm.test_comput_embeddings(self, private_spatial_net)
        if self.share_embeddings is None:
            share_embeddings = tm.test_comput_embeddings(self, share_spatial_net)
            pickle.dump(share_embeddings,
                        open(f'./teacher_predict_result/{config.distance_type}/{config.data_name}_share_embeddings',
                             'wb'))
            self.share_embeddings = share_embeddings
        decoder_embeddings = tm.test_comput_embeddings(self, decoder_net, private_embeddings=private_embeddings,
                                                       share_embeddings=self.share_embeddings, decoder=True)
        pickle.dump(decoder_embeddings, open(
            f'./teacher_predict_result/{config.distance_type}/{config.data_name}_{config.source_distance[distance_id]}_decoder_embeddings',
            'wb'))
