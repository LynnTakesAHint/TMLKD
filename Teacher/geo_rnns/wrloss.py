import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch

import tools.config as config
import numpy as np


class WeightMSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            self.weight.append(0.)
            for traj_index in range(sampling_num):
                self.weight.append(np.array([config.sampling_num - traj_index]))

        self.weight = np.array(self.weight, dtype=object).astype(np.float32)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad=False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU=False):
        div = target - input.view(-1, 1)
        if isReLU:
            div = F.relu(div.view(-1, 1))
        square = torch.mul(div.view(-1, 1), div.view(-1, 1))
        weight_square = torch.mul(square.view(-1, 1), self.weight.view(-1, 1))

        loss = torch.sum(weight_square)
        return loss


class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False)
        negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), True)
        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss, negative_mse_loss])
        return loss


class DiffLoss(Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, shared_feats, task_feats):
        task_feats = task_feats.clone() - torch.mean(task_feats, dim=0)
        shared_feats = shared_feats.clone() - torch.mean(shared_feats, dim=0)
        task_feats = torch.nn.functional.normalize(task_feats, p=2, dim=1).cuda()
        shared_feats = torch.nn.functional.normalize(shared_feats, p=2, dim=1).cuda()

        correlation_matrix = torch.matmul(task_feats.t(), shared_feats)
        cost = torch.mean(correlation_matrix ** 2)
        cost = torch.where(cost > 0, cost, torch.tensor(0.0).cuda())
        return cost * 0.001



class AllTheTeacherLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(AllTheTeacherLoss, self).__init__()
        self.target_loss = WeightedRankingLoss(batch_size, sampling_num)
        self.contrative_loss = DiffLoss()

    def forward(self, input_list):
        target_loss_lst = []
        contrastive_loss_lst = []
        self.share_loss = 0
        self.private_loss = []
        self.decoder_loss = []

        share_input, private_input, decoder_input = input_list
        for pi in private_input:
            p_input, p_target, n_input, n_target = pi
            trajs_target_loss = self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda())
            self.private_loss.append(trajs_target_loss)
            target_loss_lst.append(trajs_target_loss)
        for di in decoder_input:
            p_input, p_target, n_input, n_target = di
            trajs_target_loss = self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda())
            self.decoder_loss.append(trajs_target_loss)
            target_loss_lst.append(trajs_target_loss)
        for p_input, p_target, n_input, n_target, share_embeddings, private_embeddings in share_input:
            # Base Loss
            trajs_target_loss = self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda()) / len(share_input)
            self.share_loss = self.share_loss + trajs_target_loss
            target_loss_lst.append(trajs_target_loss)
            # Representation Loss
            trajs_contrastive_loss = sum(
                [self.contrative_loss(se.cuda(), pe.cuda()) for se, pe in zip(share_embeddings, private_embeddings)]
            ) / len(share_embeddings)
            contrastive_loss_lst.append(trajs_contrastive_loss)
            self.share_loss = self.share_loss + trajs_contrastive_loss
        loss = sum(target_loss_lst) + sum(contrastive_loss_lst) / 2  # 2: positive, negative
        return loss
