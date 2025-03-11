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
    def __init__(self, batch_size=config.batch_size, diff_loss_weight=config.diff_loss_weight):
        super(DiffLoss, self).__init__()
        self.batch_size = batch_size
        self.diff_loss_weight = diff_loss_weight

    def forward(self, input1, input2):
        input1 = input1.view(self.batch_size, -1)
        input2 = input2.view(self.batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
        diff_loss = torch.mean(torch.abs(input1_l2.t().mm(input2_l2)))
        return diff_loss * self.diff_loss_weight


class AllTheTeacherLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(AllTheTeacherLoss, self).__init__()
        self.target_loss = WeightedRankingLoss(batch_size, sampling_num)
        self.contrative_loss = DiffLoss()

    def forward(self, input_list, freeze_share, freeze_private):
        target_loss_lst = []
        contrastive_loss_lst = []
        self.share_loss = 0
        self.private_loss = 0
        self.decoder_loss = 0
        self.contrative_loss_val = 0
        share_input, private_input, decoder_input = input_list
        if not freeze_private:
            p_input, p_target, n_input, n_target = private_input[0]
            trajs_target_loss = self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda())
            self.private_loss += trajs_target_loss
            target_loss_lst.append(trajs_target_loss)
        p_input, p_target, n_input, n_target = decoder_input[0]
        trajs_target_loss = self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda())
        self.decoder_loss += trajs_target_loss
        target_loss_lst.append(trajs_target_loss)
        for p_input, p_target, n_input, n_target, share_embeddings, private_embeddings in share_input:
            # Base Loss
            if not freeze_share:
                trajs_target_loss = (self.target_loss(p_input.cuda(), p_target.cuda(), n_input.cuda(), n_target.cuda())
                                     / len(share_input))
                self.share_loss += self.share_loss + trajs_target_loss
                target_loss_lst.append(trajs_target_loss)
            # Representation Loss
            trajs_contrastive_loss_list = []
            for i in range(len(share_embeddings)):
                if not freeze_share or not freeze_private:
                    se = share_embeddings[i]
                    pe = private_embeddings[i]
                    cl = self.contrative_loss(se.cuda(), pe.cuda())
                    trajs_contrastive_loss_list.append(cl)
                    self.contrative_loss_val += cl
            trajs_contrastive_loss = 0 if not trajs_contrastive_loss_list else sum(trajs_contrastive_loss_list) / len(trajs_contrastive_loss_list)
            contrastive_loss_lst.append(trajs_contrastive_loss)
            self.share_loss += trajs_contrastive_loss
        loss = sum(target_loss_lst) + sum(contrastive_loss_lst)  # 2: positive, negative
        return loss
