import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch

import tools.config as config
from tools.compute_list import correlation_partition


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


class DiffLoss(Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, shared_feats, task_feats):
        task_feats = task_feats.clone() - torch.mean(task_feats, dim=0)
        shared_feats = shared_feats.clone() - torch.mean(shared_feats, dim=0)
        task_feats = torch.nn.functional.normalize(task_feats, p=2, dim=1).cuda()
        shared_feats = torch.nn.functional.normalize(shared_feats, p=2, dim=1).cuda()
        correlation_matrix = torch.matmul(task_feats.t(), shared_feats)
        cost = torch.mean(correlation_matrix ** 2) * 0.01
        cost = torch.where(cost > 0, cost, torch.tensor(0.0).cuda())
        return cost


class Distill_Rank_Loss_ListWise(Module):
    def __init__(self, loss_weight=config.rank_loss_weight):
        super(Distill_Rank_Loss_ListWise, self).__init__()
        self.loss_weight = loss_weight
        print("Rank Loss Weight:", loss_weight)

    def forward(self, partition, student_predict_result, batch_size=5):
        strong, weak, no = partition
        student_predict_result = student_predict_result.view(batch_size,
                                                             student_predict_result.size(0) // batch_size)
        tot_loss = torch.tensor(0.).cuda()

        for batch_idx in range(batch_size):
            strong_indices = torch.tensor(strong[batch_idx], dtype=torch.long).cuda()
            weak_indices = torch.tensor(weak[batch_idx], dtype=torch.long).cuda()
            no_indices = torch.tensor(no[batch_idx], dtype=torch.long).cuda()

            prediction = student_predict_result[batch_idx]  # 获取当前批次的预测结果

            strong_predicted_res = prediction.index_select(0, strong_indices)
            weak_predicted_res = prediction.index_select(0, weak_indices)
            no_predicted_res = prediction.index_select(0, no_indices)

            strong_above = strong_predicted_res.log().sum()
            weak_above = weak_predicted_res.log().sum()
            strong_below_1 = strong_predicted_res.flip(0).cumsum(0)
            strong_below_2 = weak_predicted_res.sum()  # weak-correlated
            strong_below_3 = no_predicted_res.sum()  # non-correlated

            strong_loss = (strong_below_1 + strong_below_2 + strong_below_3).log().sum() - strong_above
            weak_loss = (strong_below_3 - weak_above)

            tot_loss += strong_loss
            tot_loss += weak_loss

        return tot_loss * self.loss_weight


class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target):
        p_target = torch.tensor(p_target).cuda()
        trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False)
        self.trajs_mse_loss = trajs_mse_loss
        loss = sum([trajs_mse_loss])
        return loss


class default():
    def __init__(self):
        pass

    def item(self):
        return "Not Included!"


class Total_Loss(Module):
    def __init__(self):
        super(Total_Loss, self).__init__()
        self.target_loss = WeightedRankingLoss(config.batch_size, 5)
        self.source_label_loss = Distill_Rank_Loss_ListWise()
        self.source_reprensentation_loss = DiffLoss()

    def forward(self, target_pred, target_distance,
                soft_labels, predicted_distance,
                specific_rerpesentation, invariant_representtaion
                ):
        # Base Loss
        target_distance = autograd.Variable(torch.tensor(target_distance)).cuda()
        Trajs_Target_Loss = self.target_loss(target_pred, target_distance).cuda()
        Representation_loss = self.source_reprensentation_loss(specific_rerpesentation, invariant_representtaion)
        partitions = correlation_partition(soft_labels)
        Trajs_Rank_Loss = self.source_label_loss(partitions, predicted_distance)

        self.trajs_mse_loss = Trajs_Target_Loss
        self.trajs_rank_loss = Trajs_Rank_Loss
        self.trajs_latent_loss = Representation_loss
        loss = sum([Trajs_Target_Loss, Representation_loss, Trajs_Rank_Loss])
        return loss
