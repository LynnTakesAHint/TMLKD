# -*- coding: utf-8 -*-
import torch

import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

import tools.config as config
from geo_rnns.sam_cells import SAM_LSTMCell, SAM_GRUCell
from geo_rnns.t3s_model import RNNEncoder, AttentionModule


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM=False, incell=True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        if self.stard_LSTM:
            if config.recurrent_unit == 'GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
            else:
                self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cuda()
        else:
            if config.recurrent_unit == 'GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()


    def forward(self, inputs_a, initial_state=None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out = None
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state
        outputs = []
        for t in range(time_steps):
            if self.stard_LSTM:
                cell_input = inputs[:, t, :][:, :-2]
            else:
                cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out)
        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v - 1][b, :].view(1, -1))
        return torch.cat(mask_out, dim=0)

    def batch_grid_state_gates(self, inputs_a, initial_state=None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cuda()
        batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cuda()
        batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cuda()
        batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cuda()
        for t in range(time_steps):
            cell_input = inputs[:, t, :]
            self.cell.batch_update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)


class NeuTraj_Network(nn.Module):
    def __init__(self, input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM=False, incell=True):
        super(NeuTraj_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                            requires_grad=False).cuda()
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda(),
                           autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda())
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                              incell=incell).cuda()

    def forward(self, inputs, lens):
        anchor_input, pos_input, neg_input, _ = inputs
        anchor_input = torch.tensor(anchor_input, dtype=torch.float32).clone()
        pos_input = torch.tensor(pos_input, dtype=torch.float32).clone()
        neg_input = torch.tensor(neg_input, dtype=torch.float32).clone()
        anchor_len, pos_len, neg_len, _ = lens
        anchor_embedding = self.get_embeddings(anchor_input, anchor_len, self.hidden.clone()).clone()
        positive_trajs_embedding = self.get_embeddings(pos_input, pos_len, self.hidden.clone()).clone()
        negative_trajs_embedding = self.get_embeddings(neg_input, neg_len, self.hidden.clone()).clone()
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), positive_trajs_embedding.clone(), p=2))
        neg_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), negative_trajs_embedding.clone(), p=2))
        return (anchor_embedding, positive_trajs_embedding, negative_trajs_embedding), (pos_distance, neg_distance)

    def get_embeddings(self, trajs_input, lens, hidden=None):
        return self.rnn([autograd.Variable(trajs_input.clone(), requires_grad=False).cuda(), lens],
                        hidden)

    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays
        batch_hidden = (
            autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda(),
            autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda())
        self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cuda(), batch_traj_len], batch_hidden)


class NeuTraj_Share_Network(nn.Module):
    def __init__(self, input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM=False, incell=True):
        super(NeuTraj_Share_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                            requires_grad=False).cuda()
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda(),
                           autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                             requires_grad=False).cuda())
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                              incell=incell).cuda()
        self.multihead_attn = AttentionModule()

    def forward(self, inputs, lens):
        anchor_input, pos_input, neg_input, _ = inputs
        anchor_input = torch.tensor(anchor_input, dtype=torch.float32)
        pos_input = torch.tensor(pos_input, dtype=torch.float32)
        neg_input = torch.tensor(neg_input, dtype=torch.float32)
        anchor_len, pos_len, neg_len, _ = lens
        anchor_embedding = self.get_embeddings(anchor_input, anchor_len, self.hidden.clone())
        positive_trajs_embedding = self.get_embeddings(pos_input, pos_len, self.hidden.clone())
        negative_trajs_embedding = self.get_embeddings(neg_input, neg_len, self.hidden.clone())
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embedding, positive_trajs_embedding, p=2))
        neg_distance = torch.exp(-F.pairwise_distance(anchor_embedding, negative_trajs_embedding, p=2))
        return (anchor_embedding, positive_trajs_embedding, negative_trajs_embedding), (pos_distance, neg_distance)

    def get_embeddings(self, trajs_input, lens, hidden=None):
        first_embeddings = self.rnn([autograd.Variable(trajs_input, requires_grad=False).cuda(), lens], hidden)
        second_embeddings = self.multihead_attn(first_embeddings)
        return second_embeddings

    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays
        batch_hidden = (
            autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda(),
            autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda())
        self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cuda(), batch_traj_len], batch_hidden)


class NeuTraj_Decoder(nn.Module):
    def __init__(self, input_size=config.d, target_size=config.d):
        super(NeuTraj_Decoder, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True)).cuda()
        self.share_linear = nn.Linear(input_size, target_size)
        self.private_linear = nn.Linear(input_size, target_size)

    def forward(self, share_embeddings, private_embeddings):
        anchor_share_embedding_input, pos_share_embedding_input, neg_share_embedding_input = share_embeddings
        anchor_private_embedding_input, pos_private_embedding_input, neg_private_embedding_input = private_embeddings
        anchor_embedding = self.get_embeddings(anchor_share_embedding_input, anchor_private_embedding_input)
        pos_embedding = self.get_embeddings(pos_share_embedding_input, pos_private_embedding_input)
        neg_embedding = self.get_embeddings(neg_share_embedding_input, neg_private_embedding_input)
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embedding, pos_embedding, p=2))
        neg_distance = torch.exp(-F.pairwise_distance(anchor_embedding, neg_embedding, p=2))
        return pos_distance, neg_distance

    def get_embeddings(self, share_embedding, private_embedding):
        share_embedding = self.share_linear(share_embedding)
        private_embedding = self.private_linear(private_embedding)
        return self.beta * share_embedding + (1 - self.beta) * private_embedding