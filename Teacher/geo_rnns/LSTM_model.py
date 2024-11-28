# -*- coding: utf-8 -*-
import torch

import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Module
import tools.config as config
import torch.nn as nn


class RNNEncoder(Module):
    def __init__(self, hidden_size):
        super(RNNEncoder, self).__init__()
        self.input_size = 2
        self.hidden_size = hidden_size
        if config.recurrent_unit == 'GRU':
            self.cell = torch.nn.GRUCell(self.input_size, hidden_size).cuda()
        elif config.recurrent_unit == 'SimpleRNN':
            self.cell = torch.nn.RNNCell(self.input_size, hidden_size).cuda()
        else:
            self.cell = torch.nn.LSTMCell(self.input_size, hidden_size).cuda()

    def forward(self, inputs, input_len, initial_state=None):
        time_steps = inputs.size(1)
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state
        outputs = []
        for t in range(time_steps):
            cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out)
        final_outputs = []
        for b, v in enumerate(input_len):
            final_outputs.append(outputs[v - 1][b, :].view(1, -1))
        final_outputs = torch.cat(final_outputs, dim=0)
        return final_outputs


class LSTM_Network(Module):
    def __init__(self, target_size, batch_size, sampling_num,
                 last_num=(config.train_size % config.batch_size)):
        super(LSTM_Network, self).__init__()
        self.target_size = target_size
        self.sampling_num = sampling_num
        self.teacher_batch_size = 50 * batch_size
        self.last_batch_size = last_num * (1 + sampling_num)
        self.batch_size = batch_size * (1 + sampling_num)
        self.sampling_num = sampling_num
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            self.hidden = autograd.Variable(torch.zeros(self.batch_size, self.target_size),
                                            requires_grad=False).cuda()
            self.teacher_hidden = autograd.Variable(torch.zeros(self.teacher_batch_size, self.target_size),
                                                    requires_grad=False).cuda()
            self.last_hidden = autograd.Variable(torch.zeros(self.teacher_batch_size, self.target_size),
                                                 requires_grad=False).cuda()
        else:
            self.hidden = (autograd.Variable(torch.zeros(self.batch_size, self.target_size),
                                             requires_grad=False).cuda(),
                           autograd.Variable(torch.zeros(self.batch_size, self.target_size),
                                             requires_grad=False).cuda())

        self.rnn1 = RNNEncoder(self.target_size).cuda()
        self.rnn2 = RNNEncoder(self.target_size).cuda()
        self.linear = nn.Linear(self.target_size * 2, self.target_size)

    def forward(self, coors, grids, lens, last=False, teacher_flag=False):
        anchor_coor_input, pos_coor_input = coors
        anchor_grid_input, pos_grid_input = grids
        anchor_len, pos_len = lens
        if teacher_flag:
            hidden = self.teacher_hidden
        elif last:
            hidden = self.last_hidden
        else:
            hidden = self.hidden

        anchor_embs = self.get_embeddings(anchor_coor_input, anchor_grid_input, anchor_len, hidden)
        pos_embs = self.get_embeddings(pos_coor_input, pos_grid_input, pos_len, hidden)
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embs, pos_embs, p=2))
        return pos_distance, pos_embs

    def get_embeddings(self, coor, grid, length, hidden=None):
        coor = autograd.Variable(torch.Tensor(coor), requires_grad=False).cuda()
        grid = autograd.Variable(torch.Tensor(grid), requires_grad=False).cuda()
        coor_embs = self.rnn1(coor, length, hidden)
        grid_embs = self.rnn2(grid, length, hidden)
        final_embeddings = self.linear(torch.cat([coor_embs, grid_embs], dim=1))
        return final_embeddings
