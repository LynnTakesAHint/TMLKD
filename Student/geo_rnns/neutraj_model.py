import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter

import tools.config as config
from geo_rnns.sam_cells import SAM_LSTMCell, SAM_GRUCell


def list2tensor(a):
    return torch.tensor([item.cpu().detach().numpy() for item in a]).cuda()


class RNNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM=False, incell=True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        if self.stard_LSTM:
            if config.recurrent_unit == 'GRU':
                self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
            elif config.recurrent_unit == 'SimpleRNN':
                self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cuda()
            else:
                self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cuda()
        else:
            if config.recurrent_unit == 'GRU':
                self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
            elif config.recurrent_unit == 'SimpleRNN':
                raise NotImplementedError
            else:
                self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()

        print(self.cell)
        print('in cell update: {}'.format(incell))

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
        batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cuda()
        batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cuda()
        batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cuda()
        batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cuda()
        for t in range(time_steps):
            cell_input = inputs[:, t, :]
            self.cell.batch_update_memory(cell_input, (out, state),
                                          batch_weight_ih, batch_weight_hh,
                                          batch_bias_ih, batch_bias_hh)


class NeuTraj_Network(Module):
    def __init__(self, input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM=False, incell=True):
        super(NeuTraj_Network, self).__init__()
        last_num = (config.train_size % config.batch_size)
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
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
        self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                              incell=incell).cuda()
        self.share_projection = Linear(config.teacher_dim, self.target_size).cuda()
        self.alpha = Parameter(torch.tensor(0.5, requires_grad=True)).cuda()

    def forward(self, coors, grids, lens, invariant_representation, last=False, teacher_flag=False):
        anchor_coor_input, pos_coor_input = coors
        anchor_grid_input, pos_grid_input = grids
        anchor_len, pos_len = lens
        anchor_inva, pos_inva = invariant_representation
        if teacher_flag:
            hidden = self.teacher_hidden
        elif last:
            hidden = self.last_hidden
        else:
            hidden = self.hidden
        anchor_embs, _ = self.get_embeddings(anchor_coor_input, anchor_grid_input, anchor_len, anchor_inva, hidden)
        pos_embs, pos_specific = self.get_embeddings(pos_coor_input, pos_grid_input, pos_len, pos_inva, hidden)
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embs, pos_embs, p=2))
        return pos_distance, pos_specific

    def get_embeddings(self, coors, grids, lens, inva=None, hidden=None):
        trajs_input = torch.tensor(np.concatenate([coors, grids], axis=2).astype(np.float32))
        final_embeddings = self.rnn([autograd.Variable(trajs_input, requires_grad=False).cuda(), lens],
                                    hidden)
        inva = inva.cuda()
        inva = self.share_projection(inva)
        add_share_embeddings = self.alpha * final_embeddings + (1 - self.alpha) * inva
        return add_share_embeddings, final_embeddings
