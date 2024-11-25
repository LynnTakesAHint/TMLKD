import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Module, Linear, MultiheadAttention, Parameter

import tools.config as config
from geo_rnns.Embedding_layer import FixedAbsolutePositionEmbedding


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


class T3S_Network(Module):
    def __init__(self, target_size, batch_size, sampling_num,
                 last_num=(config.train_size % config.batch_size)):
        super(T3S_Network, self).__init__()
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

        self.rnn = RNNEncoder(self.target_size).cuda()
        self.lookup = Linear(2, self.target_size).cuda()
        self.position_embedding = FixedAbsolutePositionEmbedding(max_position_embeddings=config.max_len, hidden_size=2,
                                                                 position_embedding_type='fixed').cuda()
        self.position_result = Linear(2, self.target_size).cuda()
        self.multihead_attn_1 = MultiheadAttention(self.target_size, 16, batch_first=True).cuda()
        self.multihead_attn_2 = MultiheadAttention(self.target_size, 16, batch_first=True).cuda()
        self.gamma = Parameter(torch.tensor(0.5, requires_grad=True)).cuda()
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

    def get_embeddings(self, coor, grid, length, inva=None, hidden=None):
        coor = autograd.Variable(torch.Tensor(coor), requires_grad=False).cuda()
        grid = autograd.Variable(torch.Tensor(grid), requires_grad=False).cuda()
        coor_embs = self.rnn(coor, length, hidden)
        sigmoid_grids = self.lookup(grid)
        pos_grids = self.position_result(self.position_embedding(grid))
        m_grids = sigmoid_grids + pos_grids
        first_attn_embs = self.multihead_attn_1(m_grids, m_grids, m_grids,
                                                need_weights=False)[0]
        grid_embs = torch.mean(first_attn_embs, dim=1)
        final_embeddings = self.gamma * coor_embs + (1 - self.gamma) * grid_embs
        inva = inva.cuda()
        inva = self.share_projection(inva)
        add_share_embeddings = self.alpha * final_embeddings + (1 - self.alpha) * inva
        return add_share_embeddings, final_embeddings
