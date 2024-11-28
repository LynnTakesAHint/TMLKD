import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import tools.config as config
from geo_rnns.Embedding_layer import FixedAbsolutePositionEmbedding


def list2tensor(a):
    return torch.tensor([item.cpu().detach().numpy() for item in a]).cuda()


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(RNNEncoder, self).__init__()
        self.input_size = 2
        self.hidden_size = hidden_size
        if config.recurrent_unit == 'GRU':
            self.cell = torch.nn.GRUCell(self.input_size, hidden_size).cuda()
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


class T3S_Network(nn.Module):
    def __init__(self, target_size, batch_size, sampling_num):
        super(T3S_Network, self).__init__()
        self.target_size = target_size
        self.sampling_num = sampling_num
        self.teacher_batch_size = 50 * batch_size
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
        self.lookup = nn.Linear(2, self.target_size)
        self.position_embedding = FixedAbsolutePositionEmbedding(max_position_embeddings=config.max_len, hidden_size=2,
                                                                 position_embedding_type='fixed')
        self.position_result = nn.Linear(2, self.target_size)
        self.multihead_attn_1 = nn.MultiheadAttention(128, 16, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor(0.5, requires_grad=True)).cuda()

    def forward(self, inputs, lens):
        anchor_input, pos_input, neg_input = inputs
        anchor_coor, anchor_grid = anchor_input[:, :, :2], anchor_input[:, :, 2:]
        pos_coor, pos_grid = pos_input[:, :, :2], pos_input[:, :, 2:]
        neg_coor, neg_grid = neg_input[:, :, :2], neg_input[:, :, 2:]
        anchor_len, pos_len, neg_len = lens
        anchor_embedding = self.get_embeddings(anchor_coor, anchor_grid, anchor_len, self.hidden.clone()).clone()
        positive_trajs_embedding = self.get_embeddings(pos_coor, pos_grid, pos_len, self.hidden.clone()).clone()
        negative_trajs_embedding = self.get_embeddings(neg_coor, neg_grid, neg_len, self.hidden.clone()).clone()
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), positive_trajs_embedding.clone(), p=2))
        neg_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), negative_trajs_embedding.clone(), p=2))
        return (anchor_embedding, positive_trajs_embedding, negative_trajs_embedding), (pos_distance, neg_distance)

    def get_embeddings(self, coor, grid, lens, hidden=None):
        coor = autograd.Variable(torch.Tensor(coor), requires_grad=False).cuda()
        grid = autograd.Variable(torch.Tensor(grid), requires_grad=False).cuda()
        coor_embs = self.rnn(coor, lens, hidden)
        sigmoid_grids = self.lookup(grid)
        pos_grids = self.position_result(self.position_embedding(grid))
        m_grids = sigmoid_grids + pos_grids
        first_attn_embs = self.multihead_attn_1(m_grids, m_grids, m_grids,
                                                need_weights=False)[0]
        grid_embs = torch.mean(first_attn_embs, dim=1)
        final_embeddings = self.gamma * coor_embs + (1 - self.gamma) * grid_embs
        return final_embeddings


class AttentionModule(nn.Module):
    def __init__(self, input_dim = 128, embed_dim = 128):
        super(AttentionModule, self).__init__()
        self.embed_dim = embed_dim
        self.W_Q = nn.Linear(input_dim, embed_dim)
        self.W_K = nn.Linear(input_dim, embed_dim)
        self.W_V = nn.Linear(input_dim, embed_dim)

    def forward(self, s_input):
        s_hat = s_input.unsqueeze(1)
        Q = self.W_Q(s_hat)  # (batch_size, seq_len, embed_dim)
        K = self.W_K(s_hat)  # (batch_size, seq_len, embed_dim)
        V = self.W_V(s_hat)  # (batch_size, seq_len, embed_dim)

        K_T = K.transpose(-1, -2)
        attention_scores = torch.bmm(Q, K_T)  # (batch_size, seq_len, seq_len)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))  # 缩放

        A = F.softmax(attention_scores, dim=-1)
        s = torch.bmm(A, V)  # (batch_size, seq_len, embed_dim)

        return s.squeeze(1)


class T3S_Share_Network(nn.Module):
    def __init__(self, target_size, batch_size, sampling_num):
        super(T3S_Share_Network, self).__init__()
        self.target_size = target_size
        self.sampling_num = sampling_num
        self.teacher_batch_size = 50 * batch_size
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
        self.lookup = nn.Linear(2, self.target_size)
        self.position_embedding = FixedAbsolutePositionEmbedding(max_position_embeddings=config.max_len, hidden_size=2,
                                                                 position_embedding_type='fixed')
        self.position_result = nn.Linear(2, self.target_size)
        self.multihead_attn_1 = nn.MultiheadAttention(128, 16, batch_first=True)
        self.multihead_attn_2 = AttentionModule()
        self.gamma = nn.Parameter(torch.tensor(0.5, requires_grad=True)).cuda()

    def forward(self, inputs, lens):
        anchor_input, pos_input, neg_input = inputs
        anchor_coor, anchor_grid = anchor_input[:, :, :2], anchor_input[:, :, 2:]
        pos_coor, pos_grid = pos_input[:, :, :2], pos_input[:, :, 2:]
        neg_coor, neg_grid = neg_input[:, :, :2], neg_input[:, :, 2:]
        anchor_len, pos_len, neg_len = lens
        anchor_embedding = self.get_embeddings(anchor_coor, anchor_grid, anchor_len, self.hidden.clone())
        positive_trajs_embedding = self.get_embeddings(pos_coor, pos_grid, pos_len, self.hidden.clone())
        negative_trajs_embedding = self.get_embeddings(neg_coor, neg_grid, neg_len, self.hidden.clone())
        pos_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), positive_trajs_embedding.clone(), p=2))
        neg_distance = torch.exp(-F.pairwise_distance(anchor_embedding.clone(), negative_trajs_embedding.clone(), p=2))
        return (anchor_embedding, positive_trajs_embedding, negative_trajs_embedding), (pos_distance, neg_distance)

    def get_embeddings(self, coor, grid, lens, hidden=None):
        coor = autograd.Variable(torch.Tensor(coor), requires_grad=False).cuda()
        grid = autograd.Variable(torch.Tensor(grid), requires_grad=False).cuda()
        coor_embs = self.rnn(coor, lens, hidden)
        sigmoid_grids = self.lookup(grid)
        pos_grids = self.position_result(self.position_embedding(grid))
        m_grids = sigmoid_grids + pos_grids
        first_attn_embs = self.multihead_attn_1(m_grids, m_grids, m_grids,
                                                need_weights=False)[0]
        grid_embs = torch.mean(first_attn_embs, dim=1)
        final_embeddings = self.gamma * coor_embs + (1 - self.gamma) * grid_embs
        second_attn_embs = self.multihead_attn_2(final_embeddings)
        return second_attn_embs


class T3S_Decoder(nn.Module):
    def __init__(self, input_size=config.d, target_size=config.d):
        super(T3S_Decoder, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True)).cuda()
        self.share_Linear = nn.Linear(input_size, target_size)
        self.private_Linear = nn.Linear(input_size, target_size)

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
        share_embedding = self.share_Linear(share_embedding)
        private_embedding = self.private_Linear(private_embedding)
        return self.beta * share_embedding + (1 - self.beta) * private_embedding
