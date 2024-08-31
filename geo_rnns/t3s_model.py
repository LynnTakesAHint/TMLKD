import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Module
import tools.config as config
import torch.nn as nn
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
                 last_num = (config.train_size % config.batch_size)):
        super(T3S_Network, self).__init__()
        # under construction: Will be organized before publication
        pass

    def forward(self, coors, grids, lens, invariant_representation, last=False, teacher_flag=False):
        # under construction: Will be organized before publication
        pass
    def get_embeddings(self, coor, grid, length, inva=None, hidden = None):
        # under construction: Will be organized before publication
        pass