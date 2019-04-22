import torch
from torch import nn
from torch.autograd import Variable

from encoders.base_encoder import BaseEncoder

class LSTMEncoder(BaseEncoder):
    def __init__(
            self,
            embedding_dim=300,
            hidden_dim=2048,
            bidirectional=False,
            max_pooling=False):
        super(LSTMEncoder, self).__init__()

        self._input_dimensions = 4*hidden_dim

        if bidirectional:
            self._input_dimensions *= 2

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = hidden_dim
        self.bidirectional = bidirectional
        self.max_pooling = max_pooling

        if self.bidirectional:
            self.output_size *= 2

        self.model = nn.LSTM(input_size=self.embedding_dim,
                             hidden_size=self.hidden_dim,
                             bidirectional=bidirectional)

    def forward(self, x, len_x):
        out = self.transform_and_calculate(x, len_x)
        return out

    def transform_and_calculate(self, x, len_x):
        len_x_sorted, idx = torch.sort(len_x, 0, descending=True)

        x_select = x.index_select(1, Variable(idx))

        x_packed = nn.utils.rnn.pack_padded_sequence(x_select, len_x_sorted)
        x_output = self.model(x_packed)[0]

        x_output = nn.utils.rnn.pad_packed_sequence(x_output)[0]

        if self.max_pooling:
            x_output, _ = torch.max(x_output, dim=0)
            return x_output

        original_indices = torch.tensor([idx[i] for i in idx]).cuda()
        x_output = x_output.index_select(1, Variable(original_indices))

        unsorted_lens = len_x_sorted.index_select(0, original_indices)
        x_output = torch.stack([x_output[item_length - 1, i, :] for i, item_length in enumerate(unsorted_lens)])

        return x_output