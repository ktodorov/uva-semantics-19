import torch
from encoders.base_encoder import BaseEncoder

class MeanEncoder(BaseEncoder):
    def __init__(self):
        super(MeanEncoder, self).__init__()

        self._input_dimensions = 4*300

    def forward(self, x, x_len):
        out = torch.div(torch.sum(x, 0), x_len.float().view(-1, 1))

        return out
