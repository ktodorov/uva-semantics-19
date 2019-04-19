import torch
from encoders.base_encoder import BaseEncoder


class MeanEncoder(BaseEncoder):
    def __init__(self):
        super(MeanEncoder, self).__init__()

    def forward(self, x, x_len):
        out = torch.sum(x, dim=0) / x_len.view(-1, 1).to(torch.float)

        return out
