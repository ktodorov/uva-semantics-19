import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

        self._input_dimensions = 0

    @property
    def input_dimensions(self):
        return self._input_dimensions