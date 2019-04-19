import torch
from encoders.base_encoder import BaseEncoder


class UniLSTMEncoder(BaseEncoder):
    def __init__(self):
        super(UniLSTMEncoder, self).__init__()

        self._input_dimensions = 4*2048
