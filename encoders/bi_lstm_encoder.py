import torch
from encoders.base_encoder import BaseEncoder


class BiLSTMEncoder(BaseEncoder):
    def __init__(self, use_max_pooling: bool = False):
        super(BiLSTMEncoder, self).__init__()
