from encoders.base_encoder import BaseEncoder
from encoders.mean_encoder import MeanEncoder
from encoders.uni_lstm_encoder import UniLSTMEncoder
from encoders.bi_lstm_encoder import BiLSTMEncoder

class EncodingHelper():
    def __init__(self):
        pass

    def get_encoder(self, encoding_model: str) -> BaseEncoder:
        encoder = None
        if encoding_model == 'mean':
            encoder = MeanEncoder()
        elif encoding_model == 'uni-lstm':
            encoder = UniLSTMEncoder()
        elif encoding_model == 'bi-lstm' or encoding_model == 'bi-lstm-max-pool':
            include_max_pooling = encoding_model == 'bi-lstm-max-pool'
            encoder = BiLSTMEncoder(include_max_pooling)
        else:
            raise Exception('Unrecognized encoding model passed')

        return encoder

    def forward_pass(self):
        pass