from encoders.base_encoder import BaseEncoder
from encoders.mean_encoder import MeanEncoder
from encoders.lstm_encoder import LSTMEncoder

class EncodingHelper():
    def __init__(self):
        pass

    def get_encoder(self, encoding_model: str) -> BaseEncoder:
        encoder = None
        if encoding_model == 'mean':
            encoder = MeanEncoder()
        elif encoding_model == 'uni-lstm' or encoding_model == 'bi-lstm' or encoding_model == 'bi-lstm-max-pool':
            bidirectional = (encoding_model == 'bi-lstm' or encoding_model == 'bi-lstm-max-pool')
            include_max_pooling = encoding_model == 'bi-lstm-max-pool'
            
            encoder = LSTMEncoder(bidirectional=bidirectional, max_pooling=include_max_pooling)
        else:
            raise Exception('Unrecognized encoding model passed')

        return encoder

    def forward_pass(self):
        pass