from torchnlp.datasets import snli_dataset

from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.text import WhitespaceEncoder

from torchnlp.word_to_vector import GloVe

class DataStorage:
    def __init__(self):
        self.train_set = None
        self.dev_set = None
        self.test_set = None

        self.train_encoder = None
        self.dev_encoder = None
        self.test_encoder = None

    def get_snli_encoders(self, data_cap: int = None):
        if not self.train_set or not self.dev_set or not not self.test_set:
            self._load_data_sets(data_cap)

        self.train_encoder = WhitespaceEncoder(
            [row['premise'] for row in self._dataset_iterator(self.train_set)] + [row['hypothesis'] for row in self._dataset_iterator(self.train_set)])

        return self.train_encoder

    def get_label_encoder(self, data_cap: int = None):
        if not self.train_set or not self.dev_set or not not self.test_set:
            self._load_data_sets(data_cap)
        
        self.train_labels = [row['label'] for row in self._dataset_iterator(self.train_set)]
        self.train_label_encoder = LabelEncoder(self.train_labels)

        return self.train_label_encoder

    def get_vocabulary(self, word_tokens):
        glove_vectors = GloVe()

        vocabulary = {wt: glove_vectors[wt] for wt in word_tokens.vocab}
        return vocabulary

    def _load_data_sets(self, data_cap: int=None):
        self.train_set, self.dev_set, self.test_set = snli_dataset(
            train=True, dev=True, test=True)

        if data_cap:
            self.train_set = self.train_set[0:data_cap]
            self.dev_set = self.dev_set[0:data_cap]
            self.test_set = self.test_set[0:data_cap]

    def _dataset_iterator(self, dataset):
        for row in dataset:
            yield row
