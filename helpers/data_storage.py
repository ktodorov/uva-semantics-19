import torchtext
from torchtext import data, datasets

class DataStorage:
    def __init__(self, max_samples:int=None):
        self.text_field = data.Field(sequential=True,
                                  tokenize=lambda x: x.split(),
                                  include_lengths=True,
                                  use_vocab=True)
            
        self.label_field = data.Field(sequential=False,
                                   use_vocab=True,
                                   pad_token=None,
                                   unk_token=None)

        self.train_split, self.dev_split, self.test_split = datasets.SNLI.splits(self.text_field, self.label_field)
    
        if max_samples:
            train_end = max_samples if len(self.train_split) > max_samples else len(self.train_split)
            self.train_split.examples = self.train_split.examples[:train_end]

            dev_end = max_samples if len(self.dev_split) > max_samples else len(self.dev_split)
            self.dev_split.examples = self.dev_split.examples[:dev_end]

            test_end = max_samples if len(self.test_split) > max_samples else len(self.test_split)
            self.test_split.examples = self.test_split.examples[:test_end]

    def get_vocabulary(self, vectors_name='840B', dimensions=300):
        glove = torchtext.vocab.GloVe(name=vectors_name, dim=dimensions)
        self.text_field.build_vocab(self.train_split, self.dev_split, self.test_split, vectors=glove)
        self.label_field.build_vocab(self.test_split)

        return self.text_field.vocab, self.label_field.vocab

    def get_dataset_iterators(self, batch_size, device):
        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_split, self.dev_split, self.test_split), batch_size=batch_size, device=device, shuffle=True)

        return train_iterator, dev_iterator, test_iterator

    def get_train_split_size(self) -> int:
        return len(self.train_split)

    def get_dev_split_size(self) -> int:
        return len(self.dev_split)
        
    def get_test_split_size(self) -> int:
        return len(self.test_split)