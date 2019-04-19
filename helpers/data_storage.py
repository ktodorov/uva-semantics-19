from torchtext import data, datasets


class DataStorage:
    def __init__(self):
        self.text_field = data.Field(
            lower=False,
            tokenize=lambda x: x.split(),
            include_lengths=True,
            use_vocab=True)
            
        self.label_field = data.Field(
            sequential=False,
            use_vocab=True,
            pad_token=None,
            unk_token=None,
            batch_first=None)

        self.train_split, self.dev_split, self.test_split = datasets.SNLI.splits(self.text_field, self.label_field)

    def get_vocabulary(self, vectors_name='glove.840B.300d'):
        self.text_field.build_vocab(self.train_split, self.dev_split, self.test_split)
        
        
        # if os.path.isfile(args.vector_cache):
        #     inputs.vocab.vectors = torch.load(args.vector_cache)
        # else:
        # makedirs(os.path.dirname(args.vector_cache))
        # torch.save(inputs.vocab.vectors, args.vector_cache)
        self.text_field.vocab.load_vectors(vectors_name)
        self.label_field.build_vocab(self.train_split)

        return self.text_field.vocab, self.label_field.vocab

    def get_dataset_iterators(self, batch_size):
        train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_split, self.dev_split, self.test_split), batch_size=batch_size)  # , device=device)

        return train_iterator, dev_iterator, test_iterator

    def get_train_split_size(self) -> int:
        return len(self.train_split)

    def get_dev_split_size(self) -> int:
        return len(self.dev_split)
        
    def get_test_split_size(self) -> int:
        return len(self.test_split)