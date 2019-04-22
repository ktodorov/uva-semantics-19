import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
import torchtext.data
import os
import io

from encoders.encoding_helper import EncodingHelper

from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage
from helpers.parameters_helper import ParametersHelper

from snli_classifier import SNLIClassifier

import senteval
import time

# Create dictionary
PATH_TO_DATA = 'senteval/data/'
PATH_TO_GLOVE = '.vector_cache/glove.840B.300d.txt.pt'
MODEL_PATH = 'results/uni-lstm/best_snapshot_devacc_34.52150974025974_devloss_1.0959851741790771__iter_25752_model.pt'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_GLOVE), \
    'Set MODEL and GloVe PATHs'

glove = torchtext.vocab.GloVe()

def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    # inverse sort
    sorted_words = sorted(words.items(), key=lambda x: -x[1])
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

def prepare(params, samples):
    # params.inputs.build_vocab(samples)
    # params.inputs.vocab.load_vectors('glove.840B.300d')
    params.id2word, params.word2id = create_dictionary(samples)
    # set glove as the embedding model
    params.word_vec = glove
    params.wvec_dim = 300


def batcher(params, batch):
    sentences = []
    for s in batch:
        sentence = params.inputs.preprocess(s)
        sentences.append(sentence)

    sentences = params.inputs.process(sentences, train=True, device=0)
    params.hbmp = params.hbmp.cuda()
    emb = params.hbmp.forward(sentences.cuda())
    embeddings = []

    for sent in emb:
        sent = sent.cpu()
        embeddings.append(sent.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

# Load input arguments
parameters_helper = ParametersHelper()
parameters_helper.load_arguments()

device = torch.device("cuda")

# Check if we can get the cached model. If not, raise an exception
cache_storage = CacheStorage()

print('Loading model...', end='')
model = cache_storage.load_model_snapshot(MODEL_PATH)# parameters_helper.snapshot_location)
if not model:
    raise Exception('Model not found!')

print('Loaded model')

# Load the data sets and the vocabulary
print('Loading data...')
data_storage = DataStorage()

token_vocabulary, _ = data_storage.get_vocabulary()
test_split_size = data_storage.get_test_split_size()

print('Starting evaluation...')

# SentEval evaluation

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}

params_senteval['classifier'] = {'nhid': 0, 
                'optim': 'rmsprop', 
                'batch_size': 128,
                'tenacity': 3,
                'epoch_size': 2}

if __name__ == "__main__":

    params_senteval['infersent'] = model.encoder.to(device)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    transfer_tasks = 'MR'
    #  ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
    #                   'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    #                   'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    # ['MR', 'CR', 'SUBJ', 'MPQA', 'STSBenchmark', 'SST2', 'SST5', 'TREC', 'MRPC',
    #  'SICKRelatedness', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)