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

import senteval
import time

# Create dictionary
PATH_TO_DATA = 'senteval/data/'
PATH_TO_GLOVE = 'data/glove/glove.840B.300d.txt'
MODEL_PATH = 'results/uni-lstm/best_snapshot_devacc_34.52150974025974_devloss_1.0959851741790771__iter_25752_model.pt'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_GLOVE), \
    'Model and/or GloVe path is incorrect'

glove = torchtext.vocab.GloVe()

# Create dictionary
def create_dictionary(sentences, threshold=0):
    """function that creates a dictionary, stollen form SentEval"""
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    """function that returns the embeddings, stollen form SentEval"""
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    return word_vec

def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_GLOVE, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

# Load input arguments
device = torch.device("cuda")

# Check if we can get the cached model. If not, raise an exception
cache_storage = CacheStorage()

print('Loading model...', end='')
model = cache_storage.load_model_snapshot(MODEL_PATH)# parameters_helper.snapshot_location)
if not model:
    raise Exception('Model not found!')

print('Loaded')

print('Starting evaluation...')

# SentEval evaluation

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}

params_senteval['classifier'] = {'nhid': 0, 
                'optim': 'adam', 
                'batch_size': 64,
                'tenacity': 3,
                'epoch_size': 2}

if __name__ == "__main__":

    params_senteval['infersent'] = model.encoder.to(device)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    transfer_tasks = [ 'MR', 'CR', 'SUBJ', 'MPQA', 'TREC', 'SST']
    #  ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
    #                   'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
    #                   'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    # ['MR', 'CR', 'SUBJ', 'MPQA', 'STSBenchmark', 'SST2', 'SST5', 'TREC', 'MRPC',
    #  'SICKRelatedness', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)