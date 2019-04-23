import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
import torchtext.data
import os
import io
import argparse

from encoders.encoding_helper import EncodingHelper

from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage
from helpers.calculations_helper import CalculationsHelper

import senteval
import time

PATH_TO_DATA = 'senteval/data/'
DEFAULT_MODEL_PATH = 'results/bi-lstm-max-pool/best_snapshot_devacc_37.14123376623377_devloss_1.1007678508758545__iter_34336_model.pt'
# DEFAULT_MODEL_PATH = 'results/mean/best_snapshot_devacc_60.797077922077925_devloss_0.9943482875823975__iter_25752_model.pt'
DEFAULT_EVAL_MODE = 'snli'

# Create dictionary
def create_dictionary(sentences, threshold=0):
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


def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    return word_vec


def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = glove
    params.wvec_dim = 300
    return


def batcher(params, batch): 
    batch = [sent if sent != [] else ['.'] for sent in batch]

    batch_size = len(batch)
    tokens_length = [len(tok) for tok in batch]

    words_tensor = torch.zeros(max(tokens_length), batch_size, 300)
    word_lengths = torch.tensor(tokens_length)

    for n in range(batch_size):
        for l, word in enumerate(batch[n]):
            words_tensor[l, n] = params.word_vec[word]

    embeddings = params.infersent.forward(words_tensor.cuda(), word_lengths.cuda())

    return embeddings.cpu().detach().numpy()

def perform_senteval(model, device):

    # SentEval evaluation

    params_senteval = {'task_path': PATH_TO_DATA,
                       'usepytorch': True, 'kfold': 5}

    params_senteval['classifier'] = {'nhid': 0,
                                     'optim': 'rmsprop',
                                     'batch_size': 128,
                                     'tenacity': 3,
                                     'epoch_size': 2}

    params_senteval['infersent'] = model.encoder.to(device)
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    # We use the same as those used in the paper
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'TREC', 'SST2', 'MRPC', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)


def perform_snli_eval(model, device):
    calculations_helper = CalculationsHelper()

    batch_size = 64

    # Load the data sets and the vocabulary
    print('Loading data...')
    data_storage = DataStorage()

    _ = data_storage.get_vocabulary()
    test_split_size = data_storage.get_test_split_size()

    _, _, test_iterator = data_storage.get_dataset_iterators(
        batch_size, device)

    # Switch model to evaluation mode
    model.eval()

    # Calculate the accuracy on the dev set
    test_correct_predictions = 0
    test_epoch_accuracies = []
    for test_batch in test_iterator:
        test_predictions = model.forward(test_batch)
        test_correct_predictions += calculations_helper.calculate_correct_predictions(
            test_predictions, test_batch.label)

        test_accuracy = calculations_helper.calculate_full_accuracy(
            test_predictions, test_batch.label)

        test_epoch_accuracies.append(test_accuracy)

    test_macro_accuracy = np.mean(test_epoch_accuracies)
    test_micro_accuracy = calculations_helper.calculate_accuracy(
        test_correct_predictions, test_split_size)

    print(f'test macro accuracy: {test_macro_accuracy}')
    print(f'test micro accuracy: {test_micro_accuracy}')


def main():
    device = torch.device("cuda")

    # Check if we can get the cached model. If not, raise an exception
    cache_storage = CacheStorage()

    print('Loading model...', end='')

    assert os.path.isfile(
        FLAGS.model_path), 'Model path is incorrect'

    model = cache_storage.load_model_snapshot(FLAGS.model_path)
    if not model:
        raise Exception('Model not found!')

    print('Loaded')
    print('Starting evaluation...')

    if FLAGS.eval_mode == 'snli':
        perform_snli_eval(model, device)
    elif FLAGS.eval_mode == 'senteval':
        perform_senteval(model, device)
    else:
        raise Exception('Invalid evaluation mode')


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the model file')
    parser.add_argument('--eval_mode', type=str, default=DEFAULT_EVAL_MODE,
                        help='type of evaluation: "snli" or "senteval"')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.eval_mode == 'senteval':
        glove = torchtext.vocab.GloVe()

    main()
