from nltk import word_tokenize
import numpy as np

import torch
import torchtext
import torchtext.data
import os
import io

from encoders.encoding_helper import EncodingHelper

from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage

from inference_model import InferenceModel

MODEL_PATH = 'results/mean/best_snapshot_devacc_60.797077922077925_devloss_0.9943482875823975__iter_25752_model.pt'

def transform_sentence(sentence, w2i_dict, device):
    sentence = word_tokenize(sentence)
    indexes = torch.tensor([w2i_dict[word] for word in sentence]).to(device)
    length = torch.Tensor([len(indexes)]).to(device)

    return indexes, length

def initialize_data(model_path):
    assert os.path.isfile(model_path), 'Model path is not valid'
    device = torch.device("cuda")

    # Check if we can get the cached model. If not, raise an exception
    cache_storage = CacheStorage()

    print('Loading model...', end='')
    # parameters_helper.snapshot_location)
    model = cache_storage.load_model_snapshot(model_path)
    if not model:
        raise Exception('Model not found!')

    print('Loaded')

    # Load the data sets and the vocabulary
    print('Loading data...', end='')

    data_storage = DataStorage()
    token_vocabulary, _ = data_storage.get_vocabulary()

    print('Loaded')

    label_dictionary = {
        0: "entails",
        1: "contradicts",
        2: "is neutral to"
    }

    return device, model, token_vocabulary, label_dictionary

def calculate_inference(model, token_vocabulary, label_dictionary, device, premise, hypothesis):
    premise, premise_length = transform_sentence(
        premise, token_vocabulary.stoi, device)

    hypothesis, hypothesis_length = transform_sentence(
        hypothesis, token_vocabulary.stoi, device)

    inference_model = InferenceModel(
        premise.expand(1, -1).transpose(0, 1),
        premise_length,
        hypothesis.expand(1, -1).transpose(0, 1),
        hypothesis_length)

    model_prediction = model.forward(inference_model)

    print(
        f"The premise {label_dictionary[model_prediction.argmax().item()]} the hypothesis")

device, model, token_vocabulary, label_dictionary = initialize_data(MODEL_PATH)

premise = input('Enter premise:\n')
hypothesis = input('Enter hypothesis:\n')

calculate_inference(model, token_vocabulary,
                    label_dictionary, device, premise, hypothesis)
