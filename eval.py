import numpy as np

import torch
import torch.optim as optim
import os

from encoders.encoding_helper import EncodingHelper

from helpers.calculations_helper import CalculationsHelper
from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage
from helpers.parameters_helper import ParametersHelper

# Create dictionary
MODEL_PATH = 'results/uni-lstm/best_snapshot_devacc_34.52150974025974_devloss_1.0959851741790771__iter_25752_model.pt'

assert os.path.isfile(MODEL_PATH), 'Set MODEL and GloVe PATHs'

calculations_helper = CalculationsHelper()

# Load input arguments
parameters_helper = ParametersHelper()
parameters_helper.load_arguments()

device = torch.device("cuda")

# Check if we can get the cached model. If not, raise an exception
cache_storage = CacheStorage()

print('Loading model...', end='')
model = cache_storage.load_model_snapshot(MODEL_PATH)
if not model:
    raise Exception('Model not found!')

print('Loaded model')

# Load the data sets and the vocabulary
print('Loading data...')
data_storage = DataStorage()

token_vocabulary, _ = data_storage.get_vocabulary()
test_split_size = data_storage.get_test_split_size()

print('Starting evaluation...')

_, _, test_iterator = data_storage.get_dataset_iterators(
    parameters_helper.batch_size, device)

# Switch model to evaluation mode
model.eval()

# Calculate the accuracy on the dev set
test_correct_predictions, test_loss = 0, 0
test_epoch_accuracies = []
for test_batch_idx, test_batch in enumerate(test_iterator):
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