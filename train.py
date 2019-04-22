import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from encoders.encoding_helper import EncodingHelper

from helpers.calculations_helper import CalculationsHelper
from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage
from helpers.parameters_helper import ParametersHelper
from helpers.plot_helper import PlotHelper
from helpers.statistics_helper import StatisticsHelper

from snli_classifier import SNLIClassifier

calculations_helper = CalculationsHelper()

minimum_learning_rate = 1e-5

# Load input arguments
parameters_helper = ParametersHelper()
parameters_helper.load_arguments()
parameters_helper.print_arguments()

device = torch.device("cuda")

# Load the data sets and the vocabulary
print('Loading data...')
data_storage = DataStorage(parameters_helper.max_samples)

token_vocabulary, _ = data_storage.get_vocabulary()
# train_iterator, dev_iterator, test_iterator = data_storage.get_dataset_iterators(parameters_helper.batch_size, device)
dev_split_size = data_storage.get_dev_split_size()

# Check if we can get a cached model. If not, create a new one
cache_storage = CacheStorage(parameters_helper.encoding_model)

print('Loading model...')
model = cache_storage.load_model_snapshot(parameters_helper.snapshot_location)
if not model:
    encoding_helper = EncodingHelper()
    encoder = encoding_helper.get_encoder(parameters_helper.encoding_model)
    model = SNLIClassifier(encoder, token_vocabulary).to(device)

# Initialize the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=parameters_helper.learning_rate,
    weight_decay=parameters_helper.weight_decay)

iterations = 0
best_dev_accuracy = -1

print('Starting training...')

statistics_helper = StatisticsHelper()
statistics_helper.print_header()

plot_helper = PlotHelper()
epoch = 0
current_learning_rate = parameters_helper.learning_rate

# Start the training
while current_learning_rate > minimum_learning_rate and epoch < parameters_helper.max_epochs:
    n_correct, n_total = 0, 0

    train_epoch_accuracies = []

    train_iterator, dev_iterator, test_iterator = data_storage.get_dataset_iterators(
        parameters_helper.batch_size, device)

    for batch_idx, batch in enumerate(train_iterator):

        # Switch the model to training mode
        model.train()

        # Make the forward pass
        train_predictions = model.forward(batch)

        # Calculate the accuracy of the predictions in the current batch
        n_correct += calculations_helper.calculate_correct_predictions(
            train_predictions, batch.label)

        n_total += batch.batch_size

        train_accuracy = calculations_helper.calculate_full_accuracy(
            train_predictions, batch.label)
        train_epoch_accuracies.append(train_accuracy)

        # calculate loss of the network output with respect to training labels
        train_loss = criterion(train_predictions, batch.label)
        plot_helper.add_train_result(train_accuracy, train_loss.item())

        # Clear gradient accumulators
        optimizer.zero_grad()

        # Make the backward pass
        train_loss.backward()

        # Update the parameters using the optimizer
        optimizer.step()

        # Cache model
        if iterations % parameters_helper.save_every_steps == 0:
            cache_storage.save_snapshot(
                model, iterations, train_accuracy, train_loss.item())

        # Output statistics periodically
        if iterations % parameters_helper.log_every_steps == 0:
            train_micro_accuracy = calculations_helper.calculate_accuracy(
                n_correct, n_total)
            # plot_helper.plot_grad_flow(model.named_parameters())
            statistics_helper.print_train_results(
                epoch,
                iterations,
                batch_idx,
                train_loss.item(),
                train_micro_accuracy,
                np.mean(train_epoch_accuracies),
                train_batches_size=len(train_iterator))

        iterations += 1

    train_micro_accuracy = calculations_helper.calculate_accuracy(
        n_correct, n_total)
    train_macro_accuracy = np.mean(train_epoch_accuracies)

    # Switch model to evaluation mode
    model.eval()

    # Calculate the accuracy on the dev set
    dev_correct_predictions, dev_loss = 0, 0
    dev_epoch_accuracies = []
    for dev_batch_idx, dev_batch in enumerate(dev_iterator):
        dev_predictions = model.forward(dev_batch)
        dev_correct_predictions += calculations_helper.calculate_correct_predictions(
            dev_predictions, dev_batch.label)

        dev_accuracy = calculations_helper.calculate_full_accuracy(
            dev_predictions, dev_batch.label)

        dev_epoch_accuracies.append(dev_accuracy)
        dev_loss = criterion(dev_predictions, dev_batch.label)

    dev_macro_accuracy = np.mean(dev_epoch_accuracies)
    dev_micro_accuracy = calculations_helper.calculate_accuracy(
        dev_correct_predictions, dev_split_size)

    statistics_helper.print_epoch_results(
        epoch,
        iterations,
        train_micro_accuracy,
        train_macro_accuracy,
        dev_loss.item(),
        dev_micro_accuracy,
        dev_macro_accuracy)

    plot_helper.add_dev_result(dev_accuracy, dev_loss.item())
    plot_helper.update_plot()

    # Update best dev set accuracy if we have
    # found a model with a better dev set accuracy
    if dev_accuracy > best_dev_accuracy:
        best_dev_accuracy = dev_accuracy
        cache_storage.save_best_snapshot(
            model, iterations, dev_accuracy, dev_loss.item())
    elif dev_accuracy < best_dev_accuracy:
        print('Reducing learning rate...')
        current_learning_rate /= 5.0
        if current_learning_rate < minimum_learning_rate:
            print(
                f'Learning rate dropped below the minimum one({minimum_learning_rate}). Stopping training...')
            break

        for g in optimizer.param_groups:
            g['lr'] = current_learning_rate

    epoch += 1

plot_helper.save_plot(
    f'results/{parameters_helper.encoding_model}/result-final.pickle')
