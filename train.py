import torch
import torch.optim as optim
import torch.nn as nn

# from encoders.mean_encoder import MeanEncoder
from encoders.encoding_helper import EncodingHelper

from helpers.calculations_helper import CalculationsHelper
from helpers.cache_storage import CacheStorage
from helpers.data_storage import DataStorage
from helpers.parameters_helper import ParametersHelper
from helpers.plot_helper import PlotHelper
from helpers.statistics_helper import StatisticsHelper

from snli_classifier import SNLIClassifier

calculations_helper = CalculationsHelper()

# Load input arguments
parameters_helper = ParametersHelper()
parameters_helper.load_arguments()
parameters_helper.print_arguments()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load the data sets and the vocabulary
print('Loading data...')
data_storage = DataStorage()

token_vocabulary, _ = data_storage.get_vocabulary()
train_iterator, dev_iterator, test_iterator = data_storage.get_dataset_iterators(
    parameters_helper.batch_size)
dev_split_size = data_storage.get_dev_split_size()

# Check if we can get a cached model. If not, create a new one
cache_storage = CacheStorage(parameters_helper.encoding_model)

print('Loading model...')
model = cache_storage.load_model_snapshot(parameters_helper.snapshot_location)
if not model:
    encoding_helper = EncodingHelper()
    encoder = encoding_helper.get_encoder(parameters_helper.encoding_model)
    model = SNLIClassifier(len(token_vocabulary), 300,
                           encoder, token_vocabulary)

# Initialize the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())  # learning rate

iterations = 0
best_dev_accuracy = -1

print('Starting training...')

statistics_helper = StatisticsHelper(train_batches_size=len(train_iterator))
statistics_helper.print_header()

plot_helper = PlotHelper()

# Start the training
for epoch in range(parameters_helper.max_epochs):
    train_iterator.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iterator):

        # Switch the model to training mode
        model.train()

        # Clear gradient accumulators
        optimizer.zero_grad()

        # Make the forward pass
        train_predictions = model(batch)

        # Calculate the accuracy of the predictions in the current batch
        n_correct += calculations_helper.calculate_correct_predictions(
            train_predictions, batch.label)

        n_total += batch.batch_size
        train_accuracy = calculations_helper.calculate_full_accuracy(
            train_predictions, batch.label)

        train_accuracy_normalized = calculations_helper.calculate_accuracy(
            n_correct, n_total)

        # calculate loss of the network output with respect to training labels
        train_loss = criterion(train_predictions, batch.label)
        plot_helper.add_train_result(train_accuracy, train_loss.item())

        # Make the backward pass
        train_loss.backward()

        # Update the parameters using the optimizer
        optimizer.step()

        # Cache model
        if iterations % parameters_helper.save_every_steps == 0:
            cache_storage.save_snapshot(
                model, iterations, train_accuracy_normalized, train_loss.item())

        # Evaluate performance on the dev set periodically
        if iterations % parameters_helper.evaluate_every_steps == 0:

            # Switch model to evaluation mode
            model.eval()
            dev_iterator.init_epoch()

            # Calculate the accuracy on the dev set
            dev_correct_predictions, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iterator):
                    dev_predictions = model(dev_batch)
                    dev_correct_predictions += calculations_helper.calculate_correct_predictions(
                        dev_predictions, dev_batch.label)

                    dev_loss = criterion(dev_predictions, dev_batch.label)

            dev_accuracy = calculations_helper.calculate_accuracy(
                dev_correct_predictions, dev_split_size)

            statistics_helper.print_results(
                epoch,
                iterations,
                batch_idx,
                train_loss.item(),
                train_accuracy_normalized,
                dev_loss.item(),
                dev_accuracy)

            plot_helper.add_dev_result(dev_accuracy, dev_loss.item())
            plot_helper.update_plot()

            # Update best dev set accuracy if we have
            # found a model with a better dev set accuracy
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                cache_storage.save_best_snapshot(
                    model, iterations, dev_accuracy, dev_loss.item())

        # Output statistics periodically
        elif iterations % parameters_helper.log_every_steps == 0:
            statistics_helper.print_results(
                epoch,
                iterations,
                batch_idx,
                train_loss.item(),
                train_accuracy_normalized)

        iterations += 1
