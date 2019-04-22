import argparse

DEFAULT_SAVE_EVERY_STEPS = 1000
DEFAULT_LOG_EVERY_STEPS = 50
DEFAULT_MAX_SAMPLES = None
DEFAULT_SNAPSHOT_LOCATION = None
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_ENCODING_MODEL = 'mean'
DEFAULT_WEIGHT_DECAY = 0.01

class ParametersHelper():

    def __init__(self):
        self._save_every_steps = DEFAULT_SAVE_EVERY_STEPS
        self._log_every_steps = DEFAULT_LOG_EVERY_STEPS
        self._max_samples = DEFAULT_MAX_SAMPLES
        self._snapshot_location = DEFAULT_SNAPSHOT_LOCATION
        self._batch_size = DEFAULT_BATCH_SIZE
        self._max_epochs = DEFAULT_MAX_EPOCHS
        self._learning_rate = DEFAULT_LEARNING_RATE
        self._encoding_model = DEFAULT_ENCODING_MODEL
        self._weight_decay = DEFAULT_WEIGHT_DECAY

        self.FLAGS = None

    def load_arguments(self):
        # Command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_every_steps', type=int, default=DEFAULT_SAVE_EVERY_STEPS,
                            help='Number of steps after which the model will be saved')
        parser.add_argument('--log_every_steps', type=int, default=DEFAULT_LOG_EVERY_STEPS,
                            help='Number of steps after which the current results will be printed')
        parser.add_argument('--max_samples', type=int, default=DEFAULT_MAX_SAMPLES,
                            help='Number of samples to get from the datasets. If None, all samples will be used')
        parser.add_argument('--snapshot_location', type=str, default=DEFAULT_SNAPSHOT_LOCATION,
                            help='Snapshot location from where to load the model')
        parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                            help='Batch size to use for the dataset')
        parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS,
                            help='Amount of max epochs of the training')
        parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                            help='Learning rate')
        parser.add_argument('--encoding_model', type=str, default=DEFAULT_ENCODING_MODEL,
                            help="Model type for encoding sentences. Choose from 'mean', 'uni-lstm', 'bi-lstm' and 'bi-lstm-max-pool'")
        parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                            help="Weight decay for the optimizer")

        self.FLAGS, _ = parser.parse_known_args()

        self._save_every_steps = self.FLAGS.save_every_steps
        self._log_every_steps = self.FLAGS.log_every_steps
        self._max_samples = self.FLAGS.max_samples
        self._snapshot_location = self.FLAGS.snapshot_location
        self._batch_size = self.FLAGS.batch_size
        self._max_epochs = self.FLAGS.max_epochs
        self._learning_rate = self.FLAGS.learning_rate
        self._encoding_model = self.FLAGS.encoding_model
        self._weight_decay = self.FLAGS.weight_decay

    def print_arguments(self):
        print("Arguments:")
        for key, value in vars(self.FLAGS).items():
            print(key + ' : ' + str(value))
        
        print("-----------------------------------------")

    @property
    def save_every_steps(self):
        return self._save_every_steps

    @property
    def log_every_steps(self):
        return self._log_every_steps

    @property
    def max_samples(self):
        return self._max_samples

    @property
    def snapshot_location(self):
        return self._snapshot_location

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def encoding_model(self):
        return self._encoding_model

    @property
    def weight_decay(self):
        return self._weight_decay
