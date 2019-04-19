import argparse

DEFAULT_SAVE_EVERY_STEPS = 1000
DEFAULT_LOG_EVERY_STEPS = 50
DEFAULT_EVALUATE_EVERY_STEPS = 1000
DEFAULT_SNAPSHOT_LOCATION = None
DEFAULT_BATCH_SIZE = 64


class ParametersHelper():

    def __init__(self):
        self._save_every_steps = DEFAULT_SAVE_EVERY_STEPS
        self._log_every_steps = DEFAULT_LOG_EVERY_STEPS
        self._evaluate_every_steps = DEFAULT_EVALUATE_EVERY_STEPS
# '.\\results\\best_snapshot_devacc_68.67506604348709_devloss_1.143693447113037__iter_4000_model.pt'
        self._snapshot_location = DEFAULT_SNAPSHOT_LOCATION
        self._batch_size = DEFAULT_BATCH_SIZE

        self.FLAGS = None

    def load_arguments(self):
        # Command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_every_steps', type=int, default=DEFAULT_SAVE_EVERY_STEPS,
                            help='Number of steps after which the model will be saved')
        parser.add_argument('--log_every_steps', type=int, default=DEFAULT_LOG_EVERY_STEPS,
                            help='Number of steps after which the current results will be printed')
        parser.add_argument('--evaluate_every_steps', type=int, default=DEFAULT_EVALUATE_EVERY_STEPS,
                            help='Number of steps after which the current model will be evaluated')
        parser.add_argument('--snapshot_location', type=str, default=DEFAULT_SNAPSHOT_LOCATION,
                            help='Snapshot location from where to load the model')
        parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                            help='Batch size to use for the dataset')
        self.FLAGS, _ = parser.parse_known_args()

        self._save_every_steps = self.FLAGS.save_every_steps
        self._log_every_steps = self.FLAGS.log_every_steps
        self._evaluate_every_steps = self.FLAGS.evaluate_every_steps
        # '.\\results\\best_snapshot_devacc_68.67506604348709_devloss_1.143693447113037__iter_4000_model.pt'
        self._snapshot_location = self.FLAGS.snapshot_location
        self._batch_size = self.FLAGS.batch_size

    def print_arguments(self):
        for key, value in vars(self.FLAGS).items():
            print(key + ' : ' + str(value))

    @property
    def save_every_steps(self):
        return self._save_every_steps

    @property
    def log_every_steps(self):
        return self._log_every_steps

    @property
    def evaluate_every_steps(self):
        return self._evaluate_every_steps

    @property
    def snapshot_location(self):
        return self._snapshot_location

    @property
    def batch_size(self):
        return self._batch_size
