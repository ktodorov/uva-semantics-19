import time


class StatisticsHelper():
    def __init__(self):
        self.header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
        self.dev_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'
            .split(','))
        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

        self.start_time = time.time()

    def print_header(self):
        print(self.header)

    def print_results(
            self,
            epoch: int,
            iteration: int,
            current_batch_id: int,
            train_loss: float,
            train_accuracy: float,
            train_batches_size: int,
            dev_loss: float = None,
            dev_accuracy: float = None):

        if dev_loss and dev_accuracy:
            print(
                self.dev_log_template.format(
                    time.time()-self.start_time,
                    epoch,
                    iteration,
                    1 + current_batch_id,
                    train_batches_size,
                    100. * (1+current_batch_id) / train_batches_size,
                    train_loss,
                    dev_loss,
                    train_accuracy,
                    dev_accuracy))
        else:
            print(
                self.log_template.format(
                    time.time()-self.start_time,
                    epoch,
                    iteration,
                    1 + current_batch_id,
                    train_batches_size,
                    100. * (1+current_batch_id) / train_batches_size,
                    train_loss,
                    ' '*8,
                    train_accuracy,
                    ' '*12))
