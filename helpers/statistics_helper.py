import time


class StatisticsHelper():
    def __init__(self):
        self.header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Train/Micro-Accuracy  Train/Macro-Accuracy  Dev/Micro-Accuracy  Dev/Macro-Accuracy'
        self.dev_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{} {} ,{},{:8.6f},{:24.4f},{:21.4f},{:19.4f},{:19.4f}'
            .split(','))
        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:24.4f},{:21.4f},{},{}'.split(','))

        self.start_time = time.time()

    def print_header(self):
        print(self.header)

    def print_train_results(
            self,
            epoch: int,
            iteration: int,
            current_batch_id: int,
            train_loss: float,
            train_micro_accuracy: float,
            train_macro_accuracy: float,
            train_batches_size: int):

        print(
            self.log_template.format(
                time.time()-self.start_time,
                epoch,
                iteration,
                1 + current_batch_id,
                train_batches_size,
                100. * (1+current_batch_id) / train_batches_size,
                train_loss,
                '-'*8,
                train_micro_accuracy,
                train_macro_accuracy,
                '-'*19,
                '-'*19))

    def print_epoch_results(
            self,
            epoch: int,
            iteration: int,
            train_micro_accuracy: float,
            train_macro_accuracy: float,
            dev_loss: float,
            dev_micro_accuracy: float,
            dev_macro_accuracy: float):

        print(
            self.dev_log_template.format(
                time.time()-self.start_time,
                epoch,
                iteration,
                '-'*11,
                '-'*9,
                '-'*6,
                dev_loss,
                train_micro_accuracy,
                train_macro_accuracy,
                dev_micro_accuracy,
                dev_macro_accuracy))
