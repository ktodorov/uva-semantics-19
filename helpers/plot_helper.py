import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import math
import pickle


class PlotHelper():
    def __init__(self):
        self.train_steps = []
        self.dev_steps = []
        self.progress_figure = None

        self.train_accuracies = []
        self.train_losses = []

        self.dev_accuracies = []
        self.dev_losses = []

        self._initialize_plot()

    def add_train_result(self, train_accuracy, train_loss):
        self.train_accuracies.append(train_accuracy)
        self.train_losses.append(train_loss)

    def add_dev_result(self, dev_accuracy, dev_loss):
        self.dev_accuracies.append(dev_accuracy)
        self.dev_losses.append(dev_loss)

    def get_dev_accuracies(self):
        return self.dev_accuracies

    def update_plot(self):
        self.train_steps = np.linspace(
            0, len(self.dev_accuracies), len(self.train_accuracies))
        self.dev_steps = np.linspace(
            0, len(self.dev_accuracies), len(self.dev_accuracies))

        self.refresh_plot()

    def refresh_plot(self):
        plt.clf()
        plt.subplot(121)
        plt.plot(self.train_steps,
                 self.train_accuracies, color='skyblue', label='train accuracy')
        plt.plot(self.dev_steps, self.dev_accuracies,
                 color='tomato', label='dev accuracy')

        plt.subplot(122)
        plt.plot(self.train_steps,
                 self.train_losses, color='skyblue', label='train loss')
        plt.plot(self.dev_steps, self.dev_losses,
                 color='tomato', label='dev loss')
        plt.draw()
        plt.pause(0.001)

    def save_plot(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.progress_figure, file)

    def _initialize_plot(self):
        self.progress_figure = plt.figure(1, figsize=(12, 6), facecolor='w', edgecolor='k')
        plt.subplot(121)
        plt.title('accuracies')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.plot([],
                 [], color='skyblue', label='train accuracy')
        plt.plot([], [], color='tomato', label='dev accuracy')
        plt.legend()

        plt.subplot(122)
        plt.title('losses')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot([],
                 [], color='skyblue', label='train loss')
        plt.plot([], [], color='tomato', label='dev loss')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

        plt.ion()
        self.progress_figure.show()
        plt.show()
