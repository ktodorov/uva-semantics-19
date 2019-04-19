import matplotlib.pyplot as plt
import numpy as np
import math


class PlotHelper():
    def __init__(self):
        self.train_steps_accuracy = []
        self.dev_steps_accuracy = []
        self.train_steps_loss = []
        self.dev_steps_loss = []

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

    def update_plot(self):
        self.train_steps_accuracy = np.arange(len(self.train_accuracies))
        self.dev_steps_accuracy = np.linspace(
            0, len(self.train_accuracies), len(self.dev_accuracies))

        self.train_steps_loss = self.train_steps_accuracy.copy()
        self.dev_steps_loss = self.dev_steps_accuracy.copy()

        self.refresh_plot()

    def refresh_plot(self):
        plt.figure(1)
        plt.subplot(121)
        plt.plot(self.train_steps_accuracy,
                 self.train_accuracies, color='skyblue', label='train accuracy')
        plt.plot(self.dev_steps_accuracy, self.dev_accuracies,
                 color='tomato', label='dev accuracy')

        plt.subplot(122)
        plt.plot(self.train_steps_loss,
                 self.train_losses, color='skyblue', label='train loss')
        plt.plot(self.dev_steps_loss, self.dev_losses,
                 color='tomato', label='dev loss')
        plt.draw()
        plt.pause(0.001)
        
    def save_plot(self, path):
        plt.savefig(path)

    def _initialize_plot(self):
        plt.figure(1)
        plt.subplot(121)
        plt.title('accuracies')
        plt.xlabel('steps')
        plt.ylabel('accuracy')
        plt.plot([],
                 [], color='skyblue', label='train accuracy')
        plt.plot([], [], color='tomato', label='dev accuracy')
        plt.legend()

        plt.subplot(122)
        plt.title('losses')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.plot([],
                 [], color='skyblue', label='train loss')
        plt.plot([], [], color='tomato', label='dev loss')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

        plt.ion()
        plt.show()
