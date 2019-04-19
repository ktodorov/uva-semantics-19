import torch

class CalculationsHelper():
    def __init__(self):
        pass

    def calculate_correct_predictions(self, predictions, labels) -> int:
        correct_predictions = (torch.max(predictions, 1)[1].view(
            labels.size()) == labels).sum().item()

        return correct_predictions

    def calculate_accuracy(self, correct_predictions, total_size):
        accuracy = 100. * correct_predictions/total_size
        
        return accuracy

    def calculate_full_accuracy(self, predictions, labels, total_size) -> float:
        correct_predictions = self.calculate_correct_predictions(
            predictions, labels)

        accuracy = self.calculate_accuracy(correct_predictions, total_size)
        return accuracy

        
