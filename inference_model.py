class InferenceModel():
    def __init__(
            self,
            premise,
            premise_length,
            hypothesis,
            hypothesis_length):

        self.premise = (premise, premise_length)
        self.hypothesis = (hypothesis, hypothesis_length)
