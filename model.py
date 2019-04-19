import torch
import torch.nn as nn


class SNLIClassifier(nn.Module):

    def __init__(self, n_embed, d_out, encoder, vocabulary=None):
        super(SNLIClassifier, self).__init__()
        
        self.embed = nn.Embedding(n_embed, d_out)

        if vocabulary:
            self.embed.weight.data.copy_(vocabulary.vectors)

        self.encoder = encoder

        self.out = nn.Sequential(
            nn.Linear(4*300, 512),
            nn.Tanh(),
            nn.Linear(512, 3))

    def forward(self, batch):
        premise, premise_length = batch.premise
        hypothesis, hypothesis_length = batch.hypothesis

        premise_embedding = self.embed(premise)
        hypothesis_embedding = self.embed(hypothesis)

        premise_encoded = self.encoder.forward(
            premise_embedding, premise_length)
        hypothesis_encoded = self.encoder.forward(
            hypothesis_embedding, hypothesis_length)

        absolute_diff = (premise_encoded - hypothesis_encoded).abs()
        multiplication = (premise_encoded * hypothesis_encoded)
        input_embedding = torch.cat(
            (premise_encoded, hypothesis_encoded, absolute_diff, multiplication), 1)

        scores = self.out(input_embedding)
        return scores
