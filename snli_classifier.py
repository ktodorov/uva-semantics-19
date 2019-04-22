import torch
import torch.nn as nn

from encoders.base_encoder import BaseEncoder

class SNLIClassifier(nn.Module):

    def __init__(
            self,
            encoder: BaseEncoder,
            vocabulary=None,
            n_hidden: int = 512,
            n_classes: int = 3):

        super(SNLIClassifier, self).__init__()

        self.embed = nn.Embedding.from_pretrained(vocabulary.vectors)
        self.embed.requires_grad = False
        
        self.encoder = encoder

        self.out = nn.Sequential(
            nn.Linear(encoder.input_dimensions, n_hidden),
            nn.Linear(n_hidden, n_classes))

    def forward(self, batch):
        premise, premise_length = batch.premise
        hypothesis, hypothesis_length = batch.hypothesis

        premise_embedding = self.embed(premise)
        hypothesis_embedding = self.embed(hypothesis)

        premise_encoded = self.encoder.forward(
            premise_embedding, premise_length)
        hypothesis_encoded = self.encoder.forward(
            hypothesis_embedding, hypothesis_length)

        # (u, v, |u - v|, u*v)
        absolute_diff = (premise_encoded - hypothesis_encoded).abs()
        multiplication = (premise_encoded * hypothesis_encoded)
        input_embedding = torch.cat(
            (premise_encoded, hypothesis_encoded, absolute_diff, multiplication), 1)

        scores = self.out(input_embedding)

        return scores
