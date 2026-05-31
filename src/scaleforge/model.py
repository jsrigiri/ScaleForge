import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """
    Small transformer classifier used to validate
    distributed training infrastructure.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.encoder(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits