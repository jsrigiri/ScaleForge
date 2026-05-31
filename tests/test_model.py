import torch

from scaleforge.model import TinyTransformer


def test_forward_pass():

    model = TinyTransformer(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_classes=2,
    )

    x = torch.randint(
        0,
        128,
        (8, 32),
    )

    y = model(x)

    assert y.shape == (8, 2)