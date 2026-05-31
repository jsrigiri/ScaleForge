import torch

from scaleforge.checkpoint import CheckpointManager
from scaleforge.model import TinyTransformer


def test_checkpoint_save_and_load(tmp_path):
    model = TinyTransformer(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_classes=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    manager = CheckpointManager(
        str(tmp_path)
    )

    manager.save(
        model=model,
        optimizer=optimizer,
        step=5,
    )

    latest = tmp_path / "latest.pt"

    assert latest.exists()

    loaded_step = manager.load(
        path=str(latest),
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loaded_step == 5