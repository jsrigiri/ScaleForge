from pathlib import Path

import torch


class CheckpointManager:

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        step: int,
    ) -> Path:
        checkpoint_path = (
            self.checkpoint_dir
            / f"checkpoint_step_{step}.pt"
        )

        checkpoint = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def save_state_dicts(
        self,
        model_state: dict,
        optimizer_state: dict,
        step: int,
    ) -> Path:
        checkpoint_path = (
            self.checkpoint_dir
            / f"checkpoint_step_{step}.pt"
        )

        checkpoint = {
            "step": step,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }

        torch.save(checkpoint, checkpoint_path)

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load(
        self,
        path: str,
        model,
        optimizer,
        device,
    ) -> int:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        step = checkpoint["step"]

        print(f"Loaded checkpoint from step {step}")

        return step