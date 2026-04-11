"""Upload training checkpoints to Weights & Biases."""

from pathlib import Path

import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCheckpointCallback(BaseCallback):
    """Uploads newly created checkpoint files to the active W&B run."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        check_freq: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.check_freq = check_freq
        self._last_check = 0
        self._uploaded: set[str] = set()

    def _upload_file(self, path: Path) -> None:
        if wandb.run is None or not path.exists():
            return
        wandb.save(str(path), base_path=str(self.output_dir), policy="now")
        self._uploaded.add(path.name)
        if self.verbose:
            print(f"[WandbCheckpoint] uploaded {path.name}")

    def _upload_new_checkpoints(self) -> None:
        if wandb.run is None or not self.checkpoint_dir.exists():
            return

        for path in sorted(self.checkpoint_dir.glob("*.zip")):
            if path.name not in self._uploaded:
                self._upload_file(path)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check >= self.check_freq:
            self._upload_new_checkpoints()
            self._last_check = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._upload_new_checkpoints()
        for rel_path in ("final_model.zip", "vec_normalize.pkl"):
            self._upload_file(self.output_dir / rel_path)
