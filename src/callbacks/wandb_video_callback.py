"""Callback to log rollout episode videos to W&B."""
import tempfile
from pathlib import Path

import imageio
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.envs.lift_cube import LiftCubeCartesianEnv
from src.envs.pick_cube import PickCubeEnv


class WandbVideoCallback(BaseCallback):
    """Runs one eval episode every `log_freq` steps, records frames, and logs to W&B."""

    def __init__(self, env_cfg: dict, vec_normalize: VecNormalize, log_freq: int = 50_000,
                 camera: str = "closeup", fps: int = 30, env_type: str = "lift",
                 verbose: int = 0):
        super().__init__(verbose)
        self.env_cfg = env_cfg
        self.vec_normalize = vec_normalize
        self.log_freq = log_freq
        self.camera = camera
        self.fps = fps
        self.env_type = env_type
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.log_freq:
            return True
        self._last_log = self.num_timesteps
        self._record_and_log()
        return True

    def _record_and_log(self):
        env = self._make_env()

        obs, _ = env.reset()
        frames = [self._render_frame(env)]
        done = False
        while not done:
            # Normalize obs the same way training env does
            obs_norm = self.vec_normalize.normalize_obs(obs)
            action, _ = self.model.predict(obs_norm, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(self._render_frame(env))
            done = terminated or truncated
        env.close()

        frames = [f.astype(np.uint8) for f in frames]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        imageio.mimwrite(tmp_path, frames, fps=self.fps, codec="libx264")
        wandb.log(
            {"rollout/video": wandb.Video(tmp_path, fps=self.fps, format="mp4")},
            step=self.num_timesteps,
        )
        Path(tmp_path).unlink(missing_ok=True)
        if self.verbose:
            print(f"[WandbVideo] Logged episode video at step {self.num_timesteps}")

    def _make_env(self):
        if self.env_type == "pick":
            return PickCubeEnv(
                render_mode="rgb_array",
                max_episode_steps=self.env_cfg.get("max_episode_steps", 400),
                action_scale=self.env_cfg.get("action_scale", 0.05),
                reward_config=self.env_cfg.get("reward_config", None),
            )

        place_target = self.env_cfg.get("place_target")
        if place_target is not None:
            place_target = tuple(place_target)

        return LiftCubeCartesianEnv(
            render_mode="rgb_array",
            max_episode_steps=self.env_cfg.get("max_episode_steps", 200),
            action_scale=self.env_cfg.get("action_scale", 0.02),
            lift_height=self.env_cfg.get("lift_height", 0.08),
            hold_steps=self.env_cfg.get("hold_steps", 10),
            reward_type=self.env_cfg.get("reward_type", "dense"),
            reward_version=self.env_cfg.get("reward_version", "v7"),
            curriculum_stage=self.env_cfg.get("curriculum_stage", 0),
            lock_wrist=self.env_cfg.get("lock_wrist", False),
            place_target=place_target,
        )

    def _render_frame(self, env):
        if self.env_type == "pick":
            return env.render()
        return env.render(camera=self.camera)
