"""Callback to log per-component reward breakdown and stage distribution to W&B/TensorBoard."""

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

# Stage labels for reward_v21 (and any future reward that sets info["reward_stage"])
STAGE_NAMES = {
    0: "reaching",
    1: "grasping",
    2: "transporting",
    3: "lowering",
    4: "placed",
}


class RewardComponentCallback(BaseCallback):
    """Logs reward components and stage distribution per episode.

    Reads from info dict keys populated by the reward function:
      - info["reward_components"]: dict[str, float] — per-step component values
      - info["reward_stage"]: int — discrete stage label (optional, v21+)

    Logs every `log_freq` timesteps to both W&B and TensorBoard:
      - reward_components/mean_<name>   : per-step mean within episodes
      - reward_components/ep_sum_<name> : total contribution per episode
      - reward_fraction/<name>          : fraction of total reward per component
      - stages/frac_<name>              : fraction of steps spent in each stage
      - stages/max_stage_reached        : highest stage reached on average
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_log = 0

        # Per-episode accumulators (flushed at episode end)
        self._step_components: list[dict[str, float]] = []
        self._step_stages: list[int] = []

        # Cross-episode buffer (flushed at log time)
        self._episode_data: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            components = info.get("reward_components")
            if components is not None:
                self._step_components.append(components)

            stage = info.get("reward_stage")
            if stage is not None:
                self._step_stages.append(int(stage))

            # Episode ended — store summary and reset step buffers
            episode_done = (
                info.get("terminal_observation") is not None
                or info.get("TimeLimit.truncated", False)
            )
            if episode_done and self._step_components:
                keys = self._step_components[0].keys()
                self._episode_data.append({
                    "ep_means": {k: float(np.mean([s[k] for s in self._step_components])) for k in keys},
                    "ep_sums":  {k: float(np.sum( [s[k] for s in self._step_components])) for k in keys},
                    "stages":   list(self._step_stages),
                })
                self._step_components = []
                self._step_stages = []

        if self.num_timesteps - self._last_log >= self.log_freq and self._episode_data:
            self._flush()
            self._last_log = self.num_timesteps

        return True

    def _flush(self):
        log_dict = {}

        # --- Reward components ---
        keys = self._episode_data[0]["ep_means"].keys()
        for k in keys:
            log_dict[f"reward_components/mean_{k}"] = float(np.mean(
                [ep["ep_means"][k] for ep in self._episode_data]
            ))
            log_dict[f"reward_components/ep_sum_{k}"] = float(np.mean(
                [ep["ep_sums"][k] for ep in self._episode_data]
            ))

        # Fraction of total reward from each component
        total = sum(log_dict[f"reward_components/ep_sum_{k}"] for k in keys)
        if abs(total) > 1e-8:
            for k in keys:
                log_dict[f"reward_fraction/{k}"] = log_dict[f"reward_components/ep_sum_{k}"] / total

        # --- Stage distribution ---
        all_stages = [s for ep in self._episode_data for s in ep["stages"]]
        if all_stages:
            total_steps = len(all_stages)
            for sid, name in STAGE_NAMES.items():
                log_dict[f"stages/frac_{name}"] = all_stages.count(sid) / total_steps

            # Average of the max stage reached per episode
            max_per_ep = [max(ep["stages"]) if ep["stages"] else 0 for ep in self._episode_data]
            log_dict["stages/max_stage_reached"] = float(np.mean(max_per_ep))

        # Log to W&B
        if wandb.run is not None:
            wandb.log(log_dict, step=self.num_timesteps)

        # Log to TensorBoard
        if self.logger is not None:
            for key, val in log_dict.items():
                self.logger.record(key, val)

        if self.verbose:
            n = len(self._episode_data)
            stage_summary = {STAGE_NAMES[s]: f"{all_stages.count(s)/len(all_stages)*100:.1f}%"
                             for s in STAGE_NAMES if s in all_stages} if all_stages else {}
            print(f"[RewardComponents] step={self.num_timesteps} episodes={n} stages={stage_summary}")

        self._episode_data = []
