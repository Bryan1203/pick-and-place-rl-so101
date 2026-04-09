"""Train a lift policy using MBPO (Model-Based Policy Optimization).

MBPO learns an ensemble of probabilistic dynamics models to generate synthetic
rollouts that augment real experience for SAC training, improving sample
efficiency over pure model-free SAC.

Key idea:
  1. Collect real transitions with SAC policy.
  2. Train an ensemble of probabilistic (obs, action) -> (Δobs, reward) models.
  3. Roll out the current policy through the model to generate synthetic data.
  4. Train SAC on a mix of real + synthetic data.

Reference: Janner et al. (2019) "When to Trust Your Model: Model-Based
Policy Optimization" https://arxiv.org/abs/1906.08253

1: Initialize policy π(a|s), predictive model pθ (s', r|s, a), empty dataset D.
2: for N epochs do
    Collect data with π in real environment: D = D + {(si, ai, s'i, ri)}i
    Train model pθ on dataset D via maximum likelihood: θ ← argmaxθ ED [log pθ (s', r|s, a)]
    Optimize policy under predictive model: π ← argmax π′ ˆη[π′] : C(eps_m, eps_π )
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.callbacks.wandb_video_callback import WandbVideoCallback
from src.envs.lift_cube import LiftCubeCartesianEnv


# ---------------------------------------------------------------------------
# Dynamics model (probabilistic ensemble)
# ---------------------------------------------------------------------------

class SingleDynamicsNet(nn.Module):
    """Single member of the dynamics ensemble.
    Maps (obs, action) -> (mean, log_var) for a Gaussian over (Δobs, reward).
    The output dim is obs_dim + 1 (delta obs + scalar reward).
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 200,
        n_layers: int = 4,
        max_log_var: float = 0.5,
        min_log_var: float = -10.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(obs_dim + act_dim, hidden_size), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)

        out_dim = obs_dim + 1  # obs + reward
        self.mean_head = nn.Linear(hidden_size, out_dim)
        self.log_var_head = nn.Linear(hidden_size, out_dim)
        self.max_log_var = max_log_var
        self.min_log_var = min_log_var

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(torch.cat([obs, act], dim=-1))
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        # Soft-clamp via double-softplus (Chua et al., 2018)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var

class EnsembleDynamicsModel:
    """Ensemble of probabilistic dynamics networks.

    Trains E independent networks; uses disagreement between members as an
    implicit uncertainty measure.  At prediction time, one member is chosen
    uniformly at random per transition (Thompson sampling style).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int = 5,
        hidden_size: int = 200,
        n_layers: int = 4,
        lr: float = 1e-3,
        max_log_var: float = 0.5,
        min_log_var: float = -10.0,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_size = ensemble_size
        self.device = device

        self.nets = nn.ModuleList([
            SingleDynamicsNet(obs_dim, act_dim, hidden_size, n_layers, max_log_var, min_log_var)
            for _ in range(ensemble_size)
        ])
        self.nets.to(device)
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=lr)

        # Running input normalizer (fitted on real data)
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_std = torch.ones(obs_dim, device=device)
        self.act_mean = torch.zeros(act_dim, device=device)
        self.act_std = torch.ones(act_dim, device=device)

    # ------------------------------------------------------------------
    def fit_normalizer(self, obs: np.ndarray, act: np.ndarray) -> None:
        """Update running input statistics from a batch of real data."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(act).to(self.device)
        self.obs_mean = obs_t.mean(0)
        self.obs_std = obs_t.std(0).clamp(min=1e-8)
        self.act_mean = act_t.mean(0)
        self.act_std = act_t.std(0).clamp(min=1e-8)

    def _normalize(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_n = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        act_n = (act - self.act_mean) / (self.act_std + 1e-8)
        return obs_n, act_n

    # ------------------------------------------------------------------
    def train_step(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        next_obs: torch.Tensor,
        rew: torch.Tensor,
    ) -> float:
        """One gradient step over all ensemble members. Returns mean NLL."""
        obs_n, act_n = self._normalize(obs, act)
        delta = next_obs - obs  # predict residuals (more stable)
        target = torch.cat([delta, rew.unsqueeze(-1)], dim=-1)  # (B, obs+1)

        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)
        for net in self.nets:
            mean, log_var = net(obs_n, act_n)
            inv_var = torch.exp(-log_var)
            nll = (inv_var * (target - mean).pow(2) + log_var).sum(-1).mean()
            total_loss = total_loss + nll
        total_loss.backward()
        self.optimizer.step()
        return (total_loss / self.ensemble_size).item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, obs: np.ndarray, act: np.ndarray, sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next_obs and reward for a batch of (obs, action) pairs.

        Randomly selects one ensemble member per sample (Thompson sampling).
        Returns (next_obs, reward) as float32 numpy arrays.
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(act).to(self.device)
        obs_n, act_n = self._normalize(obs_t, act_t)

        B = obs_t.shape[0]
        all_means, all_log_vars = [], []
        for net in self.nets:
            m, lv = net(obs_n, act_n)
            all_means.append(m)
            all_log_vars.append(lv)

        all_means = torch.stack(all_means)     # (E, B, out)
        all_log_vars = torch.stack(all_log_vars)  # (E, B, out)

        # Randomly pick one member per sample
        idx = torch.randint(self.ensemble_size, (B,), device=self.device)
        arange = torch.arange(B, device=self.device)
        mean_sel = all_means[idx, arange]       # (B, out)
        lv_sel = all_log_vars[idx, arange]      # (B, out)

        if sample:
            pred = mean_sel + torch.exp(0.5 * lv_sel) * torch.randn_like(mean_sel)
        else:
            pred = mean_sel

        delta_obs = pred[:, :-1]
        reward = pred[:, -1]
        next_obs_t = obs_t + delta_obs
        return next_obs_t.cpu().numpy(), reward.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.nets.state_dict(), path)

    def load(self, path: str) -> None:
        self.nets.load_state_dict(torch.load(path, map_location=self.device))


# ---------------------------------------------------------------------------
# Combined replay buffer (real + synthetic)
# ---------------------------------------------------------------------------

class CombinedReplayBuffer:
    """Samples from a real and a synthetic replay buffer with a fixed real_ratio.

    IMPORTANT storage convention
    ----------------------------
    • real_buffer stores *unnormalized* observations (SB3 default with VecNormalize).
      Sampling it WITH env= normalizes on the fly.
    • model_buffer stores *already-normalized* observations (generated by the policy
      in the normalized observation space).
      Sampling it WITHOUT env= returns normalized data directly.

    Both code paths therefore yield normalized tensors for the SAC update.
    """

    def __init__(
        self,
        real_buffer: ReplayBuffer,
        model_buffer: ReplayBuffer,
        real_ratio: float = 0.05,
    ):
        self.real_buffer = real_buffer
        self.model_buffer = model_buffer
        self.real_ratio = real_ratio

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        n_real = max(1, int(batch_size * self.real_ratio))
        n_model = batch_size - n_real

        # Real: pass env so VecNormalize normalizes stored unnorm observations
        real = self.real_buffer.sample(n_real, env=env)
        # Model: already in normalized space, do NOT pass env
        synth = self.model_buffer.sample(n_model, env=None)

        return ReplayBufferSamples(
            observations=torch.cat([real.observations, synth.observations]),
            actions=torch.cat([real.actions, synth.actions]),
            next_observations=torch.cat([real.next_observations, synth.next_observations]),
            dones=torch.cat([real.dones, synth.dones]),
            rewards=torch.cat([real.rewards, synth.rewards]),
        )

    # Proxy attributes that SB3's train() may inspect
    def size(self) -> int:
        return self.model_buffer.size()

    @property
    def full(self) -> bool:
        return self.model_buffer.full

    @property
    def pos(self) -> int:
        return self.model_buffer.pos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    place_target = env_cfg.get("place_target")
    if place_target is not None:
        place_target = tuple(place_target)
    return LiftCubeCartesianEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
        reward_version=env_cfg.get("reward_version", "v7"),
        curriculum_stage=env_cfg.get("curriculum_stage", 0),
        lock_wrist=env_cfg.get("lock_wrist", False),
        place_target=place_target,
    )


def get_rollout_length(step: int, schedule: list) -> int:
    """Linearly interpolate rollout length from a [(start, end, len_start, len_end)] schedule."""
    for start, end, l_start, l_end in schedule:
        if start <= step < end:
            frac = (step - start) / max(end - start, 1)
            return int(l_start + frac * (l_end - l_start))
    return schedule[-1][3]  # Beyond schedule: use final length


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MBPO training for lift task")
    parser.add_argument("--config", type=str, default="configs/lift_MB.yaml")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]
    mbpo_cfg = config["mbpo"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_name or f"{exp_cfg['name']}_{timestamp}"
        wandb.init(
            project=exp_cfg.get("wandb_project", "pick-101"),
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    # Environments
    env = DummyVecEnv([lambda: make_env(env_cfg)])
    env = VecNormalize(
        env,
        norm_obs=env_cfg.get("normalize_obs", True),
        norm_reward=env_cfg.get("normalize_reward", True),
    )
    eval_env = DummyVecEnv([lambda: make_env(env_cfg)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=env_cfg.get("normalize_obs", True),
        norm_reward=False,
        training=False,
    )

    # SAC model — we drive the training loop manually, so gradient_steps and
    # train_freq are set to 1 here and overridden in our loop below.
    sac = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        learning_starts=sac_cfg["learning_starts"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        train_freq=1,
        gradient_steps=1,   # actual steps controlled manually
        verbose=1,
        seed=train_cfg["seed"],
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    obs_dim: int = env.observation_space.shape[0]
    act_dim: int = env.action_space.shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # Dynamics model (ensemble)
    world_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        ensemble_size=mbpo_cfg["ensemble_size"],
        hidden_size=mbpo_cfg["model_hidden_size"],
        n_layers=mbpo_cfg["model_hidden_layers"],
        lr=mbpo_cfg["model_lr"],
        max_log_var=mbpo_cfg.get("max_log_var", 0.5),
        min_log_var=mbpo_cfg.get("min_log_var", -10.0),
        device=device,
    )

    # Separate replay buffer for synthetic (model-generated) transitions.
    # Stores *normalized* observations — see CombinedReplayBuffer docstring.
    model_buffer = ReplayBuffer(
        buffer_size=mbpo_cfg["model_buffer_size"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=1,
    )

    combined_buffer = CombinedReplayBuffer(
        real_buffer=sac.replay_buffer,
        model_buffer=model_buffer,
        real_ratio=mbpo_cfg["real_ratio"],
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="mbpo_lift",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )
    plot_cb = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
    )
    callback_list: List = [checkpoint_cb, eval_cb, plot_cb]
    if use_wandb:
        callback_list.append(WandbCallback(
            gradient_save_freq=train_cfg.get("save_freq", 50_000),
            verbose=2,
        ))
        callback_list.append(WandbVideoCallback(
            env_cfg=env_cfg,
            vec_normalize=env,
            log_freq=train_cfg.get("video_log_freq", 50_000),
            camera=train_cfg.get("video_camera", "closeup"),
            verbose=1,
        ))
    callback = CallbackList(callback_list)

    # ------------------------------------------------------------------
    # Training loop parameters
    # ------------------------------------------------------------------
    total_timesteps: int = train_cfg["timesteps"]
    learning_starts: int = sac_cfg["learning_starts"]
    batch_size: int = sac_cfg["batch_size"]
    gradient_steps: int = mbpo_cfg.get("gradient_steps", 20)
    model_train_freq: int = mbpo_cfg["model_train_freq"]
    model_train_steps: int = mbpo_cfg["model_train_steps"]
    model_batch_size: int = mbpo_cfg.get("model_batch_size", 256)
    rollouts_per_step: int = mbpo_cfg["rollouts_per_step"]
    rollout_schedule: list = mbpo_cfg["rollout_schedule"]

    # Initialize SB3 internal state (sets _last_obs, logger, etc.)
    total_timesteps, callback = sac._setup_learn(
        total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="mbpo",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())

    print(f"\nStarting MBPO Lift training for {total_timesteps} timesteps …")
    print(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for step in range(total_timesteps):
        sac.num_timesteps += 1

        # ── 1. Collect one real environment step ─────────────────────────
        action, buffer_action = sac._sample_action(
            learning_starts, action_noise=sac.action_noise, n_envs=1
        )
        new_obs, reward, dones, infos = env.step(action)

        # _store_transition adds to sac.replay_buffer and updates _last_obs
        sac._store_transition(sac.replay_buffer, buffer_action, new_obs, reward, dones, infos)

        callback.update_locals(locals())
        if not callback.on_step():
            break

        # ── 2. Train dynamics model (periodically) ────────────────────────
        if step >= learning_starts and step % model_train_freq == 0:
            real_size = sac.replay_buffer.size()
            if real_size >= model_batch_size:

                # Fit input normalizer on a fresh sample of real data
                if mbpo_cfg.get("normalize_model_inputs", True):
                    n_norm = min(real_size, 10_000)
                    norm_batch = sac.replay_buffer.sample(n_norm, env=env)
                    world_model.fit_normalizer(
                        norm_batch.observations.cpu().numpy(),
                        norm_batch.actions.cpu().numpy(),
                    )

                # Gradient steps on the dynamics model
                losses = []
                for _ in range(model_train_steps):
                    mb = sac.replay_buffer.sample(model_batch_size, env=env)
                    loss = world_model.train_step(
                        mb.observations,
                        mb.actions,
                        mb.next_observations,
                        mb.rewards.squeeze(-1),
                    )
                    losses.append(loss)
                mean_loss = float(np.mean(losses))

                if use_wandb:
                    wandb.log({"mbpo/dynamics_loss": mean_loss, "global_step": step})

                # ── 3. Generate synthetic rollouts ────────────────────────
                rollout_len = get_rollout_length(step, rollout_schedule)

                # Sample starting states from real buffer (normalized, via env)
                start_batch = sac.replay_buffer.sample(rollouts_per_step, env=env)
                rollout_obs = start_batch.observations.cpu().numpy()  # (N, obs)

                for _ in range(rollout_len):
                    # Policy actions in normalized obs space
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(rollout_obs).to(device)
                        rollout_actions = sac.actor(obs_t, deterministic=False).cpu().numpy()
                        rollout_actions = np.clip(rollout_actions, -1.0, 1.0)

                    # Dynamics model prediction (obs and rewards in normalized space)
                    next_rollout_obs, rollout_rewards = world_model.predict(
                        rollout_obs, rollout_actions, sample=True
                    )
                    next_rollout_obs = np.clip(next_rollout_obs, -10.0, 10.0)
                    rollout_rewards = np.clip(rollout_rewards, -10.0, 10.0)

                    # Add normalized transitions to model buffer one at a time
                    N = rollout_obs.shape[0]
                    for i in range(N):
                        model_buffer.add(
                            obs=rollout_obs[i : i + 1],
                            next_obs=next_rollout_obs[i : i + 1],
                            action=rollout_actions[i : i + 1],
                            reward=rollout_rewards[i : i + 1],
                            done=np.array([False]),
                            infos=[{}],
                        )

                    rollout_obs = next_rollout_obs

                if use_wandb:
                    wandb.log({
                        "mbpo/rollout_length": rollout_len,
                        "mbpo/model_buffer_size": model_buffer.size(),
                        "global_step": step,
                    })

        # ── 4. SAC policy update on mixed real + synthetic data ──────────
        if (
            step >= learning_starts
            and sac.replay_buffer.size() >= batch_size
            and model_buffer.size() >= batch_size
        ):
            # Temporarily swap in combined buffer so sac.train() samples from both
            original_buffer = sac.replay_buffer
            sac.replay_buffer = combined_buffer
            sac.train(gradient_steps=gradient_steps, batch_size=batch_size)
            sac.replay_buffer = original_buffer

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    callback.on_training_end()
    sac.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")
    world_model.save(str(output_dir / "dynamics_model.pt"))

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
