"""Sim-to-Real World Model Training + MBPO for the lift task.

Pipeline
--------
  Phase 1 │ Collect a large batch of trajectories from MuJoCo (online, random policy).
  Phase 2 │ Load a small set of real-world trajectories from a data file (.npz / .h5).
  Phase 3 │ Fit VecNormalize statistics on the combined sim + real observation corpus.
  Phase 4 │ Pre-train the ensemble dynamics model on sim data until convergence
           │   (early stopping on held-out validation loss).
  Phase 5 │ Fine-tune the dynamics model on real data with a reduced learning rate
           │   (early stopping again).
  Phase 6 │ RL training: online SAC over MuJoCo sim, augmented by world-model rollouts.
           │   SAC batch = real_data (5%) + sim_data (20%) + model_rollouts (75%).

The world model is kept frozen after fine-tuning — it acts as a bridge that is
initially calibrated on cheap sim data and then corrected toward real-world dynamics.
"""
import argparse
import copy
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


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics model (probabilistic ensemble)
# ─────────────────────────────────────────────────────────────────────────────

class SingleDynamicsNet(nn.Module):
    """One member of the dynamics ensemble.

    (obs, action) → (mean, log_var) of a Gaussian over (Δobs ∥ reward).
    Output dimension = obs_dim + 1.
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
        out_dim = obs_dim + 1
        self.mean_head    = nn.Linear(hidden_size, out_dim)
        self.log_var_head = nn.Linear(hidden_size, out_dim)
        self.max_log_var  = max_log_var
        self.min_log_var  = min_log_var

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(torch.cat([obs, act], dim=-1))
        mean    = self.mean_head(h)
        log_var = self.log_var_head(h)
        # Double-softplus clamp (Chua et al., 2018)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var


class EnsembleDynamicsModel:
    """E independently trained probabilistic dynamics networks.

    Uncertainty is represented implicitly through inter-member disagreement.
    At roll-out time, one member is selected uniformly at random per sample
    (Thompson sampling), preventing the policy from exploiting a single bias.
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
        self.obs_dim        = obs_dim
        self.act_dim        = act_dim
        self.ensemble_size  = ensemble_size
        self.device         = device

        self.nets = nn.ModuleList([
            SingleDynamicsNet(obs_dim, act_dim, hidden_size, n_layers, max_log_var, min_log_var)
            for _ in range(ensemble_size)
        ])
        self.nets.to(device)
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=lr)
        self.base_lr = lr

        # Input normalizer — synced from VecNormalize before training
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_std  = torch.ones(obs_dim,  device=device)
        # Actions are already in [-1, 1]; no normalization needed
        self.act_mean = torch.zeros(act_dim, device=device)
        self.act_std  = torch.ones(act_dim,  device=device)

    # ------------------------------------------------------------------
    def sync_normalizer_from_vecenv(self, vec_env: VecNormalize) -> None:
        """Copy obs running-stats from VecNormalize so the model input
        normalizer stays consistent with the SAC policy's observation space."""
        mean = vec_env.obs_rms.mean.astype(np.float32)
        std  = np.sqrt(vec_env.obs_rms.var.astype(np.float32) + vec_env.epsilon)
        self.obs_mean = torch.FloatTensor(mean).to(self.device)
        self.obs_std  = torch.FloatTensor(std).to(self.device)

    def _normalize(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_n = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        act_n = (act - self.act_mean) / (self.act_std + 1e-8)
        return obs_n, act_n

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ------------------------------------------------------------------
    def train_step(
        self,
        obs:      torch.Tensor,   # (B, obs_dim)  normalized
        act:      torch.Tensor,   # (B, act_dim)
        next_obs: torch.Tensor,   # (B, obs_dim)  normalized
        rew:      torch.Tensor,   # (B,)
    ) -> float:
        """One gradient step on all members. Returns mean NLL loss."""
        obs_n, act_n = self._normalize(obs, act)
        target = torch.cat([next_obs - obs, rew.unsqueeze(-1)], dim=-1)

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

    @torch.no_grad()
    def eval_loss(
        self,
        obs:      torch.Tensor,
        act:      torch.Tensor,
        next_obs: torch.Tensor,
        rew:      torch.Tensor,
        batch_size: int = 512,
    ) -> float:
        """NLL on a fixed validation set (no gradient update)."""
        n = obs.shape[0]
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            obs_b      = obs[i : i + batch_size]
            act_b      = act[i : i + batch_size]
            next_obs_b = next_obs[i : i + batch_size]
            rew_b      = rew[i : i + batch_size]
            obs_n, act_n = self._normalize(obs_b, act_b)
            target = torch.cat([next_obs_b - obs_b, rew_b.unsqueeze(-1)], dim=-1)
            for net in self.nets:
                mean, log_var = net(obs_n, act_n)
                inv_var = torch.exp(-log_var)
                nll = (inv_var * (target - mean).pow(2) + log_var).sum(-1).mean()
                total += nll.item()
            count += 1
        return total / max(count * self.ensemble_size, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, obs: np.ndarray, act: np.ndarray, sample: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predict (next_obs, reward). Thompson-samples one member per row."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(act).to(self.device)
        obs_n, act_n = self._normalize(obs_t, act_t)

        B = obs_t.shape[0]
        all_means, all_lvars = [], []
        for net in self.nets:
            m, lv = net(obs_n, act_n)
            all_means.append(m)
            all_lvars.append(lv)

        all_means = torch.stack(all_means)   # (E, B, out)
        all_lvars = torch.stack(all_lvars)   # (E, B, out)
        idx    = torch.randint(self.ensemble_size, (B,), device=self.device)
        arange = torch.arange(B, device=self.device)
        mean_s = all_means[idx, arange]
        lv_s   = all_lvars[idx, arange]

        pred = mean_s + (torch.exp(0.5 * lv_s) * torch.randn_like(mean_s) if sample else 0)
        next_obs_t = obs_t + pred[:, :-1]
        return next_obs_t.cpu().numpy(), pred[:, -1].cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.nets.state_dict(), path)

    def load(self, path: str) -> None:
        self.nets.load_state_dict(torch.load(path, map_location=self.device))


# ─────────────────────────────────────────────────────────────────────────────
# Three-way combined replay buffer  (real + sim + model)
# ─────────────────────────────────────────────────────────────────────────────

class ThreeWayCombinedBuffer:
    """Samples from three replay buffers with fixed proportions.

    Storage convention
    ------------------
    real_buffer  : raw (unnormalized) obs  →  sample WITH  env= to normalize
    sim_buffer   : raw (unnormalized) obs  →  sample WITH  env= to normalize
    model_buffer : normalized obs          →  sample WITHOUT env= (already normalized)
    """

    def __init__(
        self,
        real_buffer:  ReplayBuffer,
        sim_buffer:   ReplayBuffer,
        model_buffer: ReplayBuffer,
        real_ratio:   float = 0.05,
        sim_ratio:    float = 0.20,
    ):
        self.real_buffer  = real_buffer
        self.sim_buffer   = sim_buffer
        self.model_buffer = model_buffer
        self.real_ratio   = real_ratio
        self.sim_ratio    = sim_ratio

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        n_real  = max(1, int(batch_size * self.real_ratio))
        n_sim   = max(1, int(batch_size * self.sim_ratio))
        n_model = batch_size - n_real - n_sim

        real  = self.real_buffer.sample(n_real,  env=env)
        sim   = self.sim_buffer.sample(n_sim,    env=env)
        model = self.model_buffer.sample(n_model, env=None)   # already normalized

        return ReplayBufferSamples(
            observations      = torch.cat([real.observations,      sim.observations,      model.observations]),
            actions           = torch.cat([real.actions,           sim.actions,           model.actions]),
            next_observations = torch.cat([real.next_observations, sim.next_observations, model.next_observations]),
            dones             = torch.cat([real.dones,             sim.dones,             model.dones]),
            rewards           = torch.cat([real.rewards,           sim.rewards,           model.rewards]),
        )

    # Proxy attributes SB3's train() may inspect
    def size(self) -> int:      return self.model_buffer.size()
    @property
    def full(self) -> bool:     return self.model_buffer.full
    @property
    def pos(self) -> int:       return self.model_buffer.pos


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

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
    for start, end, l0, l1 in schedule:
        if start <= step < end:
            return int(l0 + (step - start) / max(end - start, 1) * (l1 - l0))
    return schedule[-1][3]


def load_real_data(path: str, buffer: ReplayBuffer, max_transitions: Optional[int] = None) -> int:
    """Load offline real-world trajectories into a ReplayBuffer.

    Supported formats
    -----------------
    .npz   : np.load with keys observations / next_observations / actions / rewards / dones
    .h5    : HDF5 with the same keys

    Stores raw (unnormalized) observations so the buffer is consistent with
    the SB3 convention (VecNormalize normalizes on the fly during sampling).

    Returns the number of transitions loaded.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npz":
        data     = np.load(p, allow_pickle=True)
        obs      = data["observations"].astype(np.float32)
        next_obs = data["next_observations"].astype(np.float32)
        actions  = data["actions"].astype(np.float32)
        rewards  = data["rewards"].astype(np.float32).reshape(-1)
        dones    = data["dones"].astype(bool).reshape(-1)
    elif suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            obs      = f["observations"][:].astype(np.float32)
            next_obs = f["next_observations"][:].astype(np.float32)
            actions  = f["actions"][:].astype(np.float32)
            rewards  = f["rewards"][:].astype(np.float32).reshape(-1)
            dones    = f["dones"][:].astype(bool).reshape(-1)
    else:
        raise ValueError(f"Unsupported real-data format: {suffix!r}. Use .npz or .h5/.hdf5")

    n = len(obs)
    if max_transitions is not None and n > max_transitions:
        idx      = np.random.choice(n, max_transitions, replace=False)
        obs      = obs[idx];      next_obs = next_obs[idx]
        actions  = actions[idx];  rewards  = rewards[idx];  dones = dones[idx]
        n = max_transitions

    print(f"Loading {n} real-world transitions from {p}")
    for i in range(n):
        buffer.add(
            obs=obs[i : i + 1],
            next_obs=next_obs[i : i + 1],
            action=actions[i : i + 1],
            reward=rewards[i : i + 1],
            done=dones[i : i + 1],
            infos=[{}],
        )
    return n


def collect_sim_data(
    sim_env:           VecNormalize,
    sim_buffer:        ReplayBuffer,
    n_steps:           int,
    use_random_policy: bool = True,
    sac:               Optional[SAC] = None,
) -> None:
    """Collect transitions from MuJoCo and store (unnormalized) in sim_buffer.

    Uses the original-obs interface so the buffer mirrors SB3's convention.
    """
    print(f"\nCollecting {n_steps} sim transitions (random={use_random_policy}) …")
    obs = sim_env.reset()
    last_original_obs = sim_env.get_original_obs().copy()

    for step in range(n_steps):
        if use_random_policy or sac is None:
            action = np.array([sim_env.action_space.sample()])
        else:
            action, _ = sac.predict(obs, deterministic=False)

        next_obs, _reward, done, info = sim_env.step(action)

        original_next_obs  = sim_env.get_original_obs().copy()
        original_reward    = sim_env.get_original_reward().copy()

        # Preserve terminal observations across episode resets
        real_next_obs = original_next_obs.copy()
        for i, d in enumerate(done):
            if d and info[i].get("terminal_observation") is not None:
                real_next_obs[i] = info[i]["terminal_observation"]

        sim_buffer.add(
            obs=last_original_obs,
            next_obs=real_next_obs,
            action=action,
            reward=original_reward,
            done=done,
            infos=info,
        )

        last_original_obs = original_next_obs.copy()
        if any(done):
            obs = sim_env.reset()
            last_original_obs = sim_env.get_original_obs().copy()
        else:
            obs = next_obs

        if (step + 1) % max(1, n_steps // 5) == 0:
            print(f"  Collected {step + 1}/{n_steps} sim steps")


def initialize_vecnormalize_from_data(
    vec_env:     VecNormalize,
    sim_buffer:  ReplayBuffer,
    real_buffer: ReplayBuffer,
) -> None:
    """Overwrite VecNormalize's obs running-stats with statistics
    computed from the combined sim + real observation corpus.

    This ensures the normalizer covers both distributions before any
    dynamics-model training or SAC policy training begins.
    """
    n_sim  = sim_buffer.size()
    n_real = real_buffer.size()
    sim_obs  = sim_buffer.observations[:n_sim,   0, :].astype(np.float64)
    real_obs = real_buffer.observations[:n_real, 0, :].astype(np.float64)
    all_obs  = np.concatenate([sim_obs, real_obs], axis=0)

    vec_env.obs_rms.mean  = all_obs.mean(0)
    vec_env.obs_rms.var   = np.maximum(all_obs.var(0), 1e-6)
    vec_env.obs_rms.count = float(len(all_obs))

    print(f"VecNormalize fitted on {n_sim} sim + {n_real} real observations "
          f"(total={len(all_obs)})")


def _buffer_to_tensors(
    buffer:  ReplayBuffer,
    vec_env: VecNormalize,
    device:  str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract all valid transitions as normalized tensors.

    Returns (obs, actions, next_obs, rewards) on `device`.
    Obs are normalized via VecNormalize stats (without updating them).
    Rewards are standardized to zero-mean / unit-variance using buffer statistics.
    """
    n        = buffer.size()
    obs_raw      = buffer.observations[:n, 0, :].astype(np.float32)
    acts_raw     = buffer.actions[:n,      0, :].astype(np.float32)
    next_obs_raw = buffer.next_observations[:n, 0, :].astype(np.float32)
    rews_raw     = buffer.rewards[:n, 0].astype(np.float32)

    mean = vec_env.obs_rms.mean.astype(np.float32)
    std  = np.sqrt(vec_env.obs_rms.var.astype(np.float32) + vec_env.epsilon)
    obs_norm      = (obs_raw      - mean) / std
    next_obs_norm = (next_obs_raw - mean) / std

    rew_mean = rews_raw.mean()
    rew_std  = rews_raw.std() + 1e-8
    rews_norm = (rews_raw - rew_mean) / rew_std

    return (
        torch.FloatTensor(obs_norm).to(device),
        torch.FloatTensor(acts_raw).to(device),
        torch.FloatTensor(next_obs_norm).to(device),
        torch.FloatTensor(rews_norm).to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# World-model training  (pre-train / fine-tune, shared logic)
# ─────────────────────────────────────────────────────────────────────────────

def train_world_model(
    dynamics:   EnsembleDynamicsModel,
    buffer:     ReplayBuffer,
    vec_env:    VecNormalize,
    n_epochs:   int,
    batch_size: int,
    val_ratio:  float,
    patience:   int,
    lr:         float,
    device:     str,
    phase_name: str,
    use_wandb:  bool = False,
) -> float:
    """Train the dynamics ensemble on `buffer` until validation loss converges.

    Parameters
    ----------
    phase_name : "pretrain" (on sim data) or "finetune" (on real data).
                 Used for logging only.
    Returns the best achieved validation NLL.
    """
    print(f"\n{'='*60}")
    print(f"  World Model  ·  {phase_name.upper()}")
    print(f"{'='*60}")

    dynamics.set_lr(lr)

    # Extract all data as tensors and perform a fixed train / val split
    obs, acts, next_obs, rews = _buffer_to_tensors(buffer, vec_env, device)
    n = obs.shape[0]

    perm    = torch.randperm(n)
    n_val   = max(batch_size, int(n * val_ratio))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    val_obs,  val_acts,  val_next, val_rews = (
        obs[val_idx], acts[val_idx], next_obs[val_idx], rews[val_idx]
    )
    tr_obs, tr_acts, tr_next, tr_rews = (
        obs[tr_idx], acts[tr_idx], next_obs[tr_idx], rews[tr_idx]
    )
    n_train = tr_obs.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    print(f"  Train: {n_train}  |  Val: {n_val}  |  Steps/epoch: {steps_per_epoch}")
    print(f"  Max epochs: {n_epochs}  |  Patience: {patience}  |  LR: {lr}")

    best_val   = float("inf")
    best_state = copy.deepcopy(dynamics.nets.state_dict())
    no_improve = 0
    log_every  = max(1, n_epochs // 10)

    for epoch in range(n_epochs):
        # ── train pass ──────────────────────────────────────────────────
        perm_t = torch.randperm(n_train)
        tr_losses = []
        for i in range(steps_per_epoch):
            bi = perm_t[i * batch_size : (i + 1) * batch_size]
            if len(bi) < 2:
                continue
            loss = dynamics.train_step(tr_obs[bi], tr_acts[bi], tr_next[bi], tr_rews[bi])
            tr_losses.append(loss)

        tr_loss  = float(np.mean(tr_losses)) if tr_losses else float("inf")
        val_loss = dynamics.eval_loss(val_obs, val_acts, val_next, val_rews)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs} │ "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"patience={no_improve}/{patience}")

        if use_wandb:
            wandb.log({
                f"world_model/{phase_name}/train_loss": tr_loss,
                f"world_model/{phase_name}/val_loss":   val_loss,
                f"world_model/{phase_name}/epoch":      epoch + 1,
            })

        # ── early stopping ───────────────────────────────────────────────
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = copy.deepcopy(dynamics.nets.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}  (best val={best_val:.4f})")
                break

    dynamics.nets.load_state_dict(best_state)
    print(f"  ✓ Done.  Best val_loss = {best_val:.4f}\n")
    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real World Model + MBPO lift training")
    parser.add_argument("--config",     type=str, default="configs/lift_MB.yaml")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb",   action="store_true")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    exp_cfg   = cfg["experiment"]
    train_cfg = cfg["training"]
    sac_cfg   = cfg["sac"]
    env_cfg   = cfg["env"]
    mbpo_cfg  = cfg["mbpo"]
    real_cfg  = cfg["real_data"]
    sim_cfg   = cfg["sim_collection"]
    pt_cfg    = cfg["world_model_pretrain"]
    ft_cfg    = cfg["world_model_finetune"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_name or f"{exp_cfg['name']}_{timestamp}"
        wandb.init(
            project=exp_cfg.get("wandb_project", "pick-101"),
            name=run_name,
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    # ── Environments ──────────────────────────────────────────────────────────
    sim_env = DummyVecEnv([lambda: make_env(env_cfg)])
    sim_env = VecNormalize(
        sim_env,
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

    obs_dim = sim_env.observation_space.shape[0]
    act_dim = sim_env.action_space.shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # ── Replay buffers ────────────────────────────────────────────────────────
    _buf_kwargs = dict(
        observation_space=sim_env.observation_space,
        action_space=sim_env.action_space,
        device=device,
        n_envs=1,
    )
    real_buffer  = ReplayBuffer(buffer_size=real_cfg.get("max_transitions", 50_000),  **_buf_kwargs)
    sim_buffer   = ReplayBuffer(buffer_size=sim_cfg.get("buffer_size", 300_000),       **_buf_kwargs)
    model_buffer = ReplayBuffer(buffer_size=mbpo_cfg["model_buffer_size"],             **_buf_kwargs)

    # ── Dynamics model ────────────────────────────────────────────────────────
    dynamics = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        ensemble_size=mbpo_cfg["ensemble_size"],
        hidden_size=mbpo_cfg["model_hidden_size"],
        n_layers=mbpo_cfg["model_hidden_layers"],
        lr=pt_cfg["lr"],
        max_log_var=mbpo_cfg.get("max_log_var", 0.5),
        min_log_var=mbpo_cfg.get("min_log_var", -10.0),
        device=device,
    )

    # ── SAC (policy optimizer) ────────────────────────────────────────────────
    # We drive the training loop manually; SAC's own replay_buffer is unused.
    sac = SAC(
        "MlpPolicy",
        sim_env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=1000,           # placeholder — we manage our own buffers
        learning_starts=0,
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=train_cfg["seed"],
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Collect sim data from MuJoCo
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 1 · Collecting MuJoCo simulation data")
    print("═"*60)
    collect_sim_data(
        sim_env=sim_env,
        sim_buffer=sim_buffer,
        n_steps=sim_cfg["n_steps"],
        use_random_policy=sim_cfg.get("use_random_policy", True),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2 — Load real-world data from file
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 2 · Loading real-world trajectory data")
    print("═"*60)
    n_real = load_real_data(
        path=real_cfg["path"],
        buffer=real_buffer,
        max_transitions=real_cfg.get("max_transitions"),
    )
    print(f"  Loaded {n_real} real transitions.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3 — Fit VecNormalize from combined sim + real data
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 3 · Initializing observation normalizer")
    print("═"*60)
    initialize_vecnormalize_from_data(sim_env, sim_buffer, real_buffer)

    # Sync eval_env stats and dynamics model normalizer
    eval_env.obs_rms.mean  = sim_env.obs_rms.mean.copy()
    eval_env.obs_rms.var   = sim_env.obs_rms.var.copy()
    eval_env.obs_rms.count = sim_env.obs_rms.count
    dynamics.sync_normalizer_from_vecenv(sim_env)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4 — Pre-train world model on sim data until convergence
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 4 · Pre-training world model on sim data")
    print("═"*60)
    train_world_model(
        dynamics=dynamics,
        buffer=sim_buffer,
        vec_env=sim_env,
        n_epochs=pt_cfg["n_epochs"],
        batch_size=pt_cfg["batch_size"],
        val_ratio=pt_cfg["val_ratio"],
        patience=pt_cfg["patience"],
        lr=pt_cfg["lr"],
        device=device,
        phase_name="pretrain",
        use_wandb=use_wandb,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 5 — Fine-tune world model on real data
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 5 · Fine-tuning world model on real data")
    print("═"*60)
    train_world_model(
        dynamics=dynamics,
        buffer=real_buffer,
        vec_env=sim_env,
        n_epochs=ft_cfg["n_epochs"],
        batch_size=ft_cfg["batch_size"],
        val_ratio=ft_cfg["val_ratio"],
        patience=ft_cfg["patience"],
        lr=ft_cfg["lr"],
        device=device,
        phase_name="finetune",
        use_wandb=use_wandb,
    )

    dynamics.save(str(output_dir / "dynamics_pretrained.pt"))
    print(f"  World model saved → {output_dir / 'dynamics_pretrained.pt'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 6 — RL training (SAC + world-model rollouts)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Phase 6 · RL training (MBPO-style)")
    print("═"*60)

    combined_buffer = ThreeWayCombinedBuffer(
        real_buffer=real_buffer,
        sim_buffer=sim_buffer,
        model_buffer=model_buffer,
        real_ratio=mbpo_cfg["real_ratio"],
        sim_ratio=mbpo_cfg["sim_ratio"],
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
            vec_normalize=sim_env,
            log_freq=train_cfg.get("video_log_freq", 50_000),
            camera=train_cfg.get("video_camera", "closeup"),
            verbose=1,
        ))
    callback = CallbackList(callback_list)

    total_timesteps  = train_cfg["timesteps"]
    batch_size       = sac_cfg["batch_size"]
    gradient_steps   = mbpo_cfg.get("gradient_steps", 20)
    rollout_schedule = mbpo_cfg["rollout_schedule"]
    rollouts_per_step = mbpo_cfg["rollouts_per_step"]
    model_gen_freq   = mbpo_cfg.get("model_gen_freq", 250)

    # Initialise SB3 internal state (_last_obs, logger, …)
    total_timesteps, callback = sac._setup_learn(
        total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="mbpo_rl",
        progress_bar=False,
    )
    callback.on_training_start(locals(), globals())

    print(f"\nRL training for {total_timesteps} steps. Output → {output_dir}\n")

    # ── RL loop ──────────────────────────────────────────────────────────────
    for step in range(total_timesteps):
        sac.num_timesteps += 1

        # 1. Collect one step from MuJoCo sim (online exploration)
        action, buffer_action = sac._sample_action(0, action_noise=sac.action_noise, n_envs=1)
        new_obs, reward, dones, infos = sim_env.step(action)
        # Store in sim_buffer (not sac.replay_buffer)
        sac._store_transition(sim_buffer, buffer_action, new_obs, reward, dones, infos)

        callback.update_locals(locals())
        if not callback.on_step():
            break

        # 2. Generate world-model rollouts (periodically)
        if step % model_gen_freq == 0 and sim_buffer.size() >= rollouts_per_step:
            rollout_len = get_rollout_length(step, rollout_schedule)

            # Sample starting obs from a mix of real + sim (normalized via env=)
            n_from_real = rollouts_per_step // 2
            n_from_sim  = rollouts_per_step - n_from_real
            real_start  = real_buffer.sample(n_from_real, env=sim_env).observations.cpu().numpy()
            sim_start   = sim_buffer.sample(n_from_sim,   env=sim_env).observations.cpu().numpy()
            rollout_obs = np.concatenate([real_start, sim_start], axis=0)

            for _ in range(rollout_len):
                with torch.no_grad():
                    obs_t = torch.FloatTensor(rollout_obs).to(device)
                    rollout_actions = sac.actor(obs_t, deterministic=False).cpu().numpy()
                    rollout_actions = np.clip(rollout_actions, -1.0, 1.0)

                next_rollout_obs, rollout_rewards = dynamics.predict(
                    rollout_obs, rollout_actions, sample=True
                )
                next_rollout_obs  = np.clip(next_rollout_obs,  -10.0, 10.0)
                rollout_rewards   = np.clip(rollout_rewards,   -10.0, 10.0)

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
                    "mbpo/rollout_length":    rollout_len,
                    "mbpo/model_buffer_size": model_buffer.size(),
                    "global_step": step,
                })

        # 3. SAC policy update on mixed real + sim + model data
        if (
            sim_buffer.size()   >= batch_size
            and real_buffer.size()  >= batch_size
            and model_buffer.size() >= batch_size
        ):
            orig = sac.replay_buffer
            sac.replay_buffer = combined_buffer
            sac.train(gradient_steps=gradient_steps, batch_size=batch_size)
            sac.replay_buffer = orig

    # ── Finalise ─────────────────────────────────────────────────────────────
    callback.on_training_end()
    sac.save(output_dir / "final_model")
    sim_env.save(output_dir / "vec_normalize.pkl")
    dynamics.save(str(output_dir / "dynamics_final.pt"))

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
