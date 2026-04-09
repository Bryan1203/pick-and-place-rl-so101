"""Collect SO-101 simulator data and train a world model checkpoint.

Standalone script — no RL policy training involved.

Pipeline
--------
  Step 1  Roll out a random (or scripted) policy in MuJoCo to collect transitions.
          Optionally load additional data from a .npz file (e.g. real robot data).
  Step 2  Fit VecNormalize statistics on the collected corpus.
  Step 3  Train the ensemble dynamics model until validation loss converges
          (early stopping). Progress is printed every epoch.
  Step 4  Save outputs to --out-dir:
            world_model.pt        — ensemble weights (load with dynamics.load())
            vec_normalize.pkl     — observation/reward normalizer
            dataset.npz           — raw collected transitions (reusable)
            training_curve.npz    — per-epoch train/val loss history

Usage examples
--------------
  # Collect 100k sim steps, train world model, save to checkpoints/wm_run1/
  python model_base/collect_and_train_world_model.py \
      --config model_base/configs/lift_MB.yaml \
      --sim-steps 100000 \
      --out-dir checkpoints/wm_run1

  # Use scripted policy for better coverage, also mix in real data
  python model_base/collect_and_train_world_model.py \
      --config model_base/configs/lift_MB.yaml \
      --sim-steps 100000 \
      --policy scripted \
      --real-data data/real_trajectories.npz \
      --out-dir checkpoints/wm_run2

  # Resume: skip collection, train on an existing dataset
  python model_base/collect_and_train_world_model.py \
      --config model_base/configs/lift_MB.yaml \
      --load-dataset checkpoints/wm_run1/dataset.npz \
      --out-dir checkpoints/wm_run1_retrain
"""
import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── allow running from repo root or from model_base/ ─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.envs.lift_cube import LiftCubeCartesianEnv


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Dynamics Model
# ─────────────────────────────────────────────────────────────────────────────

class SingleDynamicsNet(nn.Module):
    """(obs, action) → (mean, log_var) for a Gaussian over (Δobs ∥ reward)."""

    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        hidden_size: int   = 200,
        n_layers:    int   = 4,
        max_log_var: float = 0.5,
        min_log_var: float = -10.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(obs_dim + act_dim, hidden_size), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.SiLU()]
        self.backbone     = nn.Sequential(*layers)
        out_dim           = obs_dim + 1          # Δobs + reward
        self.mean_head    = nn.Linear(hidden_size, out_dim)
        self.log_var_head = nn.Linear(hidden_size, out_dim)
        self.max_log_var  = max_log_var
        self.min_log_var  = min_log_var

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.backbone(torch.cat([obs, act], dim=-1))
        mean    = self.mean_head(h)
        log_var = self.log_var_head(h)
        # Double-softplus clamp  (Chua et al., 2018)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var


class EnsembleDynamicsModel:
    """E independently trained probabilistic dynamics networks."""

    def __init__(
        self,
        obs_dim:       int,
        act_dim:       int,
        ensemble_size: int   = 5,
        hidden_size:   int   = 200,
        n_layers:      int   = 4,
        lr:            float = 1e-3,
        max_log_var:   float = 0.5,
        min_log_var:   float = -10.0,
        device:        str   = "cpu",
    ):
        self.obs_dim       = obs_dim
        self.act_dim       = act_dim
        self.ensemble_size = ensemble_size
        self.device        = device

        self.nets = nn.ModuleList([
            SingleDynamicsNet(obs_dim, act_dim, hidden_size, n_layers, max_log_var, min_log_var)
            for _ in range(ensemble_size)
        ])
        self.nets.to(device)
        self.optimizer = torch.optim.Adam(self.nets.parameters(), lr=lr)

        # Input normalizer (synced from VecNormalize after data collection)
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_std  = torch.ones(obs_dim,  device=device)

    def sync_normalizer_from_vecenv(self, vec_env: VecNormalize) -> None:
        mean = vec_env.obs_rms.mean.astype(np.float32)
        std  = np.sqrt(vec_env.obs_rms.var.astype(np.float32) + vec_env.epsilon)
        self.obs_mean = torch.FloatTensor(mean).to(self.device)
        self.obs_std  = torch.FloatTensor(std).to(self.device)

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _normalize(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_n = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs_n, act   # actions already in [-1, 1]

    def train_step(
        self,
        obs:      torch.Tensor,   # (B, obs_dim) normalized
        act:      torch.Tensor,   # (B, act_dim)
        next_obs: torch.Tensor,   # (B, obs_dim) normalized
        rew:      torch.Tensor,   # (B,)
    ) -> float:
        obs_n, act_n = self._normalize(obs, act)
        target = torch.cat([next_obs - obs, rew.unsqueeze(-1)], dim=-1)

        self.optimizer.zero_grad()
        total = torch.tensor(0.0, device=self.device)
        for net in self.nets:
            mean, log_var = net(obs_n, act_n)
            inv_var = torch.exp(-log_var)
            nll = (inv_var * (target - mean).pow(2) + log_var).sum(-1).mean()
            total = total + nll
        total.backward()
        self.optimizer.step()
        return (total / self.ensemble_size).item()

    @torch.no_grad()
    def eval_loss(
        self,
        obs: torch.Tensor, act: torch.Tensor,
        next_obs: torch.Tensor, rew: torch.Tensor,
        batch_size: int = 512,
    ) -> float:
        n, total, count = obs.shape[0], 0.0, 0
        for i in range(0, n, batch_size):
            ob  = obs[i:i+batch_size];  ac  = act[i:i+batch_size]
            no  = next_obs[i:i+batch_size]; rw = rew[i:i+batch_size]
            ob_n, ac_n = self._normalize(ob, ac)
            tgt = torch.cat([no - ob, rw.unsqueeze(-1)], dim=-1)
            for net in self.nets:
                m, lv   = net(ob_n, ac_n)
                inv_var = torch.exp(-lv)
                nll     = (inv_var * (tgt - m).pow(2) + lv).sum(-1).mean()
                total  += nll.item()
            count += 1
        return total / max(count * self.ensemble_size, 1)

    def save(self, path: str) -> None:
        torch.save({
            "nets_state_dict": self.nets.state_dict(),
            "obs_mean": self.obs_mean.cpu(),
            "obs_std":  self.obs_std.cpu(),
            "obs_dim":  self.obs_dim,
            "act_dim":  self.act_dim,
            "ensemble_size": self.ensemble_size,
        }, path)
        print(f"  World model saved → {path}")

    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = "cpu") -> "EnsembleDynamicsModel":
        ckpt = torch.load(path, map_location=device)
        model = cls(
            obs_dim       = ckpt["obs_dim"],
            act_dim       = ckpt["act_dim"],
            ensemble_size = ckpt["ensemble_size"],
            device        = device,
        )
        model.nets.load_state_dict(ckpt["nets_state_dict"])
        model.obs_mean = ckpt["obs_mean"].to(device)
        model.obs_std  = ckpt["obs_std"].to(device)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Scripted policy (simple phase-based)
# ─────────────────────────────────────────────────────────────────────────────

def scripted_action(step_in_ep: int, rng: np.random.Generator) -> np.ndarray:
    """Phase-based scripted policy: descend → close → lift."""
    n = 0.15   # noise scale
    if step_in_ep < 25:          # descend, open gripper
        return np.array([rng.uniform(-1,1)*n, rng.uniform(-1,1)*n, -0.6+rng.uniform(-1,1)*n, 0.8], dtype=np.float32)
    elif step_in_ep < 45:        # close gripper
        return np.array([rng.uniform(-1,1)*n*0.3, rng.uniform(-1,1)*n*0.3, 0.0, -1.0], dtype=np.float32)
    else:                        # lift
        return np.array([rng.uniform(-1,1)*n*0.2, rng.uniform(-1,1)*n*0.2, 0.7, -1.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_sim_data(
    vec_env:    VecNormalize,
    n_steps:    int,
    policy:     str,   # "random" | "scripted"
    seed:       int,
) -> Dict[str, np.ndarray]:
    """Roll out the simulator for n_steps and return a transition dataset.

    Stores raw (unnormalized) observations so the dataset can be reused
    with different normalizers.

    Returns dict with keys:
        observations, next_observations, actions, rewards, dones
    """
    rng = np.random.default_rng(seed)

    obs_list, next_obs_list, act_list, rew_list, done_list = [], [], [], [], []

    obs = vec_env.reset()
    last_raw_obs = vec_env.get_original_obs().copy()   # (1, obs_dim) unnormalized
    ep_step = 0
    collected = 0

    print(f"  Collecting {n_steps} steps (policy={policy}) …")
    while collected < n_steps:
        # ── Choose action ─────────────────────────────────────────────────
        if policy == "scripted":
            action = scripted_action(ep_step, rng)[np.newaxis, :]   # (1, act_dim)
        else:
            action = np.array([vec_env.action_space.sample()])

        action = np.clip(action, -1.0, 1.0)

        # ── Step ──────────────────────────────────────────────────────────
        next_obs, _norm_rew, done, info = vec_env.step(action)

        raw_next_obs = vec_env.get_original_obs().copy()   # unnormalized next obs
        raw_reward   = vec_env.get_original_reward().copy()

        # Use terminal observation if episode ended mid-step
        stored_next = raw_next_obs.copy()
        for i, d in enumerate(done):
            if d and info[i].get("terminal_observation") is not None:
                stored_next[i] = info[i]["terminal_observation"]

        obs_list.append(last_raw_obs[0].copy())
        next_obs_list.append(stored_next[0].copy())
        act_list.append(action[0].copy())
        rew_list.append(float(raw_reward[0]))
        done_list.append(bool(done[0]))

        last_raw_obs = raw_next_obs.copy()
        collected += 1
        ep_step   += 1

        if any(done):
            obs = vec_env.reset()
            last_raw_obs = vec_env.get_original_obs().copy()
            ep_step = 0
        else:
            obs = next_obs

        if collected % max(1, n_steps // 10) == 0:
            print(f"    {collected}/{n_steps} steps collected …")

    return {
        "observations":      np.array(obs_list,      dtype=np.float32),
        "next_observations": np.array(next_obs_list, dtype=np.float32),
        "actions":           np.array(act_list,      dtype=np.float32),
        "rewards":           np.array(rew_list,      dtype=np.float32),
        "dones":             np.array(done_list,     dtype=bool),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Normalizer fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_vecnormalize_from_dataset(
    vec_env: VecNormalize,
    dataset: Dict[str, np.ndarray],
) -> None:
    """Set VecNormalize obs statistics from the collected dataset."""
    all_obs = np.concatenate([
        dataset["observations"],
        dataset["next_observations"],
    ], axis=0).astype(np.float64)

    vec_env.obs_rms.mean  = all_obs.mean(0)
    vec_env.obs_rms.var   = np.maximum(all_obs.var(0), 1e-6)
    vec_env.obs_rms.count = float(len(all_obs))

    print(f"  VecNormalize fitted on {len(all_obs)} observations.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — World model training
# ─────────────────────────────────────────────────────────────────────────────

def dataset_to_tensors(
    dataset: Dict[str, np.ndarray],
    vec_env: VecNormalize,
    device:  str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize dataset with VecNormalize stats and return tensors."""
    obs_raw      = dataset["observations"]
    next_obs_raw = dataset["next_observations"]
    acts_raw     = dataset["actions"]
    rews_raw     = dataset["rewards"]

    mean = vec_env.obs_rms.mean.astype(np.float32)
    std  = np.sqrt(vec_env.obs_rms.var.astype(np.float32) + vec_env.epsilon)

    obs_norm      = (obs_raw      - mean) / std
    next_obs_norm = (next_obs_raw - mean) / std

    # Standardize rewards with dataset statistics
    rew_mean = rews_raw.mean()
    rew_std  = rews_raw.std() + 1e-8
    rews_norm = (rews_raw - rew_mean) / rew_std

    return (
        torch.FloatTensor(obs_norm).to(device),
        torch.FloatTensor(acts_raw).to(device),
        torch.FloatTensor(next_obs_norm).to(device),
        torch.FloatTensor(rews_norm).to(device),
    )


def train_world_model(
    dynamics:   EnsembleDynamicsModel,
    dataset:    Dict[str, np.ndarray],
    vec_env:    VecNormalize,
    n_epochs:   int,
    batch_size: int,
    val_ratio:  float,
    patience:   int,
    lr:         float,
    device:     str,
) -> Tuple[List[float], List[float]]:
    """Train the dynamics ensemble until validation loss converges.

    Returns (train_loss_history, val_loss_history) per epoch.
    """
    dynamics.set_lr(lr)
    dynamics.sync_normalizer_from_vecenv(vec_env)

    obs, acts, next_obs, rews = dataset_to_tensors(dataset, vec_env, device)
    n = obs.shape[0]

    # Train / val split
    perm    = torch.randperm(n)
    n_val   = max(batch_size, int(n * val_ratio))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]

    val_obs,  val_acts,  val_next,  val_rews  = obs[val_idx], acts[val_idx], next_obs[val_idx], rews[val_idx]
    tr_obs,   tr_acts,   tr_next,   tr_rews   = obs[tr_idx],  acts[tr_idx],  next_obs[tr_idx],  rews[tr_idx]

    n_train         = tr_obs.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    log_every       = max(1, n_epochs // 20)

    print(f"\n  Train: {n_train} samples  |  Val: {n_val} samples")
    print(f"  Steps/epoch: {steps_per_epoch}  |  Max epochs: {n_epochs}  |  Patience: {patience}  |  LR: {lr}")

    best_val   = float("inf")
    best_state = copy.deepcopy(dynamics.nets.state_dict())
    no_improve = 0
    tr_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(n_epochs):
        # ── train pass ────────────────────────────────────────────────────
        perm_t    = torch.randperm(n_train)
        tr_losses = []
        for i in range(steps_per_epoch):
            bi   = perm_t[i * batch_size : (i + 1) * batch_size]
            if len(bi) < 2:
                continue
            loss = dynamics.train_step(tr_obs[bi], tr_acts[bi], tr_next[bi], tr_rews[bi])
            tr_losses.append(loss)

        tr_loss  = float(np.mean(tr_losses)) if tr_losses else float("inf")
        val_loss = dynamics.eval_loss(val_obs, val_acts, val_next, val_rews)
        tr_history.append(tr_loss)
        val_history.append(val_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  │  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"patience={no_improve}/{patience}")

        # ── early stopping ────────────────────────────────────────────────
        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = copy.deepcopy(dynamics.nets.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}  (best val={best_val:.4f})")
                break

    dynamics.nets.load_state_dict(best_state)
    print(f"\n  ✓ Training complete.  Best val_loss = {best_val:.4f}")
    return tr_history, val_history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect SO-101 sim data and train a world model checkpoint"
    )
    parser.add_argument("--config",        type=str, default="model_base/configs/lift_MB.yaml",
                        help="Config YAML (uses world_model_pretrain section)")
    parser.add_argument("--sim-steps",     type=int, default=None,
                        help="Steps to collect from MuJoCo sim "
                             "(overrides sim_collection.n_steps in config)")
    parser.add_argument("--policy",        type=str, default="random",
                        choices=["random", "scripted"],
                        help="Policy used during sim data collection")
    parser.add_argument("--real-data",     type=str, default=None,
                        help="Optional .npz of real-world data to mix in "
                             "(augments sim dataset before training)")
    parser.add_argument("--load-dataset",  type=str, default=None,
                        help="Skip collection: load an existing dataset.npz")
    parser.add_argument("--out-dir",       type=str, default="checkpoints/world_model",
                        help="Directory to save all outputs")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg  = cfg["env"]
    mbpo_cfg = cfg["mbpo"]
    pt_cfg   = cfg["world_model_pretrain"]
    sim_cfg  = cfg.get("sim_collection", {})

    n_sim_steps = args.sim_steps or sim_cfg.get("n_steps", 100_000)

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── VecNormalize environment ───────────────────────────────────────────
    vec_env = VecNormalize(
        DummyVecEnv([lambda: make_env(env_cfg)]),
        norm_obs=env_cfg.get("normalize_obs", True),
        norm_reward=env_cfg.get("normalize_reward", True),
        training=True,
    )

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1 — Dataset
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Step 1 · Dataset")
    print("═"*60)

    if args.load_dataset:
        # ── Reuse existing dataset ─────────────────────────────────────────
        print(f"  Loading dataset from {args.load_dataset}")
        sim_dataset = dict(np.load(args.load_dataset, allow_pickle=True))
        print(f"  Loaded {len(sim_dataset['rewards'])} transitions.")
    else:
        # ── Collect from MuJoCo ────────────────────────────────────────────
        sim_dataset = collect_sim_data(
            vec_env  = vec_env,
            n_steps  = n_sim_steps,
            policy   = args.policy,
            seed     = args.seed,
        )
        dataset_path = out_dir / "dataset.npz"
        np.savez_compressed(dataset_path, **sim_dataset)
        print(f"  Sim dataset saved → {dataset_path}")

    # ── Optionally mix in real data ────────────────────────────────────────
    if args.real_data:
        print(f"\n  Mixing in real data from {args.real_data}")
        real = dict(np.load(args.real_data, allow_pickle=True))
        dataset = {
            k: np.concatenate([sim_dataset[k], real[k]], axis=0)
            for k in ("observations", "next_observations", "actions", "rewards", "dones")
        }
        print(f"  Combined: {len(sim_dataset['rewards'])} sim + "
              f"{len(real['rewards'])} real = {len(dataset['rewards'])} total")
    else:
        dataset = sim_dataset

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2 — Fit VecNormalize
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Step 2 · Fitting observation normalizer")
    print("═"*60)
    fit_vecnormalize_from_dataset(vec_env, dataset)
    vec_env.save(str(out_dir / "vec_normalize.pkl"))
    print(f"  VecNormalize saved → {out_dir / 'vec_normalize.pkl'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3 — Train world model
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Step 3 · Training world model")
    print("═"*60)

    dynamics = EnsembleDynamicsModel(
        obs_dim       = obs_dim,
        act_dim       = act_dim,
        ensemble_size = mbpo_cfg["ensemble_size"],
        hidden_size   = mbpo_cfg["model_hidden_size"],
        n_layers      = mbpo_cfg["model_hidden_layers"],
        lr            = pt_cfg["lr"],
        max_log_var   = mbpo_cfg.get("max_log_var",  0.5),
        min_log_var   = mbpo_cfg.get("min_log_var", -10.0),
        device        = device,
    )

    tr_history, val_history = train_world_model(
        dynamics   = dynamics,
        dataset    = dataset,
        vec_env    = vec_env,
        n_epochs   = pt_cfg["n_epochs"],
        batch_size = pt_cfg["batch_size"],
        val_ratio  = pt_cfg["val_ratio"],
        patience   = pt_cfg["patience"],
        lr         = pt_cfg["lr"],
        device     = device,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4 — Save outputs
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("  Step 4 · Saving outputs")
    print("═"*60)

    dynamics.save(str(out_dir / "world_model.pt"))

    np.savez(
        out_dir / "training_curve.npz",
        train_loss = np.array(tr_history),
        val_loss   = np.array(val_history),
    )
    print(f"  Training curve saved → {out_dir / 'training_curve.npz'}")

    # ── Print final summary ────────────────────────────────────────────────
    print(f"""
{'='*60}
  Done!  Outputs in {out_dir}/
    world_model.pt       — ensemble dynamics weights + normalizer
    vec_normalize.pkl    — VecNormalize observation stats
    dataset.npz          — raw collected transitions
    training_curve.npz   — per-epoch train/val NLL history

  To fine-tune on real data, pass this checkpoint to train_lift_MB.py.
  To evaluate prediction accuracy, run eval_world_model.py:

    python model_base/eval_world_model.py \\
        --model-path {out_dir}/world_model.pt \\
        --vec-norm   {out_dir}/vec_normalize.pkl \\
        --config     {args.config}
{'='*60}""")


if __name__ == "__main__":
    main()
