"""Evaluate world model prediction accuracy against the MuJoCo ground-truth simulator.

No gradients are computed and the model weights are never updated.

What this script does
---------------------
  1. Load the trained EnsembleDynamicsModel checkpoint.
  2. Load VecNormalize statistics (so obs/reward scales match training).
  3. Roll out N episodes in MuJoCo using a chosen policy (random or a saved SAC model).
  4. At every step, feed the same (obs, action) pair to the world model and compare
     its predicted (next_obs, reward) against the simulator's ground truth.
  5. Compute and display per-step and multi-step (open-loop rollout) error metrics:
       • MSE / MAE for next_obs and reward  (per-step)
       • Per-dimension MAE for next_obs     (helps identify which dims are hard)
       • Ensemble disagreement (std across members)  — proxy for model uncertainty
       • Multi-step compounding error        (open-loop rollout of H steps)
  6. Optionally save a .npz result file and render comparison plots.

Usage examples
--------------
  # Quick sanity check (random policy, 10 episodes, no plots)
  python eval_world_model.py --model-path runs/lift_mbpo_real/.../dynamics_pretrained.pt \
      --vec-norm  runs/lift_mbpo_real/.../vec_normalize.pkl \
      --config    configs/lift_MB.yaml

  # Evaluate under trained SAC policy, plot and save results
  python eval_world_model.py \
      --model-path  runs/.../dynamics_final.pt \
      --vec-norm    runs/.../vec_normalize.pkl \
      --sac-path    runs/.../final_model.zip \
      --config      configs/lift_MB.yaml \
      --n-episodes  20 \
      --horizon     10 \
      --plot \
      --save-results results/wm_eval.npz
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lift_cube import LiftCubeCartesianEnv


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics model (must match the definition in train_lift_MB.py)
# ─────────────────────────────────────────────────────────────────────────────

class SingleDynamicsNet(nn.Module):
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
        self.backbone     = nn.Sequential(*layers)
        out_dim           = obs_dim + 1
        self.mean_head    = nn.Linear(hidden_size, out_dim)
        self.log_var_head = nn.Linear(hidden_size, out_dim)
        self.max_log_var  = max_log_var
        self.min_log_var  = min_log_var

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = self.backbone(torch.cat([obs, act], dim=-1))
        mean    = self.mean_head(h)
        log_var = self.log_var_head(h)
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        return mean, log_var


class EnsembleDynamicsModel:
    """Read-only wrapper for evaluation — no optimizer, no gradient ops."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble_size: int = 5,
        hidden_size: int   = 200,
        n_layers: int      = 4,
        max_log_var: float = 0.5,
        min_log_var: float = -10.0,
        device: str        = "cpu",
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
        # Permanently in eval mode — no BN/dropout updates, no grad tracking
        for net in self.nets:
            net.eval()

        # Input normalizer (synced from VecNormalize before evaluation)
        self.obs_mean = torch.zeros(obs_dim, device=device)
        self.obs_std  = torch.ones(obs_dim,  device=device)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.nets.load_state_dict(state)
        for net in self.nets:
            net.eval()
        print(f"Loaded world model from {path}")

    def sync_normalizer_from_vecenv(self, vec_env: VecNormalize) -> None:
        mean = vec_env.obs_rms.mean.astype(np.float32)
        std  = np.sqrt(vec_env.obs_rms.var.astype(np.float32) + vec_env.epsilon)
        self.obs_mean = torch.FloatTensor(mean).to(self.device)
        self.obs_std  = torch.FloatTensor(std).to(self.device)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    @torch.no_grad()
    def predict_all_members(
        self,
        obs: np.ndarray,   # (B, obs_dim) normalized
        act: np.ndarray,   # (B, act_dim)
    ) -> Dict[str, np.ndarray]:
        """Return predictions from every ensemble member without sampling.

        Returns a dict with:
          means      : (E, B, obs_dim+1)  per-member predicted means
          log_vars   : (E, B, obs_dim+1)  per-member predicted log-variances
          mean_mean  : (B, obs_dim+1)     ensemble mean of means
          mean_std   : (B, obs_dim+1)     ensemble std of means (disagreement)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(act).to(self.device)
        obs_n = self._normalize_obs(obs_t)

        all_means, all_lvars = [], []
        for net in self.nets:
            m, lv = net(obs_n, act_t)
            all_means.append(m)
            all_lvars.append(lv)

        means    = torch.stack(all_means)
        log_vars = torch.stack(all_lvars)

        return {
            "means":     means.cpu().numpy(),
            "log_vars":  log_vars.cpu().numpy(),
            "mean_mean": means.mean(0).cpu().numpy(),
            "mean_std":  means.std(0).cpu().numpy(),
        }

    @torch.no_grad()
    def predict_mean(
        self,
        obs: np.ndarray,
        act: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble-mean next_obs and reward (deterministic, no sampling).

        Used for open-loop evaluation to avoid noise compounding.
        """
        preds     = self.predict_all_members(obs, act)
        mean      = preds["mean_mean"]        # (B, obs_dim+1)
        next_obs  = obs + mean[:, :-1]
        reward    = mean[:, -1]
        return next_obs, reward


# ─────────────────────────────────────────────────────────────────────────────
# Environment helper
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
# Per-step evaluation (one-step-ahead, closed-loop ground truth)
# ─────────────────────────────────────────────────────────────────────────────

def eval_one_step(
    dynamics:   EnsembleDynamicsModel,
    vec_env:    VecNormalize,
    n_episodes: int,
    policy:     Optional[SAC],
    device:     str,
) -> Dict[str, np.ndarray]:
    """Run n_episodes in MuJoCo and collect per-step one-step prediction errors.

    At each step the simulator provides ground truth (s', r_sim).
    The world model predicts (ŝ', r̂) from the same (s, a) — no weight updates.

    Returns arrays (concatenated across all episodes × steps):
      obs_mae_per_dim : (N, obs_dim)  absolute error per obs dimension
      obs_mse         : (N,)          scalar MSE on next_obs
      rew_ae          : (N,)          absolute error on reward
      disagreement    : (N,)          mean ensemble std (uncertainty proxy)
    """
    results: Dict[str, List] = {
        "obs_mae_per_dim": [],
        "obs_mse":         [],
        "rew_ae":          [],
        "disagreement":    [],
    }

    for ep in range(n_episodes):
        norm_obs = vec_env.reset()
        done = False
        ep_steps = 0

        while not done:
            # ── Action ────────────────────────────────────────────────────
            if policy is not None:
                action, _ = policy.predict(norm_obs, deterministic=False)
            else:
                action = np.array([vec_env.action_space.sample()])

            # ── World model prediction (no gradient) ──────────────────────
            preds          = dynamics.predict_all_members(norm_obs, action)
            wm_mean        = preds["mean_mean"]              # (1, obs_dim+1)
            wm_std         = preds["mean_std"]               # (1, obs_dim+1)
            wm_next_norm   = norm_obs + wm_mean[:, :-1]
            wm_reward      = wm_mean[:, -1]                  # (1,)

            # ── Simulator step (ground truth) ─────────────────────────────
            next_norm_obs, _, terminated, _ = vec_env.step(action)
            sim_reward_raw = vec_env.get_original_reward()   # unnormalized

            # ── Metrics ───────────────────────────────────────────────────
            obs_diff = np.abs(wm_next_norm - next_norm_obs)
            obs_mse  = float(np.mean((wm_next_norm - next_norm_obs) ** 2))
            rew_ae   = float(np.abs(wm_reward - sim_reward_raw))
            disagree = float(wm_std.mean())

            results["obs_mae_per_dim"].append(obs_diff[0])
            results["obs_mse"].append(obs_mse)
            results["rew_ae"].append(rew_ae)
            results["disagreement"].append(disagree)

            norm_obs = next_norm_obs
            done = bool(terminated[0])
            ep_steps += 1

        print(f"  Episode {ep+1}/{n_episodes}  ({ep_steps} steps)")

    return {k: np.array(v) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step open-loop evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_open_loop(
    dynamics:   EnsembleDynamicsModel,
    vec_env:    VecNormalize,
    n_episodes: int,
    horizon:    int,
    policy:     Optional[SAC],
    device:     str,
) -> Dict[str, np.ndarray]:
    """Measure how fast world-model error compounds over H steps.

    Protocol per episode:
      1. Reset sim, record s_0.
      2. Run H steps in sim with the policy → ground-truth trajectory.
      3. Run the world model open-loop from s_0 using the same actions
         (world model never observes the true s_t for t > 0).
      4. Record |ŝ_t - s_t| for t in {1 … H}.

    Returns:
      obs_mae_per_step : (n_ep, H, obs_dim)
      obs_mse_per_step : (n_ep, H)
      rew_ae_per_step  : (n_ep, H)
    """
    ep_obs_mae, ep_obs_mse, ep_rew_ae = [], [], []

    for ep in range(n_episodes):
        # ── Ground-truth trajectory ────────────────────────────────────────
        gt_obs, gt_acts, gt_rews = [], [], []
        norm_obs = vec_env.reset()
        gt_obs.append(norm_obs[0].copy())

        for h in range(horizon):
            if policy is not None:
                action, _ = policy.predict(norm_obs, deterministic=True)
            else:
                action = np.array([vec_env.action_space.sample()])

            next_norm_obs, _, terminated, _ = vec_env.step(action)
            raw_reward = vec_env.get_original_reward()

            gt_acts.append(action[0].copy())
            gt_obs.append(next_norm_obs[0].copy())
            gt_rews.append(float(raw_reward[0]))

            norm_obs = next_norm_obs
            if bool(terminated[0]):
                # Pad with terminal state for remaining steps
                for _ in range(horizon - h - 1):
                    gt_acts.append(action[0].copy())
                    gt_obs.append(next_norm_obs[0].copy())
                    gt_rews.append(float(raw_reward[0]))
                break

        # ── Open-loop world model rollout ──────────────────────────────────
        wm_obs = gt_obs[0].copy()
        step_mae, step_mse, step_rew_ae = [], [], []

        for h in range(min(horizon, len(gt_acts))):
            obs_b = wm_obs[np.newaxis, :]
            act_b = gt_acts[h][np.newaxis, :]

            wm_next, wm_rew = dynamics.predict_mean(obs_b, act_b)
            wm_next = wm_next[0]
            wm_rew  = float(wm_rew[0])

            gt_next = gt_obs[h + 1]
            gt_rew  = gt_rews[h]

            step_mae.append(np.abs(wm_next - gt_next))
            step_mse.append(float(np.mean((wm_next - gt_next) ** 2)))
            step_rew_ae.append(abs(wm_rew - gt_rew))

            wm_obs = wm_next   # feed prediction back (open loop)

        ep_obs_mae.append(np.stack(step_mae))
        ep_obs_mse.append(np.array(step_mse))
        ep_rew_ae.append(np.array(step_rew_ae))

        final_mse = step_mse[-1] if step_mse else float("nan")
        final_rae = step_rew_ae[-1] if step_rew_ae else float("nan")
        print(f"  Episode {ep+1}/{n_episodes}  H={len(step_mse)}  "
              f"final obs MSE={final_mse:.4f}  final rew AE={final_rae:.4f}")

    return {
        "obs_mae_per_step": np.stack(ep_obs_mae),
        "obs_mse_per_step": np.stack(ep_obs_mse),
        "rew_ae_per_step":  np.stack(ep_rew_ae),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print summaries
# ─────────────────────────────────────────────────────────────────────────────

def print_one_step_summary(results: Dict[str, np.ndarray], obs_dim: int) -> None:
    mae   = results["obs_mae_per_dim"]
    mse   = results["obs_mse"]
    rew   = results["rew_ae"]
    disag = results["disagreement"]

    print("\n" + "═" * 60)
    print("  ONE-STEP PREDICTION SUMMARY")
    print("═" * 60)
    print(f"  Steps evaluated     : {len(mse)}")
    print(f"  Next-obs MSE        : mean={mse.mean():.5f}   std={mse.std():.5f}")
    print(f"  Reward   MAE        : mean={rew.mean():.5f}   std={rew.std():.5f}")
    print(f"  Ensemble disagreement: mean={disag.mean():.5f}   std={disag.std():.5f}")

    print(f"\n  Per-dimension MAE  (obs_dim={obs_dim})")
    print(f"  {'Dim':>4}  {'Mean':>10}  {'Std':>10}  {'Max':>10}")
    print(f"  {'-'*44}")
    for d in range(obs_dim):
        print(f"  {d:>4}  {mae[:, d].mean():>10.5f}  "
              f"{mae[:, d].std():>10.5f}  {mae[:, d].max():>10.5f}")
    print("═" * 60)


def print_open_loop_summary(results: Dict[str, np.ndarray]) -> None:
    mse = results["obs_mse_per_step"]   # (n_ep, H)
    rew = results["rew_ae_per_step"]    # (n_ep, H)
    H   = mse.shape[1]

    print("\n" + "═" * 60)
    print("  OPEN-LOOP MULTI-STEP SUMMARY")
    print("═" * 60)
    print(f"  {'Step':>5}  {'Obs MSE (mean)':>16}  {'Obs MSE (std)':>14}  {'Rew MAE (mean)':>16}")
    print(f"  {'-'*56}")
    for h in range(H):
        print(f"  {h+1:>5}  {mse[:, h].mean():>16.5f}  "
              f"{mse[:, h].std():>14.5f}  {rew[:, h].mean():>16.5f}")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Optional matplotlib plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    one_step:  Dict[str, np.ndarray],
    open_loop: Dict[str, np.ndarray],
    obs_dim:   int,
    save_dir:  Optional[Path] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not found — skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("World Model vs MuJoCo Simulator", fontsize=14, fontweight="bold")

    # (0,0) One-step obs MSE histogram
    ax = axes[0, 0]
    ax.hist(one_step["obs_mse"], bins=50, color="steelblue", edgecolor="white", lw=0.4)
    ax.axvline(one_step["obs_mse"].mean(), color="firebrick", ls="--",
               label=f'mean={one_step["obs_mse"].mean():.4f}')
    ax.set_title("One-step  |  Obs MSE distribution")
    ax.set_xlabel("MSE (normalized obs space)")
    ax.set_ylabel("Count")
    ax.legend()

    # (0,1) Per-dimension MAE bar chart
    ax = axes[0, 1]
    mae = one_step["obs_mae_per_dim"]
    ax.bar(np.arange(obs_dim), mae.mean(0), yerr=mae.std(0),
           color="steelblue", error_kw=dict(ecolor="black", capsize=3, lw=1))
    ax.set_title("One-step  |  Per-dimension MAE")
    ax.set_xlabel("Observation dimension")
    ax.set_ylabel("MAE (normalized)")
    ax.set_xticks(np.arange(obs_dim))

    # (1,0) Open-loop obs MSE vs horizon
    ax = axes[1, 0]
    mse = open_loop["obs_mse_per_step"]
    steps = np.arange(1, mse.shape[1] + 1)
    ax.plot(steps, mse.mean(0), color="steelblue", lw=2, label="mean")
    ax.fill_between(steps, mse.mean(0) - mse.std(0), mse.mean(0) + mse.std(0),
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.set_title("Open-loop  |  Obs MSE vs horizon")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MSE (normalized obs)")
    ax.legend()

    # (1,1) Open-loop reward MAE vs horizon
    ax = axes[1, 1]
    rew = open_loop["rew_ae_per_step"]
    ax.plot(steps, rew.mean(0), color="darkorange", lw=2, label="mean")
    ax.fill_between(steps, rew.mean(0) - rew.std(0), rew.mean(0) + rew.std(0),
                    alpha=0.25, color="darkorange", label="±1 std")
    ax.set_title("Open-loop  |  Reward MAE vs horizon")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Absolute reward error (raw)")
    ax.legend()

    plt.tight_layout()

    if save_dir is not None:
        out = save_dir / "world_model_eval.png"
        plt.savefig(out, dpi=150)
        print(f"[plot] Saved → {out}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare world model predictions against MuJoCo simulator (no updates)"
    )
    parser.add_argument("--model-path",    type=str, required=True,
                        help="Path to dynamics .pt checkpoint")
    parser.add_argument("--vec-norm",      type=str, required=True,
                        help="Path to VecNormalize .pkl stats file")
    parser.add_argument("--config",        type=str, default="configs/lift_MB.yaml")
    parser.add_argument("--sac-path",      type=str, default=None,
                        help="(Optional) SAC .zip — uses random policy if omitted")
    parser.add_argument("--n-episodes",    type=int, default=10,
                        help="Episodes for one-step evaluation")
    parser.add_argument("--horizon",       type=int, default=10,
                        help="Open-loop rollout horizon H")
    parser.add_argument("--n-episodes-ol", type=int, default=None,
                        help="Episodes for open-loop eval (defaults to --n-episodes)")
    parser.add_argument("--plot",          action="store_true",
                        help="Render comparison plots")
    parser.add_argument("--save-results",  type=str, default=None,
                        help="Save metrics to .npz  (e.g. results/wm_eval.npz)")
    parser.add_argument("--no-cuda",       action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg  = cfg["env"]
    mbpo_cfg = cfg["mbpo"]

    # ── Environment — VecNormalize frozen (training=False) ─────────────────
    raw_env = DummyVecEnv([lambda: make_env(env_cfg)])
    vec_env = VecNormalize.load(args.vec_norm, raw_env)
    vec_env.training    = False   # do NOT update running stats
    vec_env.norm_reward = False   # raw rewards fetched via get_original_reward()

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.shape[0]
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")

    # ── World model — eval mode, no optimizer ──────────────────────────────
    dynamics = EnsembleDynamicsModel(
        obs_dim       = obs_dim,
        act_dim       = act_dim,
        ensemble_size = mbpo_cfg["ensemble_size"],
        hidden_size   = mbpo_cfg["model_hidden_size"],
        n_layers      = mbpo_cfg["model_hidden_layers"],
        max_log_var   = mbpo_cfg.get("max_log_var",  0.5),
        min_log_var   = mbpo_cfg.get("min_log_var", -10.0),
        device        = device,
    )
    dynamics.load(args.model_path)
    dynamics.sync_normalizer_from_vecenv(vec_env)

    # ── Optional SAC policy ────────────────────────────────────────────────
    policy: Optional[SAC] = None
    if args.sac_path:
        policy = SAC.load(args.sac_path, env=vec_env, device=device)
        print(f"Loaded SAC policy from {args.sac_path}")
    else:
        print("No SAC policy provided — using random policy.")

    # ── One-step evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  One-step evaluation  ({args.n_episodes} episodes)")
    print(f"{'='*60}")
    one_step = eval_one_step(dynamics, vec_env, args.n_episodes, policy, device)
    print_one_step_summary(one_step, obs_dim)

    # ── Open-loop evaluation ───────────────────────────────────────────────
    n_ol = args.n_episodes_ol or args.n_episodes
    print(f"\n{'='*60}")
    print(f"  Open-loop evaluation  ({n_ol} episodes, horizon={args.horizon})")
    print(f"{'='*60}")
    open_loop = eval_open_loop(dynamics, vec_env, n_ol, args.horizon, policy, device)
    print_open_loop_summary(open_loop)

    # ── Save raw results ───────────────────────────────────────────────────
    if args.save_results:
        out = Path(args.save_results)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out,
                 one_step_obs_mae_per_dim   = one_step["obs_mae_per_dim"],
                 one_step_obs_mse           = one_step["obs_mse"],
                 one_step_rew_ae            = one_step["rew_ae"],
                 one_step_disagreement      = one_step["disagreement"],
                 open_loop_obs_mae_per_step = open_loop["obs_mae_per_step"],
                 open_loop_obs_mse_per_step = open_loop["obs_mse_per_step"],
                 open_loop_rew_ae_per_step  = open_loop["rew_ae_per_step"])
        print(f"\nResults saved → {out}")

    # ── Plots ──────────────────────────────────────────────────────────────
    if args.plot:
        save_dir = Path(args.save_results).parent if args.save_results else None
        plot_results(one_step, open_loop, obs_dim, save_dir)


if __name__ == "__main__":
    main()
