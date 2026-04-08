"""Collect real-world transition data from the SO-101 robot and save as .npz.

The saved file is directly consumable by load_real_data() in train_lift_MB.py.

Observation layout (21 dims) — must match LiftCubeCartesianEnv._get_obs():
    [0:6]   joint positions   (rad)
    [6:12]  joint velocities  (rad/s)
    [12:15] gripper XYZ position  (m)
    [15:18] gripper Euler angles  (rad)
    [18:21] cube XYZ position     (m)

Action layout (4 dims):
    [0]   delta X   (scaled, [-1, 1])
    [1]   delta Y
    [2]   delta Z
    [3]   gripper   (-1 = close, +1 = open)

Data source options
-------------------
  --mode sim      Use MuJoCo sim as a drop-in test (no real robot needed).
                  Useful to verify the pipeline before deploying on hardware.
  --mode scripted Run the scripted grasp policy inside MuJoCo.
  --mode robot    Interface with the physical SO-101 via your robot driver.
                  Fill in RealSO101Robot below with your actual SDK calls.
  --mode teleop   Teleoperation: you control the robot manually; script records.

Usage
-----
  # Test pipeline with sim (no robot required)
  python collect_real_data.py --mode sim --episodes 20 --out data/real_trajectories.npz

  # Scripted policy in sim
  python collect_real_data.py --mode scripted --episodes 100 --out data/real_trajectories.npz

  # Real robot (fill in RealSO101Robot first)
  python collect_real_data.py --mode robot --episodes 50 --out data/real_trajectories.npz

  # Verify a saved file
  python collect_real_data.py --verify data/real_trajectories.npz
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Expected data schema (matches load_real_data() in train_lift_MB.py)
# ─────────────────────────────────────────────────────────────────────────────

OBS_DIM = 21   # joint_pos(6) + joint_vel(6) + gripper_pos(3) + gripper_euler(3) + cube_pos(3)
ACT_DIM = 4    # delta_x, delta_y, delta_z, gripper

REQUIRED_KEYS = ("observations", "next_observations", "actions", "rewards", "dones")


# ─────────────────────────────────────────────────────────────────────────────
# Real robot interface — fill this in with your actual SDK
# ─────────────────────────────────────────────────────────────────────────────

class RealSO101Robot:
    """Thin wrapper around your physical robot driver.

    Replace each method body with calls to your actual robot SDK
    (e.g. lerobot, ROS, dynamixel SDK, or whatever you use).

    The interface is intentionally minimal:
      - reset()          → initial obs  (np.ndarray, shape (OBS_DIM,))
      - step(action)     → (next_obs, reward, done)
      - close()          → release hardware resources

    Observation construction
    ------------------------
    Your driver must provide:
      joint_pos   (6,)  rad   — from encoders
      joint_vel   (6,)  rad/s — from encoders or numerical diff
      gripper_pos (3,)  m     — from forward kinematics
      gripper_euler (3,) rad  — from forward kinematics (roll, pitch, yaw)
      cube_pos    (3,)  m     — from a camera/MoCap system

    Reward
    ------
    You can compute reward using the same formula as LiftCubeCartesianEnv,
    or use a simple proxy (e.g. cube_height > threshold → +1).
    """

    def __init__(self, reward_version: str = "v7", lift_height: float = 0.08):
        self.reward_version = reward_version
        self.lift_height    = lift_height
        # TODO: connect to your robot here
        # e.g. self.robot = lerobot.SO101Robot(port="/dev/ttyUSB0")
        raise NotImplementedError(
            "Fill in RealSO101Robot with your robot SDK before using --mode robot. "
            "Use --mode sim or --mode scripted to test the pipeline first."
        )

    def reset(self) -> np.ndarray:
        """Move robot to home pose, place cube, return initial observation."""
        # TODO:
        # self.robot.go_home()
        # time.sleep(1.0)
        # return self._get_obs()
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Send delta-EE action, return (next_obs, reward, done)."""
        # action: (4,) float32  [dx, dy, dz, gripper]
        # TODO:
        # self.robot.send_cartesian_delta(action[:3] * ACTION_SCALE)
        # self.robot.set_gripper(action[3])
        # time.sleep(1.0 / CONTROL_HZ)
        # obs = self._get_obs()
        # reward = self._compute_reward(obs)
        # done = reward > SUCCESS_THRESHOLD
        # return obs, reward, done
        raise NotImplementedError

    def close(self) -> None:
        # TODO: self.robot.disconnect()
        pass

    def _get_obs(self) -> np.ndarray:
        """Read sensor state and pack into (OBS_DIM,) array."""
        # TODO: read from your driver
        # joint_pos   = self.robot.get_joint_positions()   # (6,)
        # joint_vel   = self.robot.get_joint_velocities()  # (6,)
        # gripper_pos = self.robot.get_ee_position()       # (3,)
        # gripper_euler = self.robot.get_ee_euler()        # (3,)
        # cube_pos    = self.camera.get_cube_position()    # (3,)
        # return np.concatenate([joint_pos, joint_vel, gripper_pos, gripper_euler, cube_pos])
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo sim backend (used for --mode sim and --mode scripted)
# ─────────────────────────────────────────────────────────────────────────────

class SimBackend:
    """Wraps LiftCubeCartesianEnv so it has the same 3-return step() interface."""

    def __init__(self, reward_version: str = "v7"):
        sys.path.insert(0, str(Path(__file__).parent))
        from src.envs.lift_cube import LiftCubeCartesianEnv
        self.env = LiftCubeCartesianEnv(
            render_mode=None,
            reward_type="dense",
            reward_version=reward_version,
        )

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs.astype(np.float32), float(reward), bool(terminated or truncated)

    def sample_action(self) -> np.ndarray:
        return self.env.action_space.sample()

    def close(self) -> None:
        self.env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Scripted grasp policy (re-used from scripts/collect_scripted_grasps.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def scripted_policy(step_in_ep: int, rng: np.random.Generator) -> np.ndarray:
    """Simple phase-based scripted policy for lift task."""
    noise = 0.15
    if step_in_ep < 25:          # descend with open gripper
        return np.array([
            rng.uniform(-1, 1) * noise,
            rng.uniform(-1, 1) * noise,
            -0.6 + rng.uniform(-1, 1) * noise,
            0.8,
        ], dtype=np.float32)
    elif step_in_ep < 45:        # close gripper
        return np.array([
            rng.uniform(-1, 1) * noise * 0.3,
            rng.uniform(-1, 1) * noise * 0.3,
            rng.uniform(-1, 1) * noise * 0.3,
            -1.0,
        ], dtype=np.float32)
    else:                        # lift
        return np.array([
            rng.uniform(-1, 1) * noise * 0.2,
            rng.uniform(-1, 1) * noise * 0.2,
            0.7 + rng.uniform(-1, 1) * noise * 0.2,
            -1.0,
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Core collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_episodes(
    backend,
    n_episodes: int,
    max_steps_per_ep: int,
    mode: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Run episodes and accumulate transition data.

    Returns a dict ready for np.savez():
        observations      (N, OBS_DIM)  float32
        next_observations (N, OBS_DIM)  float32
        actions           (N, ACT_DIM)  float32
        rewards           (N,)          float32
        dones             (N,)          bool
    """
    rng = np.random.default_rng(seed)

    obs_list      = []
    next_obs_list = []
    act_list      = []
    rew_list      = []
    done_list     = []

    for ep in range(n_episodes):
        obs = backend.reset()
        ep_steps = 0

        while ep_steps < max_steps_per_ep:
            # ── Choose action ─────────────────────────────────────────────
            if mode == "scripted":
                action = scripted_policy(ep_steps, rng)
            elif mode in ("sim", "robot"):
                # Random action — replace with your policy if desired
                action = backend.sample_action() if hasattr(backend, "sample_action") \
                         else (rng.uniform(-1, 1, ACT_DIM).astype(np.float32))
            else:  # teleop
                action = _read_teleop_action()   # blocks until input received

            action = np.clip(action, -1.0, 1.0).astype(np.float32)

            # ── Environment step ──────────────────────────────────────────
            next_obs, reward, done = backend.step(action)
            next_obs = np.clip(next_obs, -1e6, 1e6).astype(np.float32)

            obs_list.append(obs.copy())
            next_obs_list.append(next_obs.copy())
            act_list.append(action.copy())
            rew_list.append(reward)
            done_list.append(done)

            obs = next_obs
            ep_steps += 1

            if done:
                break

        n_total = len(obs_list)
        print(f"  Episode {ep+1:4d}/{n_episodes}  steps={ep_steps:3d}  "
              f"total_transitions={n_total}")

    return {
        "observations":      np.array(obs_list,      dtype=np.float32),
        "next_observations": np.array(next_obs_list, dtype=np.float32),
        "actions":           np.array(act_list,      dtype=np.float32),
        "rewards":           np.array(rew_list,      dtype=np.float32),
        "dones":             np.array(done_list,     dtype=bool),
    }


def _read_teleop_action() -> np.ndarray:
    """Minimal keyboard teleoperation (WASD + QE for Z + G for gripper)."""
    print("  Action> dx dy dz gripper  (e.g.  0.5 0 0 -1): ", end="", flush=True)
    raw = input().strip().split()
    try:
        vals = [float(x) for x in raw]
        if len(vals) == 4:
            return np.array(vals, dtype=np.float32)
    except ValueError:
        pass
    print("  Bad input — using zero action.")
    return np.zeros(4, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Save / verify helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_npz(data: Dict[str, np.ndarray], path: Path) -> None:
    """Save transition dict as compressed .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    n = len(data["rewards"])
    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved {n} transitions → {path}  ({size_mb:.1f} MB)")


def verify_npz(path: Path) -> None:
    """Load and validate a saved .npz data file."""
    print(f"\nVerifying {path} …")
    data = np.load(path, allow_pickle=True)

    # Check required keys
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"  ERROR: missing keys: {missing}")
        return

    obs      = data["observations"]
    next_obs = data["next_observations"]
    actions  = data["actions"]
    rewards  = data["rewards"]
    dones    = data["dones"]

    N = obs.shape[0]
    print(f"  transitions : {N}")
    print(f"  obs shape   : {obs.shape}  (expected ({N}, {OBS_DIM}))")
    print(f"  act shape   : {actions.shape}  (expected ({N}, {ACT_DIM}))")
    print(f"  rewards     : min={rewards.min():.3f}  max={rewards.max():.3f}  "
          f"mean={rewards.mean():.3f}")
    print(f"  done rate   : {dones.mean()*100:.1f}%")
    print(f"  dtypes      : obs={obs.dtype}  act={actions.dtype}  "
          f"rew={rewards.dtype}  done={dones.dtype}")

    # Shape checks
    ok = True
    if obs.shape != (N, OBS_DIM):
        print(f"  WARN: obs shape {obs.shape} != ({N}, {OBS_DIM})")
        ok = False
    if next_obs.shape != (N, OBS_DIM):
        print(f"  WARN: next_obs shape {next_obs.shape} != ({N}, {OBS_DIM})")
        ok = False
    if actions.shape != (N, ACT_DIM):
        print(f"  WARN: actions shape {actions.shape} != ({N}, {ACT_DIM})")
        ok = False
    if rewards.shape not in ((N,), (N, 1)):
        print(f"  WARN: rewards shape {rewards.shape}")
        ok = False

    # NaN / Inf check
    for key in ("observations", "next_observations", "actions", "rewards"):
        arr = data[key]
        if not np.isfinite(arr).all():
            print(f"  WARN: {key} contains NaN or Inf!")
            ok = False

    if ok:
        print("  All checks passed.")
    else:
        print("  Some checks failed — review warnings above.")

    # Observation breakdown
    print("\n  Observation breakdown (means across all transitions):")
    obs_mean = obs.mean(0)
    labels = (
        [f"joint_pos[{i}]"   for i in range(6)] +
        [f"joint_vel[{i}]"   for i in range(6)] +
        [f"gripper_pos[{i}]" for i in range(3)] +
        [f"gripper_eul[{i}]" for i in range(3)] +
        [f"cube_pos[{i}]"    for i in range(3)]
    )
    for i, (lbl, val) in enumerate(zip(labels, obs_mean)):
        print(f"    [{i:2d}] {lbl:<18}  mean={val:+.4f}  "
              f"std={obs[:, i].std():.4f}  "
              f"range=[{obs[:, i].min():+.3f}, {obs[:, i].max():+.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect real-world data for train_lift_MB.py")
    parser.add_argument("--mode",       type=str, default="sim",
                        choices=["sim", "scripted", "robot", "teleop"],
                        help="Data source: sim | scripted | robot | teleop")
    parser.add_argument("--episodes",   type=int, default=20,
                        help="Number of episodes to collect")
    parser.add_argument("--max-steps",  type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--out",        type=str, default="data/real_trajectories.npz",
                        help="Output .npz file path")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--verify",     type=str, default=None,
                        metavar="PATH",
                        help="Just verify an existing .npz file and exit")
    args = parser.parse_args()

    # ── Verify mode ───────────────────────────────────────────────────────
    if args.verify:
        verify_npz(Path(args.verify))
        return

    # ── Build backend ─────────────────────────────────────────────────────
    print(f"Mode: {args.mode}")
    if args.mode in ("sim", "scripted"):
        backend = SimBackend(reward_version="v7")
        print("Using MuJoCo simulator as backend.")
    elif args.mode == "robot":
        backend = RealSO101Robot()
        print("Using real SO-101 robot.")
    elif args.mode == "teleop":
        backend = SimBackend(reward_version="v7")  # swap for RealSO101Robot() for real teleop
        print("Teleoperation mode. Backend: sim (swap to RealSO101Robot for real robot).")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ── Collect ───────────────────────────────────────────────────────────
    print(f"\nCollecting {args.episodes} episodes × {args.max_steps} max steps …\n")
    try:
        data = collect_episodes(
            backend=backend,
            n_episodes=args.episodes,
            max_steps_per_ep=args.max_steps,
            mode=args.mode,
            seed=args.seed,
        )
    finally:
        backend.close()

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    save_npz(data, out_path)

    # ── Auto-verify ───────────────────────────────────────────────────────
    verify_npz(out_path)


if __name__ == "__main__":
    main()
