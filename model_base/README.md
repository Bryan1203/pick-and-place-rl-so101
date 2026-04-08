# Model-based RL for Sim-Real Fusion Training

## Collect Real-world Data

### Format

Each row is one transition (s, a, s', r, done), not a full trajectory. N transitions total across all episodes.

```
data/real_trajectories.npz
  observations      : float32  (N, 21)   — state before action
  next_observations : float32  (N, 21)   — state after action
  actions           : float32  (N,  4)   — action taken
  rewards           : float32  (N,)      — scalar reward
  dones             : bool     (N,)      — episode ended?
```

Observation layout (must match LiftCubeCartesianEnv._get_obs()):

| Index   | Field                  | Unit |
|---------|------------------------|------|
| [0:6]   | Joint positions        | rad  |
| [6:12]  | Joint velocities       | rad/s|
| [12:15] | Gripper XYZ position   | m    |
| [15:18] | Gripper Euler angles   | rad  |
| [18:21] | Cube XYZ position      | m    |

### Data Collection Script

**Step 1 — Test pipeline with MuJoCo sim (no robot needed):**

`python collect_real_data.py --mode sim --episodes 20 --out data/real_trajectories.npz`

or scripted policy (better coverage):

`python collect_real_data.py --mode scripted --episodes 100 --out data/real_trajectories.npz`

**Step 2 — Verify the file:**

`python collect_real_data.py --verify data/real_trajectories.npz`

This checks shapes, dtypes, NaN/Inf, and prints per-dimension statistics.

**Step 3 — Real robot:**
Fill in `RealSO101Robot` in `collect_real_data.py` with your actual robot SDK (the `TODO` blocks), then:

```bash
python collect_real_data.py --mode robot --episodes 50 --out data/real_trajectories.npz
```

The three things you must implement in `RealSO101Robot`:

1. **`reset()`** — move to home pose, return 21-dim obs from encoders + FK + camera
2. **`step(action)`** — send delta-XYZ + gripper, wait for control cycle, return (obs, reward, done)
3. **Cube position** — needs a camera or motion capture system; this is usually the hardest part



**Step 4 — Import Data**

```python
data = np.load("data/real_trajectories.npz")
for i in range(N):
    real_buffer.add(
        obs      = data["observations"][i:i+1],
        next_obs = data["next_observations"][i:i+1],
        action   = data["actions"][i:i+1],
        reward   = data["rewards"][i:i+1],
        done     = data["dones"][i:i+1],
        infos    = [{}],
    )
```

The buffer stores raw (unnormalized) observations. VecNormalize normalizes on-the-fly when sampling with `env=`. This is the same convention as the sim buffer, so both are treated identically by the world model training and SAC updates.

