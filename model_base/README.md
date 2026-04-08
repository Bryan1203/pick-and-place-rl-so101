# Model-based RL for Sim-Real Fusion Training

## Collect Real-world Data

Format

```
data/real_trajectories.npz
  observations      : float32  (N, 21)   — state before action
  next_observations : float32  (N, 21)   — state after action
  actions           : float32  (N,  4)   — action taken
  rewards           : float32  (N,)      — scalar reward
  dones             : bool     (N,)      — episode ended?
```