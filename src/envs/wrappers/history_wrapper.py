"""Observation history wrapper for sequence-based RL policies."""
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class HistoryWrapper(gym.ObservationWrapper):
    """Wraps an environment to return a fixed-length history of observations.

    The flattened history is fed into a sequence model (LSTM/Transformer)
    feature extractor that reshapes it back to (batch, history_len, obs_dim).

    On reset, the history is filled by repeating the initial observation.

    Args:
        env: Base environment with Box observation space.
        history_len: Number of past observations to stack. Default 16.
    """

    def __init__(self, env: gym.Env, history_len: int = 16):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), (
            "HistoryWrapper only supports Box observation spaces."
        )
        self.history_len = history_len
        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._history: deque[np.ndarray] = deque(maxlen=history_len)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(history_len * self._obs_dim,),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._history.append(obs.astype(np.float32).flatten())
        return np.concatenate(list(self._history), axis=0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        flat = obs.astype(np.float32).flatten()
        # Fill history with the initial observation (no zero-padding bias)
        for _ in range(self.history_len):
            self._history.append(flat.copy())
        return np.concatenate(list(self._history), axis=0), info
