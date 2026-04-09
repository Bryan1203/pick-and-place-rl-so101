"""Sequence model feature extractors for SB3 policies.

Both extractors expect a flat observation of shape (history_len * obs_dim,)
produced by HistoryWrapper. Internally they reshape to
(batch, history_len, obs_dim) before processing.

Usage with SAC:
    from src.models.sequence_features import GRUFeatureExtractor
    from src.envs.wrappers.history_wrapper import HistoryWrapper

    env = HistoryWrapper(base_env, history_len=16)
    policy_kwargs = dict(
        features_extractor_class=GRUFeatureExtractor,
        features_extractor_kwargs=dict(history_len=16, hidden_size=128),
        net_arch=[256, 256],
    )
    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs)
"""
import math

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """GRU-based feature extractor.

    Reshapes flat history into a sequence and passes it through a GRU.
    The last hidden state is returned as the feature vector.

    Args:
        observation_space: Flat Box space of shape (history_len * obs_dim,).
        history_len: Number of timesteps in the observation history.
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 16,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super().__init__(observation_space, features_dim=hidden_size)
        self.history_len = history_len
        self.obs_dim = observation_space.shape[0] // history_len
        self.gru = nn.GRU(
            input_size=self.obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, history_len * obs_dim)
        batch = obs.shape[0]
        x = obs.view(batch, self.history_len, self.obs_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        return h_n[-1]         # (batch, hidden_size)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Transformer encoder feature extractor.

    Projects each timestep into d_model space, adds learned positional
    embeddings, runs through a TransformerEncoder, and returns the last
    token's representation.

    Args:
        observation_space: Flat Box space of shape (history_len * obs_dim,).
        history_len: Number of timesteps in the observation history.
        d_model: Transformer model dimension.
        nhead: Number of attention heads (must divide d_model).
        num_layers: Number of TransformerEncoderLayer blocks.
        dim_feedforward: FFN hidden size inside each encoder layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        history_len: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__(observation_space, features_dim=d_model)
        self.history_len = history_len
        self.obs_dim = observation_space.shape[0] // history_len

        self.input_proj = nn.Linear(self.obs_dim, d_model)
        self.pos_embedding = nn.Embedding(history_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, history_len * obs_dim)
        batch = obs.shape[0]
        x = obs.view(batch, self.history_len, self.obs_dim)

        x = self.input_proj(x)  # (batch, history_len, d_model)

        positions = torch.arange(self.history_len, device=obs.device)
        x = x + self.pos_embedding(positions)  # broadcast over batch

        x = self.transformer(x)   # (batch, history_len, d_model)
        return x[:, -1]           # (batch, d_model) — most recent token
