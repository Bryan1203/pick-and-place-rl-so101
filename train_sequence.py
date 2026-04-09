"""Train a lift policy with MLP, LSTM, or Transformer policy on state observations.

Model types
-----------
mlp         Standard MlpPolicy (no history). Baseline.
lstm        GRU feature extractor over a fixed-length observation history.
transformer Transformer encoder over a fixed-length observation history.

Example usage
-------------
# Baseline MLP
python train_sequence.py --model-type mlp --config configs/state_based/sequence_mlp.yaml

# LSTM (GRU)
python train_sequence.py --model-type lstm --config configs/state_based/sequence_lstm.yaml

# Transformer
python train_sequence.py --model-type transformer --config configs/state_based/sequence_transformer.yaml
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.envs.lift_cube_cartesian import LiftCubeCartesianEnv
from src.envs.wrappers.history_wrapper import HistoryWrapper
from src.models.sequence_features import GRUFeatureExtractor, TransformerFeatureExtractor


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict, model_type: str, history_len: int):
    env = LiftCubeCartesianEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
    )
    if model_type in ("lstm", "transformer"):
        env = HistoryWrapper(env, history_len=history_len)
    return env


def build_policy_kwargs(model_type: str, model_cfg: dict) -> dict:
    """Build SAC policy_kwargs for the chosen model type."""
    net_arch = model_cfg.get("net_arch", [256, 256])

    if model_type == "mlp":
        return dict(net_arch=net_arch)

    history_len = model_cfg["history_len"]

    if model_type == "lstm":
        return dict(
            features_extractor_class=GRUFeatureExtractor,
            features_extractor_kwargs=dict(
                history_len=history_len,
                hidden_size=model_cfg.get("hidden_size", 128),
                num_layers=model_cfg.get("num_layers", 1),
            ),
            net_arch=net_arch,
        )

    if model_type == "transformer":
        return dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                history_len=history_len,
                d_model=model_cfg.get("d_model", 64),
                nhead=model_cfg.get("nhead", 4),
                num_layers=model_cfg.get("num_layers", 2),
                dim_feedforward=model_cfg.get("dim_feedforward", 256),
                dropout=model_cfg.get("dropout", 0.0),
            ),
            net_arch=net_arch,
        )

    raise ValueError(f"Unknown model_type: {model_type!r}. Choose mlp / lstm / transformer.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["mlp", "lstm", "transformer"],
        help="Policy architecture to use.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Defaults to configs/state_based/sequence_<model_type>.yaml",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = f"configs/state_based/sequence_{args.model_type}.yaml"

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]
    model_cfg = config.get("model", {})

    history_len = model_cfg.get("history_len", 16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_cfg['name']}_{args.model_type}_{timestamp}"
    output_dir = Path(exp_cfg["base_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # Save model type alongside config for easy reference
    (output_dir / "model_type.txt").write_text(args.model_type)

    # Build environments
    env = DummyVecEnv([lambda: make_env(env_cfg, args.model_type, history_len)])
    env = VecNormalize(
        env,
        norm_obs=env_cfg.get("normalize_obs", True),
        norm_reward=env_cfg.get("normalize_reward", True),
    )

    eval_env = DummyVecEnv([lambda: make_env(env_cfg, args.model_type, history_len)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=env_cfg.get("normalize_obs", True),
        norm_reward=False,
        training=False,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"sac_{args.model_type}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )
    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
    )

    # Build policy kwargs and model
    policy_kwargs = build_policy_kwargs(args.model_type, model_cfg)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        learning_starts=sac_cfg["learning_starts"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        train_freq=sac_cfg["train_freq"],
        gradient_steps=sac_cfg["gradient_steps"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=train_cfg["seed"],
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    print(f"\nModel type  : {args.model_type}")
    print(f"History len : {history_len if args.model_type != 'mlp' else 'N/A (no history)'}")
    print(f"Timesteps   : {train_cfg['timesteps']}")
    print(f"Output dir  : {output_dir}")
    print(f"Policy net  : {model.policy}")

    model.learn(
        total_timesteps=train_cfg["timesteps"],
        callback=[checkpoint_callback, eval_callback, plot_callback],
        progress_bar=True,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")
    print(f"\nTraining complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
