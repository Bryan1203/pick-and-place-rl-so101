"""Train a pick-and-place policy using SAC with staged rewards.

train_v2.py extends train.py with:
  - --timesteps  : override timesteps without editing the YAML
  - --no-wandb   : disable W&B for quick local tests
  - --pretrained : load weights from a previous run (transfer learning)
  - W&B logging via WandbCallback
  - PlotLearningCurveCallback  (learning curve PNG saved to run dir)
  - RewardComponentCallback    (per-component reward breakdown)
  - Numeric checkpoint sorting  (sort by step number, not filename)
  - Resume creates a new timestamped directory  (<ts>_resumed/)
  - RESUME_INFO.txt / PRETRAINED_INFO.txt audit trail

Usage:
    python train_v2.py --config configs/pick_cube.yaml
    python train_v2.py --config configs/pick_cube.yaml --no-wandb
    python train_v2.py --config configs/pick_cube.yaml --timesteps 3000000
    python train_v2.py --config configs/pick_cube.yaml --resume runs/pick_cube/<timestamp>
    python train_v2.py --config configs/pick_cube.yaml --pretrained runs/pick_cube/<ts>/final_model.zip
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import wandb
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback

from src.callbacks.plot_callback import PlotLearningCurveCallback
from src.callbacks.reward_component_callback import RewardComponentCallback
from src.callbacks.wandb_checkpoint_callback import WandbCheckpointCallback
from src.callbacks.wandb_video_callback import WandbVideoCallback
from src.envs.pick_cube import PickCubeEnv


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    return PickCubeEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 400),
        action_scale=env_cfg.get("action_scale", 0.05),
        reward_config=env_cfg.get("reward_config", None),
    )


def make_env_fn(env_cfg: dict):
    """Create a picklable env factory for vectorized training."""
    def _init():
        return make_env(env_cfg)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pick_cube.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a previous run directory to resume from")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model .zip file (transfer learning)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps from config")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name (default: <exp_name>_<timestamp>)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / f"{timestamp}_resumed"
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_dir / "config.yaml")

    # Initialize W&B
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

    if args.resume:
        with open(output_dir / "RESUME_INFO.txt", "w") as f:
            f.write(f"Resumed from: {resume_dir}\n")
            f.write(f"Timestamp: {timestamp}\n")

    pretrained = args.pretrained or train_cfg.get("pretrained")
    vec_normalize_path = None
    if args.resume:
        vec_normalize_path = resume_dir / "vec_normalize.pkl"

    # Create training environment
    n_envs = int(train_cfg.get("n_envs", 1))
    if n_envs > 1:
        env = SubprocVecEnv([make_env_fn(env_cfg) for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env_fn(env_cfg)])

    if vec_normalize_path and vec_normalize_path.exists():
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    elif vec_normalize_path:
        raise ValueError(f"vec_normalize.pkl not found: {vec_normalize_path}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=env_cfg["normalize_reward"],
        )
        if pretrained:
            print("Using fresh VecNormalize for transfer (not loading old stats)")

    # Create eval environment (single-env, no reward normalization for fair eval)
    eval_env = DummyVecEnv([make_env_fn(env_cfg)])
    if vec_normalize_path and vec_normalize_path.exists():
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=False,
            training=False,
        )

    # Create or load SAC model
    resume_step = 0
    if args.resume:
        checkpoints = list((resume_dir / "checkpoints").glob("*.zip"))

        def get_step_number(path):
            for part in path.stem.split("_"):
                if part.isdigit():
                    return int(part)
            return 0

        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_step = get_step_number(latest_checkpoint)
            model = SAC.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint} (step {resume_step})")
            with open(output_dir / "RESUME_INFO.txt", "a") as f:
                f.write(f"Checkpoint: {latest_checkpoint}\n")
                f.write(f"Resume step: {resume_step}\n")
        else:
            raise ValueError(f"No checkpoints found in {resume_dir / 'checkpoints'}")
    elif pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise ValueError(f"Pretrained model not found: {pretrained_path}")
        model = SAC.load(pretrained_path, env=env, device=device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.num_timesteps = 0
        model._episode_num = 0
        model.replay_buffer.reset()
        print(f"Loaded pretrained weights from {pretrained_path}")
        with open(output_dir / "PRETRAINED_INFO.txt", "w") as f:
            f.write(f"Pretrained from: {pretrained_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
    else:
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
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_pick",
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
        resume_step=resume_step,
    )

    # RewardComponentCallback reads info["reward_components"] if present.
    # PickCubeEnv does not set that key yet — callback is harmless if missing.
    reward_component_callback = RewardComponentCallback(
        log_freq=train_cfg.get("eval_freq", 10000),
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback, plot_callback, reward_component_callback]
    if use_wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=train_cfg.get("save_freq", 100000),
            verbose=2,
        ))
        callbacks.append(WandbCheckpointCallback(
            checkpoint_dir=output_dir / "checkpoints",
            output_dir=output_dir,
            check_freq=train_cfg.get("save_freq", 100000),
            verbose=1,
        ))
        # WandbVideoCallback omitted: it is hardcoded to LiftCubeCartesianEnv
        # and PickCubeEnv.render() has no camera= argument.
        callbacks.append(WandbVideoCallback(
            env_cfg=env_cfg,
            vec_normalize=env,
            env_type="pick",
            log_freq=train_cfg.get("video_log_freq", 50_000),
            camera=train_cfg.get("video_camera", "default"),
            verbose=1,
        ))
    timesteps = args.timesteps if args.timesteps is not None else train_cfg["timesteps"]

    if args.resume:
        print(f"Loaded model num_timesteps: {model.num_timesteps}")
        reset_num_timesteps = False
        learn_timesteps = timesteps
        target_total = model.num_timesteps + timesteps
        print(f"\nResuming Pick training from step {model.num_timesteps}...")
        print(f"Training for {timesteps} additional timesteps (target: {target_total} total)")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        target_total = timesteps
        print(f"\nStarting Pick training for {timesteps} timesteps...")

    print(f"Action space: delta joint positions (6 dims)")
    print(f"Output directory: {output_dir}")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
