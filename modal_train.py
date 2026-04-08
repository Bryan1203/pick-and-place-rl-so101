"""Run SAC training on Modal with GPU.

Usage:
    # Run with default config (pick_place_container v21)
    uv run modal run modal_train.py

    # Run with custom config
    uv run modal run modal_train.py --config configs/state_based/pick_place_container.yaml

    # Run with extra args
    uv run modal run modal_train.py --timesteps 1000000

    # Run detached (keeps running even if you close terminal)
    uv run modal run --detach modal_train.py
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image: CUDA 12.8, Python 3.12, all deps + repo source
# ---------------------------------------------------------------------------
repo_root = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1-mesa-glx",       # MuJoCo rendering
        "libosmesa6-dev",        # OSMesa for headless rendering
        "libglew-dev",
        "libglib2.0-0",
        "ffmpeg",                # video encoding
    )
    .pip_install(
        "torch>=2.8.0",
        "torchvision>=0.23.0",
        "torchaudio>=2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "mujoco>=3.0.0",
        "gymnasium @ git+https://github.com/stepjam/Gymnasium.git@0.29.2",
        "stable-baselines3>=2.0.0",
        "imageio[ffmpeg]>=2.30.0",
        "moviepy>=1.0.3",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.3",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "numpy<2.0.0",
        "natsort",
        "termcolor",
        "opencv-python-headless",
        "timm",
        "scipy",
        "einops",
        "diffusers==0.29.0",
        "hydra-joblib-launcher",
    )
    # Bake the repo into the image (replaces modal.Mount)
    .add_local_dir(
        repo_root,
        remote_path="/repo",
        copy=True,
        ignore=modal.FilePatternMatcher(
            ".git/", "__pycache__/", ".venv/", "runs/", "*.pyc", "wandb/",
            "*.egg-info/", "MUJOCO_LOG.TXT",
        ),
    )
    # Install robobase from the bundled submodule
    .run_commands("pip install -e /repo/external/robobase")
    .env({"MUJOCO_GL": "osmesa", "PYTHONPATH": "/repo"})
)

app = modal.App("pick-101-train", image=image)

# Persistent volume for checkpoints and runs (survives across invocations)
vol = modal.Volume.from_name("pick-101-runs", create_if_missing=True)


@app.function(
    gpu="L4",                    # Fast and cheap ($0.80/hr), plenty for state-based SAC
    timeout=6 * 3600,            # 6 hour max
    volumes={"/runs": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],  # WANDB_API_KEY
)
def train(
    config: str = "configs/state_based/pick_place_container.yaml",
    extra_args: list[str] | None = None,
):
    import os
    import shutil
    import subprocess

    os.chdir("/repo")

    # Build command
    cmd = [
        "python", "train_lift.py",
        "--config", config,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output live
    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    # Copy runs to persistent volume
    local_runs = "/repo/runs"
    if os.path.exists(local_runs):
        for item in os.listdir(local_runs):
            src = os.path.join(local_runs, item)
            dst = os.path.join("/runs", item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        vol.commit()
        print(f"\nRuns synced to modal volume 'pick-101-runs'")

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entry point: `uv run modal run modal_train.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = "configs/state_based/pick_place_container.yaml",
    timesteps: int = 0,
    no_wandb: bool = False,
):
    extra_args = []
    if timesteps > 0:
        extra_args.extend(["--timesteps", str(timesteps)])
    if no_wandb:
        extra_args.append("--no-wandb")

    train.remote(
        config=config,
        extra_args=extra_args,
    )
