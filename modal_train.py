"""Run pick-cube SAC training on Modal with GPU.

Calls train_v2.py (PickCubeEnv, 6-dim joint-space actions, W&B + full callbacks).

Usage:
    # Run with default config
    uv run modal run modal_train.py

    # Override timesteps
    uv run modal run modal_train.py --timesteps 3000000

    # Run detached (keeps running after you close terminal)
    uv run modal run --detach modal_train.py

    # Disable W&B (useful for quick tests)
    uv run modal run modal_train.py --no-wandb

    # Set a custom W&B run name
    uv run modal run modal_train.py --wandb-name pick_cube

    # Resume from a previous run
    uv run modal run --detach modal_train.py --resume runs/pick_cube/20260408_120000

    # Transfer learning from a pretrained checkpoint
    uv run modal run --detach modal_train.py --pretrained runs/pick_cube/<ts>/final_model.zip
    tmp:
    uv run modal run --detach modal_train.py --wandb-name pick_cube_yu_reward_040901
"""

from pathlib import Path
from collections import deque
import time

import modal

# ---------------------------------------------------------------------------
# Modal image: CUDA 12.8, Python 3.10, all deps + repo source
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
    # Bake the repo into the image
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

app = modal.App("pick-101-pick-cube", image=image)

# Persistent volume for checkpoints and runs (survives across invocations)
vol = modal.Volume.from_name("pick-101-runs", create_if_missing=True)


@app.function(
    gpu="T4",                  # Sufficient for state-based SAC; more VRAM than A1
    timeout=20 * 3600,           # 20 hour max for long pick-place runs
    volumes={"/runs": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],  # WANDB_API_KEY
    memory=64 * 1024,           # 64 GB RAM preserve
)
def train(
    config: str = "configs/pick_cube.yaml",
    extra_args: list[str] | None = None,
):
    import os
    import shutil
    import subprocess

    os.chdir("/repo")

    def sync_runs_to_volume():
        local_runs = "/repo/runs"
        if not os.path.exists(local_runs):
            return

        for item in os.listdir(local_runs):
            src = os.path.join(local_runs, item)
            dst = os.path.join("/runs", item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        vol.commit()
        print("\nRuns synced to modal volume 'pick-101-runs'")

    cmd = [
        "python", "train_v2.py",
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

    recent_output = deque(maxlen=200)
    sync_interval_s = 30 * 60
    last_sync = time.monotonic()

    # Stream output live and periodically sync checkpoints to the volume.
    while True:
        line = proc.stdout.readline()
        if line:
            recent_output.append(line.rstrip("\n"))
            print(line, end="")
        elif proc.poll() is not None:
            break
        else:
            time.sleep(1.0)

        now = time.monotonic()
        if now - last_sync >= sync_interval_s:
            sync_runs_to_volume()
            last_sync = now

    proc.wait()

    # Final sync to persistent volume
    sync_runs_to_volume()

    if proc.returncode != 0:
        tail = "\n".join(recent_output).strip()
        message = f"Training failed with exit code {proc.returncode}"
        if tail:
            message += f"\nLast output lines:\n{tail}"
        raise RuntimeError(message)

    print("\nTraining complete!")


# ---------------------------------------------------------------------------
# Entry point: `uv run modal run modal_train.py`
# This relies on the project-managed `modal` dependency, not a global install.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = "configs/pick_cube.yaml",
    timesteps: int = 0,
    no_wandb: bool = False,
    wandb_name: str = "",
    resume: str = "",
    pretrained: str = "",
):
    extra_args = []
    if timesteps > 0:
        extra_args.extend(["--timesteps", str(timesteps)])
    if no_wandb:
        extra_args.append("--no-wandb")
    if wandb_name:
        extra_args.extend(["--wandb-name", wandb_name])
    if resume:
        extra_args.extend(["--resume", resume])
    if pretrained:
        extra_args.extend(["--pretrained", pretrained])

    train.remote(
        config=config,
        extra_args=extra_args,
    )
