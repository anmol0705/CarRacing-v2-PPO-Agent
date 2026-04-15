import subprocess
import sys
import os
import glob

env = os.environ.copy()
env["WANDB_MODE"] = "offline"

# Smoke test: 2000 steps, 2 envs, frequent checkpoints
result = subprocess.run(
    [
        sys.executable,
        "scripts/train.py",
        "training.total_timesteps=2000",
        "training.rollout_steps=64",
        "env.n_envs=2",
        "logging.eval_interval=1000",
        "logging.checkpoint_interval=1000",
        "logging.gif_interval=2000",
        "logging.wandb_project=carracing-ppo-test",
    ],
    capture_output=True,
    text=True,
    timeout=300,
    env=env,
)

print(result.stdout[-2000:])  # last 2000 chars
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise AssertionError("train.py smoke test failed")

checkpoints = glob.glob("checkpoints/*.pt")
assert len(checkpoints) > 0, "No checkpoints saved"
print(f"Checkpoints saved: {len(checkpoints)}")
print("PHASE 7 PASSED")
