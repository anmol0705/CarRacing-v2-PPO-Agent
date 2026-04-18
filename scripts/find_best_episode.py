"""Run N episodes per checkpoint, score on reward + smoothness, find the single best.

Does NOT store frames — only scores. The winner is re-recorded by record_showcase.py.
Uses get_greedy_action() for deterministic (mean) policy — no sampling wobble.
"""

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_utils import Float64Action, NormalizeObservation
from src.model import ActorCritic

CHECKPOINTS = [
    "checkpoints/model_best.pt",
    "checkpoints/model_5001216.pt",
    "checkpoints/model_4751360.pt",
]
SEEDS = list(range(200))
MAX_STEPS = 1200
DEVICE = torch.device("cpu")


def make_env(seed: int) -> tuple[gym.Env, gym.Env]:
    """Create wrapped env (for model) and raw env (for reward verification)."""
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    env = Float64Action(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


def run_episode(model: ActorCritic, seed: int) -> dict:
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steering_trace = []
    step = 0

    try:
        for step in range(MAX_STEPS):
            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = model.get_greedy_action(obs_t)
            action_np = action.cpu().numpy()[0]
            steering_trace.append(float(action_np[0]))
            obs, reward, terminated, truncated, _ = env.step(action_np)
            total_reward += reward
            if terminated or truncated:
                break
    finally:
        env.close()

    steps = step + 1
    if len(steering_trace) > 1:
        diffs = np.abs(np.diff(steering_trace))
        smoothness = 1.0 / (1.0 + float(diffs.mean()) * 10)
    else:
        smoothness = 0.0

    completion = min(1.0, steps / 900)
    combined = total_reward * 0.7 + smoothness * 200 + completion * 50

    return {
        "reward": total_reward,
        "steps": steps,
        "smoothness": smoothness,
        "completion": completion,
        "combined_score": combined,
    }


results = []
best_score = -9999.0
best_ckpt = ""
best_seed = 0

for ckpt_path in CHECKPOINTS:
    if not Path(ckpt_path).exists():
        print(f"Skipping {ckpt_path} — not found")
        continue

    print(f"\n{'=' * 60}")
    print(f"Checkpoint: {ckpt_path}")
    model = ActorCritic().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    for seed in SEEDS:
        print(f"  Seed {seed:2d} ", end="", flush=True)
        try:
            r = run_episode(model, seed)
            tag = ""
            if r["combined_score"] > best_score:
                best_score = r["combined_score"]
                best_ckpt = ckpt_path
                best_seed = seed
                tag = "  *** NEW BEST ***"
            print(
                f"reward={r['reward']:7.1f}  smooth={r['smoothness']:.3f}  "
                f"steps={r['steps']:4d}  score={r['combined_score']:7.1f}{tag}"
            )
            results.append(
                {"checkpoint": ckpt_path, "seed": seed, **r}
            )
        except Exception as e:
            print(f"  ERROR: {e}")

df = pd.DataFrame(results).sort_values("combined_score", ascending=False)
df.to_csv("assets/episode_leaderboard.csv", index=False)
print(f"\nLeaderboard saved ({len(df)} episodes)")
print(df.head(10).to_string(index=False))

best_info = {
    "checkpoint": best_ckpt,
    "seed": best_seed,
    "reward": float(df.iloc[0]["reward"]),
    "steps": int(df.iloc[0]["steps"]),
    "smoothness": float(df.iloc[0]["smoothness"]),
    "combined_score": float(best_score),
}
with open("assets/best_episode_info.json", "w") as f:
    json.dump(best_info, f, indent=2)

print(f"\nBEST EPISODE:")
print(json.dumps(best_info, indent=2))
