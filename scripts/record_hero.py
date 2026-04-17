"""Record a hero GIF: find a high-scoring episode and save it."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio
import numpy as np
import torch
from pathlib import Path

from src.env_utils import make_env
from src.model import ActorCritic


def record_episode(model, device, seed, max_steps=1000):
    """Record one episode, returning frames and total reward."""
    env_fn = make_env(seed=seed, render_mode="rgb_array")
    env = env_fn()
    obs, _ = env.reset()
    frames = []
    done = False
    total_reward = 0.0
    step = 0

    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_t = torch.as_tensor(
            np.array(obs), dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = model.get_greedy_action(obs_t)
        action_np = action.cpu().numpy().squeeze(0)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)
        done = terminated or truncated
        step += 1

    env.close()
    return frames, total_reward


def record_random_episode(seed, max_steps=300):
    """Record a random agent episode for comparison."""
    import gymnasium as gym
    env_fn = make_env(seed=seed, render_mode="rgb_array")
    env = env_fn()
    obs, _ = env.reset()
    frames = []
    done = False
    step = 0

    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()
    return frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    ckpt = torch.load("checkpoints/model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded best model from step {ckpt['global_step']}")

    Path("assets").mkdir(exist_ok=True)

    # Find a high-scoring episode by trying different seeds
    print("\nSearching for a high-scoring episode...")
    best_seed = None
    best_reward = -999
    best_frames = None

    for seed in range(100):
        frames, reward = record_episode(model, device, seed=seed)
        if reward > best_reward:
            best_reward = reward
            best_seed = seed
            best_frames = frames
            print(f"  Seed {seed:3d}: reward {reward:7.1f} {'<-- new best' if reward > 800 else ''}")
        if reward > 880:
            break  # Good enough

    print(f"\nBest episode: seed={best_seed}, reward={best_reward:.1f}, frames={len(best_frames)}")

    # Save hero GIF (full episode, downsampled to keep size reasonable)
    # Take every 2nd frame to reduce size
    hero_frames = best_frames[::2]
    imageio.mimsave("assets/hero.gif", hero_frames, fps=15, loop=0)
    print(f"Saved assets/hero.gif ({len(hero_frames)} frames)")

    # Save a shorter highlight clip (first 400 frames = ~13 seconds)
    highlight = best_frames[:400]
    imageio.mimsave("assets/highlight.gif", highlight, fps=30, loop=0)
    print(f"Saved assets/highlight.gif ({len(highlight)} frames)")

    # Record random agent on same seed for before/after comparison
    print(f"\nRecording random agent on seed {best_seed}...")
    random_frames = record_random_episode(seed=best_seed, max_steps=400)

    # Make before/after side-by-side
    min_len = min(len(random_frames), len(best_frames[:400]))
    min_len = min(min_len, 300)
    combined = []
    for i in range(min_len):
        left = random_frames[i]
        right = best_frames[i]
        # Add labels
        pair = np.hstack([left, right])
        combined.append(pair)

    imageio.mimsave("assets/before_after.gif", combined, fps=30, loop=0)
    print(f"Saved assets/before_after.gif ({len(combined)} frames, side-by-side)")

    print(f"\nAll hero GIFs generated!")
    print(f"  hero.gif        - Full best episode (reward {best_reward:.1f})")
    print(f"  highlight.gif   - First 13s of best episode")
    print(f"  before_after.gif - Random vs trained side-by-side")


if __name__ == "__main__":
    main()
