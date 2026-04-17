"""Detailed evaluation with per-episode breakdown."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium as gym

from src.env_utils import make_env
from src.model import ActorCritic


def eval_detailed(model, device, n_episodes=50, seed=9999):
    """Run detailed evaluation returning per-episode rewards."""
    model.eval()
    rewards = []

    for ep in range(n_episodes):
        env_fn = make_env(seed=seed + ep, render_mode=None)
        env = env_fn()
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(
                np.array(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = model.get_greedy_action(obs_t)
            action_np = action.cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += float(reward)
            done = terminated or truncated

        rewards.append(ep_reward)
        env.close()

    return rewards


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    ckpt = torch.load("checkpoints/model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best model from step {ckpt['global_step']}")

    rewards = eval_detailed(model, device, n_episodes=50)
    rewards = np.array(rewards)

    print(f"\n{'='*50}")
    print(f"50-episode evaluation results:")
    print(f"  Mean:   {rewards.mean():.1f}")
    print(f"  Std:    {rewards.std():.1f}")
    print(f"  Median: {np.median(rewards):.1f}")
    print(f"  Min:    {rewards.min():.1f}")
    print(f"  Max:    {rewards.max():.1f}")
    print(f"  >700:   {(rewards > 700).sum()}/50 ({(rewards > 700).mean()*100:.0f}%)")
    print(f"  >500:   {(rewards > 500).sum()}/50 ({(rewards > 500).mean()*100:.0f}%)")
    print(f"  >300:   {(rewards > 300).sum()}/50 ({(rewards > 300).mean()*100:.0f}%)")
    print(f"  <0:     {(rewards < 0).sum()}/50")
    print(f"\nPer-episode rewards (sorted):")
    for i, r in enumerate(sorted(rewards)):
        marker = " ***" if r > 700 else ""
        print(f"  Ep {i+1:2d}: {r:8.1f}{marker}")


if __name__ == "__main__":
    main()
