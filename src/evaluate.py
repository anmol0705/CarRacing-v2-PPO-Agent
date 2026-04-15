"""Policy evaluation with optional GIF recording."""

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from omegaconf import DictConfig

from src.env_utils import make_env
from src.model import ActorCritic


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    cfg: DictConfig,
    n_episodes: int = 10,
    record_gif: bool = False,
    gif_path: str = "assets/eval.gif",
) -> tuple[float, float]:
    """Evaluate the policy greedily (using mean action, no sampling).

    Args:
        model: Trained ActorCritic model (on CUDA).
        cfg: Config with env settings.
        n_episodes: Number of evaluation episodes.
        record_gif: Whether to record the first episode as a GIF.
        gif_path: Output path for the GIF file.

    Returns:
        Tuple of (mean_reward, std_reward) across episodes.
    """
    model.eval()
    device = next(model.parameters()).device

    render_mode = "rgb_array" if record_gif else None
    env_fn = make_env(seed=9999, render_mode=render_mode)
    env = env_fn()

    episode_rewards: list[float] = []
    frames: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = torch.as_tensor(
                np.array(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)

            action = model.get_greedy_action(obs_tensor)
            action_np = action.cpu().numpy().squeeze(0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)
            done = terminated or truncated

            if record_gif and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        episode_rewards.append(ep_reward)

    env.close()

    if record_gif and frames:
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=30, loop=0)

    model.train()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    return mean_reward, std_reward
