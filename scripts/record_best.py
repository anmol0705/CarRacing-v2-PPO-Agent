"""Record one full lap of the best agent with text overlay.

Uses rgb_array rendering + Xvfb. Timeout: 3 minutes.
"""

import signal
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

signal.alarm(180)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_utils import Float64Action
from src.model import ActorCritic

CHECKPOINT = "checkpoints/model_best.pt"
OUT_PATH = "assets/best_lap.gif"
MAX_STEPS = 1200
FPS = 30

device = torch.device("cpu")
model = ActorCritic().to(device)
state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(state["model_state_dict"])
model.eval()


def make_wrapped_env(seed: int) -> gym.Env:
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    env = Float64Action(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    from src.env_utils import NormalizeObservation

    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


env = make_wrapped_env(seed=42)
obs, _ = env.reset(seed=42)

raw_env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
raw_env = Float64Action(raw_env)
raw_obs, _ = raw_env.reset(seed=42)

frames = []
total_reward = 0.0
cumulative_rewards = []

for step in range(MAX_STEPS):
    obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)

    with torch.no_grad():
        action = model.get_greedy_action(obs_t)

    action_np = action.cpu().numpy()[0]
    obs, reward, terminated, truncated, _ = env.step(action_np)
    raw_obs, _, _, _, _ = raw_env.step(action_np)
    total_reward += reward
    cumulative_rewards.append(total_reward)

    frame = raw_env.render()
    frames.append(frame)

    if terminated or truncated:
        break

env.close()
raw_env.close()
print(f"Episode done: {step + 1} steps, reward: {total_reward:.1f}")

processed_frames = []
for i, frame in enumerate(frames):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0, 180))
    draw.text(
        (6, 4),
        f"Reward: {cumulative_rewards[i]:.0f}  |  Step: {i + 1}/{len(frames)}  |  PPO Agent",
        fill=(255, 255, 255),
    )
    processed_frames.append(np.array(img))

imageio.mimsave(OUT_PATH, processed_frames, fps=FPS, loop=0)

with open("assets/best_lap_stats.txt", "w") as f:
    f.write(f"steps={step + 1}\nreward={total_reward:.1f}\n")

print(f"Saved to {OUT_PATH} ({len(processed_frames)} frames)")
