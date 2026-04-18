"""Record one short episode per checkpoint to show learning progression.

Produces assets/progression_clean.gif with checkpoint label overlay.
"""

import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_utils import Float64Action, NormalizeObservation
from src.model import ActorCritic

CHECKPOINTS = [
    ("checkpoints/model_501760.pt", "Step 500K \u2014 early chaos"),
    ("checkpoints/model_751616.pt", "Step 750K \u2014 learning to steer"),
    ("checkpoints/model_best.pt", "Step 4.9M \u2014 best agent (reward 812)"),
    ("checkpoints/model_final.pt", "Step 5M \u2014 final model"),
]

MAX_STEPS_PER = 400
FPS = 24
BORDER = 30

device = torch.device("cpu")
all_frames = []


def make_wrapped_env(seed: int) -> gym.Env:
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    env = Float64Action(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


for ckpt_path, label in CHECKPOINTS:
    print(f"\nRecording: {label}")

    if not Path(ckpt_path).exists():
        print(f"  Skipping \u2014 {ckpt_path} not found")
        continue

    model = ActorCritic().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    env = make_wrapped_env(seed=42)
    obs, _ = env.reset(seed=42)

    raw_env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    raw_env = Float64Action(raw_env)
    raw_obs, _ = raw_env.reset(seed=42)

    total_reward = 0.0
    clip_frames = []

    for step in range(MAX_STEPS_PER):
        obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0)
        with torch.no_grad():
            action = model.get_greedy_action(obs_t)
        action_np = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, _ = env.step(action_np)
        raw_obs, _, _, _, _ = raw_env.step(action_np)
        total_reward += reward
        if terminated or truncated:
            break

        frame = raw_env.render()
        clip_frames.append(frame)

    env.close()
    raw_env.close()
    print(f"  Reward: {total_reward:.1f}, Frames: {len(clip_frames)}")

    for frame in clip_frames:
        img = Image.fromarray(frame)
        w, h = img.size
        new_img = Image.new("RGB", (w, h + BORDER), (15, 15, 20))
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        draw.text(
            (6, h + 7),
            f"{label}  |  Reward: {total_reward:.0f}",
            fill=(200, 200, 200),
        )
        all_frames.append(np.array(new_img))

    if all_frames:
        pause_frame = all_frames[-1].copy()
        all_frames.extend([pause_frame] * 12)

print(f"\nTotal frames: {len(all_frames)}")
imageio.mimsave("assets/progression_clean.gif", all_frames, fps=FPS, loop=0)
print("Saved: assets/progression_clean.gif")
