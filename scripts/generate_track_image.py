"""Render a clean overhead view of the CarRacing-v2 track as a PNG asset."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_utils import Float64Action

env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
obs, _ = env.reset(seed=42)

frames = []
for i in range(80):
    obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    if i > 20:
        frames.append(env.render())
    if terminated or truncated:
        break
env.close()

best_frame = max(
    frames,
    key=lambda f: np.sum(
        (f[:, :, 0] > 80) & (f[:, :, 0] < 130)
        & (f[:, :, 1] > 80) & (f[:, :, 1] < 130)
        & (f[:, :, 2] > 80) & (f[:, :, 2] < 130)
    ),
)

img = Image.fromarray(best_frame).resize((800, 800), Image.LANCZOS)

vignette = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
vd = ImageDraw.Draw(vignette)
for r in range(400, 0, -4):
    alpha = int(120 * (1 - r / 400) ** 2)
    vd.ellipse([400 - r, 400 - r, 400 + r, 400 + r], fill=(0, 0, 0, alpha))
img = Image.alpha_composite(img.convert("RGBA"), vignette).convert("RGB")

img.save("assets/track_layout.png", quality=95)
print(f"Saved assets/track_layout.png ({img.size})")

env2 = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
obs2, _ = env2.reset(seed=42)
frame2 = env2.render()
env2.close()
track_img = Image.fromarray(frame2).resize((600, 400), Image.LANCZOS)
track_img.save("assets/track_start.png", quality=95)
print("Saved assets/track_start.png")
