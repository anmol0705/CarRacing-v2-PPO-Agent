"""Record all final GIF assets for the project: progression, best agent, demo clip."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.env_utils import make_env
from src.model import ActorCritic


def load_model(ckpt_path, device):
    """Load a model from checkpoint."""
    model = ActorCritic().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("global_step", 0)
    reward = ckpt.get("eval_reward", None)
    return model, step, reward


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


def add_overlay(frame, text, position="top"):
    """Add text overlay to a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    if position == "top":
        x, y = 10, 10
    else:
        x, y = 10, frame.shape[0] - 30

    # Draw shadow
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path("assets").mkdir(exist_ok=True)

    # ---- PROGRESSION GIF ----
    print("=" * 60)
    print("STEP 2: Recording progression GIF")
    print("=" * 60)

    progression_ckpts = [
        ("checkpoints/model_251904.pt", "250K steps"),
        ("checkpoints/model_1001472.pt", "1M steps"),
        ("checkpoints/model_2500608.pt", "2.5M steps"),
        ("checkpoints/model_4001792.pt", "4M steps"),
        ("checkpoints/model_best.pt", "Best (4.9M)"),
    ]

    all_prog_frames = []
    for ckpt_path, label in progression_ckpts:
        print(f"  Recording {label} from {ckpt_path}...")
        model, step, reward = load_model(ckpt_path, device)
        frames, ep_reward = record_episode(model, device, seed=42, max_steps=600)

        # Add overlay to each frame
        overlaid = [add_overlay(f, f"{label} | R={ep_reward:.0f}") for f in frames[:300]]
        all_prog_frames.append(overlaid)
        print(f"    -> {len(frames)} frames, reward={ep_reward:.1f}")

    # Stitch side by side (take min length)
    min_len = min(len(f) for f in all_prog_frames)
    min_len = min(min_len, 200)
    combined_prog = []
    for t in range(min_len):
        row = []
        for ep_frames in all_prog_frames:
            f = ep_frames[t]
            # Resize to consistent height
            img = Image.fromarray(f).resize((192, 128))
            row.append(np.array(img))
        combined_prog.append(np.hstack(row))

    imageio.mimsave("assets/progression.gif", combined_prog, fps=20, loop=0)
    print(f"  Saved assets/progression.gif ({len(combined_prog)} frames)")

    # ---- BEST AGENT GIF ----
    print()
    print("=" * 60)
    print("STEP 3: Recording best agent demo")
    print("=" * 60)

    best_model, best_step, best_eval = load_model("checkpoints/model_best.pt", device)
    print(f"  Best model: step {best_step}, eval {best_eval:.1f}")

    # Record 5 episodes, pick the best
    best_ep_frames = None
    best_ep_reward = -999
    for seed in range(5):
        frames, reward = record_episode(best_model, device, seed=seed)
        print(f"  Seed {seed}: reward={reward:.1f}, frames={len(frames)}")
        if reward > best_ep_reward:
            best_ep_reward = reward
            best_ep_frames = frames

    print(f"  Best episode: reward={best_ep_reward:.1f}")

    # Save full best agent GIF (every 2nd frame for size)
    agent_frames = [add_overlay(f, f"PPO Agent | Reward: {best_ep_reward:.0f}") for f in best_ep_frames]
    imageio.mimsave("assets/best_agent.gif", agent_frames[::2], fps=15, loop=0)
    print(f"  Saved assets/best_agent.gif ({len(agent_frames[::2])} frames)")

    # Save 10-second highlight clip (frames 50-350 = smooth mid-drive)
    start = min(50, len(agent_frames) - 1)
    end = min(start + 300, len(agent_frames))
    clip = agent_frames[start:end]
    imageio.mimsave("assets/demo_clip.gif", clip, fps=30, loop=0)
    print(f"  Saved assets/demo_clip.gif ({len(clip)} frames)")

    print()
    print("All assets generated!")


if __name__ == "__main__":
    main()
