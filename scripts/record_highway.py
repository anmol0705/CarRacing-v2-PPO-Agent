"""Record showcase GIFs for each highway-env scenario."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.highway_trainer import (
    HighwayActorCritic,
    make_highway_env,
    preprocess_obs,
)

DEVICE = torch.device("cpu")

SCENARIOS = [
    {
        "env_name": "highway-v0",
        "ckpt": "checkpoints/highway/best.pt",
        "out": "assets/highway_demo.gif",
        "label": "Highway Overtaking",
        "episodes": 5,
        "max_steps": 400,
    },
    {
        "env_name": "roundabout-v0",
        "ckpt": "checkpoints/roundabout/best.pt",
        "out": "assets/roundabout_demo.gif",
        "label": "Roundabout Navigation",
        "episodes": 5,
        "max_steps": 300,
    },
    {
        "env_name": "parking-v0",
        "ckpt": "checkpoints/parking/best.pt",
        "out": "assets/parking_demo.gif",
        "label": "Autonomous Parking",
        "episodes": 5,
        "max_steps": 400,
    },
]


def load_model(ckpt_path: str, env_name: str) -> tuple:
    """Load trained model for the given environment."""
    env = make_highway_env(env_name, render=False)
    obs, _ = env.reset()
    obs_dim = preprocess_obs(obs, env_name).shape[0]
    action_space = env.action_space
    discrete = hasattr(action_space, "n")
    action_dim = action_space.n if discrete else action_space.shape[0]
    env.close()

    model = HighwayActorCritic(obs_dim, action_dim, discrete).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded {ckpt_path} (reward {ckpt.get('reward', 0):.1f})")
    return model, discrete


def record_best_episode(scenario: dict) -> float | None:
    """Record the best episode as a GIF with label overlay."""
    env_name = scenario["env_name"]
    ckpt_path = scenario["ckpt"]

    if not Path(ckpt_path).exists():
        print(f"  Checkpoint not found: {ckpt_path} — skipping")
        return None

    model, discrete = load_model(ckpt_path, env_name)

    best_frames: list = []
    best_reward = -np.inf

    for ep in range(scenario["episodes"]):
        env = make_highway_env(env_name, seed=ep, render=True)
        obs, _ = env.reset(seed=ep)
        frames: list = []
        total_r = 0.0
        for step in range(scenario["max_steps"]):
            obs_t = torch.FloatTensor(preprocess_obs(obs, env_name)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs_t, deterministic=True)
            act = int(action.item()) if discrete else action.cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(act)
            total_r += r
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if term or trunc:
                break
        env.close()
        print(f"  Episode {ep}: reward {total_r:.2f}, {len(frames)} frames")
        if total_r > best_reward and len(frames) > 10:
            best_reward = total_r
            best_frames = frames

    if not best_frames:
        print(f"  No frames captured for {env_name}")
        return None

    processed = []
    for i, frame in enumerate(best_frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        draw.rectangle([0, h - 22, w, h], fill=(10, 10, 15))
        draw.text(
            (6, h - 16),
            f"{scenario['label']}  |  Reward: {best_reward:.1f}  |  "
            f"PPO Agent  |  Step {i + 1}/{len(best_frames)}",
            fill=(180, 180, 190),
        )
        draw.rectangle([0, 0, w, 2], fill=(59, 139, 212))
        processed.append(np.array(img))

    out_path = scenario["out"]
    imageio.mimsave(out_path, processed, fps=20, loop=0)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f}MB, {len(processed)} frames, reward {best_reward:.1f})")
    return best_reward


if __name__ == "__main__":
    results = {}
    for scenario in SCENARIOS:
        print(f"\nRecording: {scenario['label']}")
        r = record_best_episode(scenario)
        if r is not None:
            results[scenario["env_name"]] = r

    print("\nRecording complete:", results)
    with open("assets/highway_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
