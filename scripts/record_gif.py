"""Record progression GIFs from training checkpoints."""

import argparse
import glob
import sys
import os
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env_utils import make_env
from src.model import ActorCritic


def record_episode(
    model: ActorCritic | None,
    device: torch.device,
    max_steps: int = 1000,
) -> list[np.ndarray]:
    """Record one episode, returning list of RGB frames.

    If model is None, uses random actions (for demo mode).
    """
    env_fn = make_env(seed=42, render_mode="rgb_array")
    env = env_fn()
    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    done = False
    step = 0

    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if model is not None:
            obs_t = torch.as_tensor(
                np.array(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = model.get_greedy_action(obs_t)
            action_np = action.cpu().numpy().squeeze(0)
        else:
            action_np = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        step += 1

    env.close()
    return frames


def select_checkpoints(checkpoint_dir: str, percentiles: list[float]) -> list[str]:
    """Select checkpoints at given percentiles of training progress."""
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.pt")))
    if not ckpts:
        return []

    selected = []
    n = len(ckpts)
    for pct in percentiles:
        idx = min(int(pct * (n - 1)), n - 1)
        selected.append(ckpts[idx])
    return selected


def make_progression_gif(
    all_frames: list[list[np.ndarray]], output_path: str, fps: int = 30
) -> None:
    """Hstack frames from multiple episodes side-by-side and save as GIF."""
    # Find minimum episode length
    min_len = min(len(f) for f in all_frames)
    min_len = min(min_len, 200)  # Cap at 200 frames to keep GIF manageable

    combined_frames = []
    for t in range(min_len):
        row = [ep_frames[t] for ep_frames in all_frames]
        # Resize all to same height
        target_h = min(f.shape[0] for f in row)
        resized = []
        for f in row:
            if f.shape[0] != target_h:
                scale = target_h / f.shape[0]
                new_w = int(f.shape[1] * scale)
                from PIL import Image

                img = Image.fromarray(f).resize((new_w, target_h))
                f = np.array(img)
            resized.append(f)
        combined = np.hstack(resized)
        combined_frames.append(combined)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, combined_frames, fps=fps, loop=0)
    print(f"Saved {output_path} ({len(combined_frames)} frames)")


def main() -> None:
    """Main entry point for GIF recording."""
    parser = argparse.ArgumentParser(description="Record training progression GIFs")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: record random agent episodes",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory containing .pt checkpoints",
    )
    parser.add_argument(
        "--output-dir", default="assets", help="Output directory for GIFs"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("Demo mode: recording 5 random-policy episodes")
        all_frames = []
        for i in range(5):
            pct = i * 25
            print(f"  Recording episode {i+1}/5 ({pct}%)")
            frames = record_episode(model=None, device=device, max_steps=200)
            all_frames.append(frames)

            # Save individual GIF
            individual_path = os.path.join(args.output_dir, f"ep_{pct}pct.gif")
            imageio.mimsave(individual_path, frames[:200], fps=30, loop=0)

        # Progression GIF
        make_progression_gif(
            all_frames, os.path.join(args.output_dir, "progression.gif")
        )
        return

    # Real mode: load checkpoints at 0%, 25%, 50%, 75%, 100%
    percentiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    ckpts = select_checkpoints(args.checkpoint_dir, percentiles)
    if not ckpts:
        print("No checkpoints found. Use --demo for placeholder GIFs.")
        return

    all_frames = []
    for i, ckpt_path in enumerate(ckpts):
        pct = int(percentiles[i] * 100)
        print(f"Loading {ckpt_path} ({pct}%)")
        model = ActorCritic().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        frames = record_episode(model=model, device=device)
        all_frames.append(frames)

        individual_path = os.path.join(args.output_dir, f"ep_{pct}pct.gif")
        imageio.mimsave(individual_path, frames[:300], fps=30, loop=0)

    make_progression_gif(
        all_frames, os.path.join(args.output_dir, "progression.gif")
    )


if __name__ == "__main__":
    main()
