"""Evaluate best checkpoint with more episodes for robust statistics."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.evaluate import evaluate_policy
from src.model import ActorCritic
from omegaconf import OmegaConf


def main() -> None:
    cfg = OmegaConf.load("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic().to(device)
    ckpt = torch.load("checkpoints/model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from step {ckpt['global_step']}, eval_reward={ckpt.get('eval_reward', 'N/A')}")

    # Run 50-episode eval with GIF of first episode
    mean_r, std_r = evaluate_policy(
        model, cfg, n_episodes=50, record_gif=True, gif_path="assets/best_eval.gif"
    )
    print(f"\n50-episode evaluation: {mean_r:.1f} +/- {std_r:.1f}")

    # Also eval the final model
    ckpt_final = torch.load("checkpoints/model_final.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt_final["model_state_dict"])
    mean_f, std_f = evaluate_policy(model, cfg, n_episodes=50)
    print(f"Final model (50 eps): {mean_f:.1f} +/- {std_f:.1f}")


if __name__ == "__main__":
    main()
