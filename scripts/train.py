"""Entry point for PPO training on CarRacing-v2."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig

from src.trainer import PPOTrainer


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Launch PPO training with Hydra config."""
    # Change back to original dir (Hydra changes cwd)
    os.chdir(hydra.utils.get_original_cwd())

    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
