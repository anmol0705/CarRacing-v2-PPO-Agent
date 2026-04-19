"""Train PPO on all three highway-env scenarios sequentially."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.highway_trainer import HighwayPPOConfig, HighwayPPOTrainer

SCENARIOS = [
    {
        "env_name": "highway-v0",
        "total_timesteps": 300_000,
        "n_envs": 8,
        "rollout_steps": 128,
        "n_epochs": 10,
        "minibatch_size": 64,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
        "checkpoint_dir": "checkpoints/highway",
        "log_file": "logs/highway_train.log",
        "eval_freq": 20_000,
        "eval_episodes": 15,
    },
    {
        "env_name": "roundabout-v0",
        "total_timesteps": 200_000,
        "n_envs": 8,
        "rollout_steps": 128,
        "n_epochs": 10,
        "minibatch_size": 64,
        "learning_rate": 3e-4,
        "ent_coef": 0.015,
        "checkpoint_dir": "checkpoints/roundabout",
        "log_file": "logs/roundabout_train.log",
        "eval_freq": 15_000,
        "eval_episodes": 15,
    },
    {
        "env_name": "parking-v0",
        "total_timesteps": 300_000,
        "n_envs": 4,
        "rollout_steps": 256,
        "n_epochs": 10,
        "minibatch_size": 64,
        "learning_rate": 1e-4,
        "ent_coef": 0.005,
        "clip_epsilon": 0.2,
        "checkpoint_dir": "checkpoints/parking",
        "log_file": "logs/parking_train.log",
        "eval_freq": 20_000,
        "eval_episodes": 15,
    },
]

if __name__ == "__main__":
    results = {}
    for scenario in SCENARIOS:
        env_name = scenario["env_name"]
        print(f"\n{'=' * 60}")
        print(f"Training: {env_name}")
        print(f"{'=' * 60}")

        cfg = HighwayPPOConfig(**scenario)
        trainer = HighwayPPOTrainer(cfg)
        t0 = time.time()
        best_reward = trainer.train()
        elapsed = (time.time() - t0) / 3600

        results[env_name] = {
            "best_reward": round(best_reward, 2),
            "training_hours": round(elapsed, 2),
        }
        print(f"Done: {env_name} — best reward {best_reward:.2f} in {elapsed:.2f}h")

    Path("assets").mkdir(exist_ok=True)
    with open("assets/highway_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("ALL HIGHWAY TRAINING COMPLETE")
    print(json.dumps(results, indent=2))
    print(f"{'=' * 60}")
