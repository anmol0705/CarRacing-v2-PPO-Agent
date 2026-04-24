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
        "total_timesteps": 500_000,
        "n_envs": 8,
        "rollout_steps": 128,
        "n_epochs": 10,
        "minibatch_size": 128,
        "learning_rate": 3e-4,
        "ent_coef": 0.01,
        "checkpoint_dir": "checkpoints/highway",
        "log_file": "logs/highway_train.log",
        "eval_freq": 25_000,
        "eval_episodes": 20,
        "log_freq": 5_000,
    },
    {
        "env_name": "roundabout-v0",
        "total_timesteps": 300_000,
        "n_envs": 8,
        "rollout_steps": 128,
        "n_epochs": 10,
        "minibatch_size": 128,
        "learning_rate": 3e-4,
        "ent_coef": 0.02,
        "checkpoint_dir": "checkpoints/roundabout",
        "log_file": "logs/roundabout_train.log",
        "eval_freq": 25_000,
        "eval_episodes": 20,
        "log_freq": 5_000,
    },
    {
        "env_name": "parking-v0",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "rollout_steps": 256,
        "n_epochs": 10,
        "minibatch_size": 128,
        "learning_rate": 5e-4,
        "ent_coef": 0.01,
        "clip_epsilon": 0.2,
        "gamma": 0.95,
        "checkpoint_dir": "checkpoints/parking",
        "log_file": "logs/parking_train.log",
        "eval_freq": 25_000,
        "eval_episodes": 20,
        "log_freq": 5_000,
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
