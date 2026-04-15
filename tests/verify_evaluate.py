import torch
from omegaconf import OmegaConf
from src.model import ActorCritic
from src.evaluate import evaluate_policy

cfg = OmegaConf.load("configs/default.yaml")
model = ActorCritic().cuda()
mean_r, std_r = evaluate_policy(model, cfg, n_episodes=1, record_gif=False)

assert isinstance(mean_r, float), f"mean_r should be float, got {type(mean_r)}"
print(f"Random agent: {mean_r:.1f} +/- {std_r:.1f}")
print("PHASE 6 PASSED")
