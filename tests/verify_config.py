from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/default.yaml")
assert cfg.env.n_envs == 8
assert cfg.training.total_timesteps == 5_000_000
assert cfg.training.gae_lambda == 0.95
print("Config keys:", list(cfg.keys()))
print("PHASE 2 PASSED")
