import numpy as np
from src.env_utils import make_vec_env

envs = make_vec_env(n_envs=2, seed=42)
obs, _ = envs.reset()

assert obs.shape == (2, 4, 84, 84), f"Wrong shape: {obs.shape}"
assert obs.min() >= 0.0 and obs.max() <= 1.0, f"Wrong range: [{obs.min()}, {obs.max()}]"

action = envs.action_space.sample()
obs2, rew, term, trunc, info = envs.step(action)
assert obs2.shape == (2, 4, 84, 84)

envs.close()
print(f"obs shape: {obs.shape}")
print(f"obs range: [{obs.min():.3f}, {obs.max():.3f}]")
print("PHASE 3 PASSED")
