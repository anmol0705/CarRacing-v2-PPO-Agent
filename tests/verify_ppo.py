import torch
from src.ppo import compute_gae

# Test 1: GAE shapes and approximate values
rewards = torch.ones(5)
values = torch.zeros(5)
dones = torch.zeros(5)
last_value = torch.tensor(0.0)
adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)

assert adv.shape == (5,), f"adv shape: {adv.shape}"
assert ret.shape == (5,), f"ret shape: {ret.shape}"
assert ret[0].item() > ret[4].item(), "returns should decrease toward end"

# Test 2: done flag cuts return correctly
rewards2 = torch.ones(4)
dones2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
values2 = torch.zeros(4)
last_value2 = torch.tensor(0.0)
adv2, ret2 = compute_gae(
    rewards2, values2, dones2, last_value2, gamma=0.99, lam=0.95
)
assert ret2[2].item() < ret2[0].item(), "done flag should cut future returns"

print(f"returns: {ret.tolist()}")
print(f"advantages: {adv.tolist()}")
print("PHASE 5 PASSED")
