import torch
from src.model import ActorCritic

model = ActorCritic().cuda()
dummy = torch.zeros(8, 4, 84, 84).cuda()
action, log_prob, value, entropy = model(dummy)

assert action.shape == (8, 3), f"action shape: {action.shape}"
assert log_prob.shape == (8,), f"log_prob shape: {log_prob.shape}"
assert value.shape == (8, 1), f"value shape: {value.shape}"
assert entropy.shape == (8,), f"entropy shape: {entropy.shape}"
assert action.min() >= -1.0 and action.max() <= 1.0, "actions out of [-1,1]"

params = sum(p.numel() for p in model.parameters())
assert 900_000 < params < 2_000_000, f"unexpected param count: {params}"

print(f"action shape: {action.shape}")
print(f"value shape: {value.shape}")
print(f"param count: {params:,}")
print("PHASE 4 PASSED")
