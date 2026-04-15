# CarRacing-v2 PPO Agent

![Training progression](assets/progression.gif)

## What this is

A Proximal Policy Optimization (PPO) agent trained from raw pixels to drive in Gymnasium's CarRacing-v2 environment. The agent uses a CNN feature extractor with continuous action control (steering, gas, brake) and targets human-level performance (reward > 700).

## Architecture

```
Input: 4 stacked grayscale frames (4 x 84 x 84)
  |
  v
Shared CNN Backbone:
  Conv2d(4, 32, 8, stride=4) -> ReLU
  Conv2d(32, 64, 4, stride=2) -> ReLU
  Conv2d(64, 64, 3, stride=1) -> ReLU
  Flatten -> Linear(3136, 512) -> ReLU
  |
  +---> Actor: Linear(512, 3) -> tanh (mean)
  |     + learnable log_std -> Normal distribution
  |     -> [steer, gas, brake] in [-1, 1]
  |
  +---> Critic: Linear(512, 1) -> state value V(s)
```

- **Continuous actions**: steer (-1 to 1), gas (0 to 1), brake (0 to 1)
- **Frame stacking**: 4 consecutive grayscale frames provide implicit velocity information
- **Orthogonal initialization**: sqrt(2) gain for CNN, 0.01 for actor, 1.0 for critic

## Results

| Checkpoint | Mean Reward | Std | Steps |
|---|---|---|---|
| Random agent | ~-50 | -- | 0 |
| 25% training | TBD | -- | 1.25M |
| 50% training | TBD | -- | 2.5M |
| Fully trained | TBD | -- | 5M |

## How to reproduce

```bash
git clone https://github.com/YOUR_USERNAME/carracing-ppo.git
cd carracing-ppo
pip install -r requirements.txt
python scripts/train.py
```

### Monitor training
```bash
tail -f logs/train.log
# Or check W&B dashboard
```

### Evaluate a checkpoint
```bash
python -c "
from src.model import ActorCritic
from src.evaluate import evaluate_policy
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.load('configs/default.yaml')
model = ActorCritic().cuda()
ckpt = torch.load('checkpoints/model_final.pt', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
mean_r, std_r = evaluate_policy(model, cfg, n_episodes=10)
print(f'Reward: {mean_r:.1f} +/- {std_r:.1f}')
"
```

### Record progression GIF
```bash
python scripts/record_gif.py          # from real checkpoints
python scripts/record_gif.py --demo   # random agent placeholder
```

### Live dashboard
```bash
streamlit run dashboard/app.py
```

## Key concepts

- **PPO over DQN**: CarRacing has a continuous action space (steering angle, gas, brake). DQN requires discrete actions, while PPO naturally handles continuous control via Gaussian policy distributions.

- **Frame stacking**: A single frame lacks motion information. Stacking 4 consecutive frames gives the network implicit velocity and acceleration cues, satisfying the Markov property.

- **GAE (Generalized Advantage Estimation)**: Balances bias vs. variance in advantage estimation via the lambda parameter. High lambda (0.95) gives lower bias at the cost of higher variance.

- **Entropy bonus**: Prevents premature policy collapse by encouraging exploration. The entropy coefficient (0.01) keeps the action distribution from becoming too narrow too early.

- **Parallel environments**: 8 async environments provide decorrelated samples for PPO updates, improving training stability and throughput.

## Hyperparameters

All hyperparameters live in `configs/default.yaml`:

| Parameter | Value | Purpose |
|---|---|---|
| n_envs | 8 | Parallel environment workers |
| rollout_steps | 128 | Steps per env before update |
| n_epochs | 4 | PPO optimization epochs |
| minibatch_size | 256 | SGD minibatch size |
| lr | 3e-4 | Initial learning rate (linear decay) |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE bias-variance tradeoff |
| clip_eps | 0.2 | PPO clipping range |
| ent_coef | 0.01 | Entropy bonus weight |
| target_kl | 0.015 | Early stopping threshold |

## Project structure

```
carracing-ppo/
├── configs/default.yaml    # All hyperparameters
├── src/
│   ├── env_utils.py        # Wrappers + VecEnv factory
│   ├── model.py            # ActorCritic CNN
│   ├── ppo.py              # GAE + PPO update
│   ├── trainer.py          # Training loop
│   └── evaluate.py         # Greedy eval + GIF recording
├── scripts/
│   ├── train.py            # Hydra entry point
│   └── record_gif.py       # Progression GIF builder
├── dashboard/app.py        # Streamlit live demo
├── tests/                  # Verification scripts
└── assets/                 # GIFs for README
```
