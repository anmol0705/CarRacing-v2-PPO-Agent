# CarRacing-v2 PPO Agent

![Training progression](assets/progression.gif)

A Proximal Policy Optimization (PPO) agent trained from raw pixels to drive in Gymnasium's CarRacing-v2 environment. Achieves **median reward of 865** (human-level ~900) using a CNN backbone with continuous action control.

## Results

| Metric | Value |
|---|---|
| **Median reward (50 eps)** | **864.7** |
| Mean reward (50 eps) | 632.1 |
| Max episode reward | 933.3 |
| Episodes > 700 | 66% |
| Episodes > 500 | 72% |
| Training steps | 5M |
| Training time | ~9.6 hours (T4 GPU) |

| Checkpoint | Eval Reward | Steps |
|---|---|---|
| Early training | -7.7 | 50K |
| 25% training | 64.9 | 1.25M |
| 50% training | 155.4 | 2.5M |
| 75% training | 225.9 | 3.75M |
| Best model | 811.9 | 4.9M |

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
  +---> Actor: Linear(512, 3) -> tanh scaling to action space
  |     + learnable log_std -> Normal distribution
  |     -> steer [-1,1], gas [0,1], brake [0,1]
  |
  +---> Critic: Linear(512, 1) -> state value V(s)
```

- **Continuous actions**: steer (-1 to 1), gas (0 to 1), brake (0 to 1) via centered tanh scaling
- **Frame stacking**: 4 consecutive grayscale frames provide implicit velocity information
- **Orthogonal initialization**: sqrt(2) gain for CNN, 0.01 for actor, 1.0 for critic
- **Learnable exploration**: log_std parameter clamped to [-2.5, 0.5], initialized at -1.0

## How to reproduce

```bash
git clone https://github.com/YOUR_USERNAME/carracing-ppo.git
cd carracing-ppo
pip install -r requirements.txt
python scripts/train.py
```

### Monitor training
```bash
tail -f logs/train_v5.log
```

### Evaluate a checkpoint
```bash
python scripts/eval_detailed.py
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

## Key design decisions

- **PPO over DQN**: CarRacing has a continuous action space (steering angle, gas, brake). DQN requires discrete actions, while PPO naturally handles continuous control via Gaussian policy distributions.

- **Frame stacking**: A single frame lacks motion information. Stacking 4 consecutive frames gives the network implicit velocity and acceleration cues, satisfying the Markov property.

- **GAE (Generalized Advantage Estimation)**: Balances bias vs. variance in advantage estimation via the lambda parameter (0.95). Higher lambda reduces bias at the cost of higher variance.

- **Entropy bonus**: Prevents premature policy collapse by encouraging exploration. The entropy coefficient (0.01) keeps the action distribution from becoming too narrow too early.

- **Reward normalization**: Running mean/std normalization stabilizes training when reward magnitudes change throughout the episode.

- **Centered tanh action scaling**: Maps unbounded network outputs to the action space without gradient saturation. Each action dimension is scaled around its center (steer: 0, gas: 0.5, brake: 0.5).

- **Parallel environments**: 8 async environments provide decorrelated samples for PPO updates, improving training stability and throughput (~144 SPS on T4).

## Hyperparameters

All hyperparameters live in `configs/default.yaml`:

| Parameter | Value | Purpose |
|---|---|---|
| n_envs | 8 | Parallel environment workers |
| rollout_steps | 256 | Steps per env before update |
| n_epochs | 4 | PPO optimization epochs |
| minibatch_size | 256 | SGD minibatch size |
| lr | 2.5e-4 | Initial learning rate (linear decay to 10%) |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE bias-variance tradeoff |
| clip_eps | 0.2 | PPO clipping range |
| ent_coef | 0.01 | Entropy bonus weight |
| target_kl | 0.02 | Early stopping threshold |

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
│   ├── record_gif.py       # Progression GIF builder
│   └── eval_detailed.py    # Detailed evaluation with stats
├── dashboard/app.py        # Streamlit live demo
├── tests/                  # Verification scripts
└── assets/                 # GIFs for README
```
