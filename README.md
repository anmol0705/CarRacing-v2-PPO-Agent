# CarRacing-v2 PPO Agent

<p align="center">
  <img src="assets/best_agent.gif" alt="Trained PPO agent completing a full lap" width="600">
  <br>
  <em>PPO agent scoring 928 — completing a full lap after 5M training steps</em>
</p>

A **from-scratch PPO implementation** that learns to drive in OpenAI Gymnasium's CarRacing-v2 from raw pixels. No pretrained models, no Stable-Baselines3 — just PyTorch, a CNN, and 5 million frames of practice.

---

## Results

| Metric | Value |
|:-------|------:|
| Best eval reward | **811.9** |
| 50-episode median | **864.7** |
| 50-episode mean | 632.1 &pm; 363 |
| Max single episode | **933.3** |
| Episodes scoring >700 | 66% (33/50) |
| Training steps | 5,000,000 |
| Training time | ~9.6 hrs (T4 GPU) |
| Target | 700 &#10004; |

> Human-level on CarRacing is ~900. The bimodal score distribution (some ~0, most ~870) is characteristic of this environment — procedurally generated tracks mean some layouts have hairpin turns that are harder to navigate.

<p align="center">
  <img src="assets/progression.gif" alt="Training progression from 250K to 4.9M steps">
  <br>
  <em>Training progression: 250K &rarr; 1M &rarr; 2.5M &rarr; 4M &rarr; 4.9M steps</em>
</p>

---

## What This Is

**Proximal Policy Optimization (PPO)** is a policy gradient algorithm that learns by collecting batches of experience, computing how much better or worse each action was compared to average (the *advantage*), then updating the policy with a clipped objective that prevents catastrophically large changes. It's the workhorse algorithm behind ChatGPT's RLHF training and many robotics applications.

This agent takes in 4 stacked grayscale frames (giving it a sense of motion), processes them through a convolutional neural network, and outputs continuous control signals: steering angle, gas pedal, and brake. Over 5 million frames of practice across 8 parallel environments, it progresses from random flailing to consistently completing full laps at near-human performance.

---

## Architecture

```
 Observation: 4 stacked grayscale frames (4 x 84 x 84)
                        |
                        v
 ┌─────────────────────────────────────────────┐
 │            Shared CNN Backbone               │
 │                                              │
 │  Conv2d(4 -> 32, 8x8, stride 4)  -> ReLU    │
 │  Conv2d(32 -> 64, 4x4, stride 2) -> ReLU    │
 │  Conv2d(64 -> 64, 3x3, stride 1) -> ReLU    │
 │  Flatten -> Linear(3136 -> 512)   -> ReLU    │
 └──────────────┬───────────────┬───────────────┘
                |               |
         ┌──────┘               └──────┐
         v                             v
 ┌────────────────┐           ┌─────────────────┐
 │   Actor Head   │           │   Critic Head   │
 │                │           │                 │
 │ Linear(512->3) │           │ Linear(512->1)  │
 │ + tanh scaling │           │ -> V(s)         │
 │ + learned std  │           └─────────────────┘
 │ -> Normal dist │
 │ -> [steer,     │
 │    gas, brake] │
 └────────────────┘
```

| Layer | Output Shape | Parameters |
|:------|:-------------|----------:|
| Input | 4 x 84 x 84 | — |
| Conv2d(4, 32, 8, stride=4) | 32 x 20 x 20 | 8,224 |
| Conv2d(32, 64, 4, stride=2) | 64 x 9 x 9 | 32,832 |
| Conv2d(64, 64, 3, stride=1) | 64 x 7 x 7 | 36,928 |
| Linear(3136, 512) | 512 | 1,606,144 |
| Actor: Linear(512, 3) | 3 | 1,539 |
| Critic: Linear(512, 1) | 1 | 513 |
| log_std (learnable) | 3 | 3 |
| **Total** | | **1,686,183** |

**Weight initialization:** Orthogonal with gain=sqrt(2) for CNN layers, gain=0.01 for actor (small initial actions), gain=1.0 for critic.

---

## Hyperparameters

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml) — nothing is hardcoded in Python.

| Parameter | Value | Purpose |
|:----------|------:|:--------|
| `n_envs` | 8 | Parallel environment workers |
| `rollout_steps` | 256 | Steps collected before each update |
| `n_epochs` | 4 | Gradient steps per rollout |
| `minibatch_size` | 256 | SGD batch size |
| `lr` | 2.5e-4 | Learning rate (linear decay to 10% floor) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_eps` | 0.2 | PPO clipping range |
| `ent_coef` | 0.01 | Entropy bonus weight |
| `target_kl` | 0.02 | KL divergence early stopping threshold |

---

## How to Run

### Install

```bash
git clone https://github.com/anmol0705/CarRacing-v2-PPO-Agent.git
cd carracing-ppo
pip install -r requirements.txt
```

### Train from scratch

```bash
# ~10 hours on a T4 GPU, ~5M steps
python scripts/train.py

# Monitor training
tail -f logs/train_v5.log
```

### Evaluate a checkpoint

```bash
# Detailed 50-episode evaluation with per-episode breakdown
python scripts/eval_detailed.py

# Quick 10-episode eval
python -c "
from src.model import ActorCritic
from src.evaluate import evaluate_policy
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.load('configs/default.yaml')
model = ActorCritic().cuda()
ckpt = torch.load('checkpoints/model_best.pt', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
mean_r, std_r = evaluate_policy(model, cfg, n_episodes=10)
print(f'Reward: {mean_r:.1f} +/- {std_r:.1f}')
"
```

### Record GIFs

```bash
python scripts/record_hero.py           # Best episode hero GIF
python scripts/record_gif.py            # Progression GIF from checkpoints
python scripts/record_final_assets.py   # All assets at once
```

### Run the Streamlit demo

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
carracing-ppo/
├── configs/
│   └── default.yaml              # All hyperparameters (single source of truth)
├── src/
│   ├── env_utils.py              # Wrappers: grayscale, resize, normalize, frame stack
│   ├── model.py                  # ActorCritic CNN with Gaussian policy
│   ├── ppo.py                    # GAE computation + clipped PPO update
│   ├── trainer.py                # Training loop: rollouts, updates, W&B logging
│   └── evaluate.py               # Greedy evaluation + GIF recording
├── scripts/
│   ├── train.py                  # Hydra entry point
│   ├── record_gif.py             # Progression GIF from checkpoints
│   ├── record_hero.py            # Best-episode hero GIF
│   ├── record_final_assets.py    # All demo assets at once
│   ├── eval_detailed.py          # 50-episode eval with full statistics
│   └── export_metrics.py         # Parse logs -> CSV for dashboard
├── dashboard/
│   └── app.py                    # Streamlit interactive demo
├── assets/
│   ├── best_agent.gif            # Best episode (928 reward)
│   ├── progression.gif           # Side-by-side training progression
│   ├── demo_clip.gif             # 10-second highlight
│   ├── training_metrics.csv      # Parsed training curves
│   └── eval_metrics.csv          # Evaluation results over training
├── tests/                        # Phase verification scripts
└── BUILD_LOG.txt                 # Detailed build + training log
```

---

## Key Implementation Details

- **Frame stacking (4 frames):** A single frame has no velocity information. Stacking 4 consecutive grayscale frames gives the network implicit velocity and acceleration cues, satisfying the Markov property required by RL algorithms.

- **GAE (&#955;=0.95):** Generalized Advantage Estimation provides a smooth tradeoff between high-bias (TD) and high-variance (Monte Carlo) advantage estimates. At &#955;=0.95, we get low-bias estimates that stabilize training.

- **PPO clip (&#949;=0.2):** The clipped surrogate objective prevents catastrophically large policy updates. The ratio between new and old action probabilities is clamped to [0.8, 1.2], giving a trust region without the computational cost of TRPO.

- **Entropy bonus (0.01):** Adds the distribution entropy to the loss, preventing premature convergence to a deterministic policy. This encourages the agent to keep exploring until it finds a good strategy.

- **Centered tanh action scaling:** Maps unbounded network outputs to valid action ranges — steer [-1,1], gas [0,1], brake [0,1] — without gradient saturation. Each dimension scales around its center point.

- **Reward normalization:** Running mean/std normalization (scale only, no shift) keeps reward magnitudes stable for the value network throughout training, even as the agent discovers higher-reward strategies.

- **Learnable exploration:** The log standard deviation starts at -1.0 (std=0.37) and is clamped to [-2.5, 0.5], allowing the agent to learn its own exploration schedule rather than relying on a fixed noise level.

---

## Training Journey

| Step | Eval Reward | Phase |
|-----:|----------:|:------|
| 50K | -7.7 | Random flailing, learning basic control |
| 500K | 1.0 | Stopped crashing immediately |
| 1.5M | 23.2 | First signs of forward driving |
| 2.5M | 99.2 | Following straight sections |
| 3.0M | 155.4 | Starting to handle turns |
| 4.0M | 307.5 | Completing partial laps |
| 4.2M | 630.5 | Near-complete laps |
| 4.7M | 752.7 | Consistent full laps |
| **4.9M** | **811.9** | **Best model — near human-level** |

The training curve shows a characteristic hockey-stick pattern: ~2M steps of slow improvement while the agent learns basic track-following, then rapid improvement from 2.5M-4.9M as it chains skills (steering + throttle control + brake timing) into complete lap completion. Entropy decreased smoothly from 1.28 to 0.98, confirming a healthy exploration-to-exploitation transition.

---

## Bugs I Fixed

| Bug | Root Cause | Fix |
|:----|:-----------|:----|
| Action saturation | `tanh(large_number)` kills gradients | Centered tanh scaling + small init gain (0.01) |
| log_std not learning | Clamped at boundary, gradient=0 | Widened clamp to [-2.5, 0.5], init at -1.0 |
| Value loss instability | Value clipping counterproductive | Removed clipping, simple MSE |
| Over-optimization | 10 epochs too aggressive | Reduced to 4 epochs per rollout |
| box2d float crash | Python 3.13 SWIG type mismatch | Custom `Float64Action` wrapper |
| CarRacing-v3 missing | gymnasium 0.29.1 only has v2 | Changed env ID |

---

## Built With

**PyTorch** &middot; **Gymnasium** &middot; **Hydra** &middot; **Weights & Biases** &middot; **Streamlit** &middot; **imageio**

---

<p align="center">
  <sub>Built from scratch as a portfolio project demonstrating reinforcement learning fundamentals.<br>No Stable-Baselines3, no pretrained models — every line of the algorithm is hand-written.</sub>
</p>
