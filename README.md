<div align="center">

# PPO for Autonomous Driving

### Teaching an AI to drive with nothing but trial, error, and 7 million attempts

<br>

<img src="assets/showcase_small.gif" alt="PPO agent racing around the track at 940 reward" width="520">

<br>

**940 / 1000** on CarRacing-v2 &mdash; that's *above human level*.
<br>No GPS. No map. No hand-coded rules. Just 4 grayscale frames and a dream.

<br>

[![Streamlit](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://carracing-v2-ppo-agent.streamlit.app)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)]()
&nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)]()

</div>

---

## The TL;DR

I built a **Proximal Policy Optimization (PPO)** agent from scratch &mdash; no Stable-Baselines3, no pretrained anything &mdash; and threw it at four different driving challenges. Here's how it did:

| Environment | What it does | Score | Train time |
|:------------|:-------------|------:|:-----------|
| **CarRacing-v2** | Full-speed laps from raw pixels | **940** (peak) / 874 (avg) | ~10 hrs |
| **Highway-v0** | Lane changes & overtaking at 130 km/h | **139.8** | ~30 min |
| **Roundabout-v0** | Enter, navigate, exit without crashing | **41.6** | ~17 min |
| **Parking-v0** | Reverse into a tight parking spot | **-11.9** (lower = better) | ~20 min |

> Human performance on CarRacing is ~900. The agent consistently beats that.

---

## Watch It Drive

### CarRacing-v2 &mdash; From zero to hero

The agent sees 4 stacked grayscale frames (84x84 each), decides how much to steer, accelerate, and brake &mdash; 50 times per second. After 5 million steps of practice across 8 parallel tracks, it goes from drunk-driving-on-ice to smooth full-lap completions.

<div align="center">
<img src="assets/progression.gif" alt="Training progression across checkpoints" width="700">
<br>
<em>Left to right: 250K steps (chaos) &rarr; 1M (learning to steer) &rarr; 2.5M (following the road) &rarr; 4.9M (full laps)</em>
</div>

<br>

### Highway-v0 &mdash; Overtaking at speed

No pixels here &mdash; the agent reads positions and velocities of nearby cars and decides: go faster, slow down, or change lanes. Think of it as a really aggressive GPS that only cares about going fast without crashing.

<div align="center">
<img src="assets/highway_demo.gif" alt="Highway overtaking demo" width="520">
</div>

<br>

### Roundabout-v0 &mdash; Navigating the chaos circle

Roundabouts are tricky even for humans. The agent has to time its entry, navigate around traffic, and exit cleanly. It manages to do this roughly 75% of the time, which is honestly better than some drivers I know.

<div align="center">
<img src="assets/roundabout_demo.gif" alt="Roundabout navigation demo" width="400">
</div>

<br>

### Parking-v0 &mdash; The final boss

Goal-conditioned sparse reward &mdash; the hardest setup in RL. The agent only gets told "how far from the target spot are you?" and has to figure out steering + throttle to park perfectly. It's not winning any valet competitions yet, but it tries.

<div align="center">
<img src="assets/parking_demo.gif" alt="Parking demo" width="520">
</div>

---

## How PPO Actually Works (the 30-second version)

```
1. Let the agent drive around and collect experiences      (rollout)
2. For each action, ask: "Was this better or worse         (advantage)
   than what I expected?"                                   estimation)
3. Update the brain, but NOT too much at once              (clipped
   (this is the "proximal" part)                            objective)
4. Repeat 5 million times
5. ???
6. Profit (or at least, finish the lap)
```

The key insight: PPO says *"hey, I know this action looked great, but let's not go crazy &mdash; only update the policy a little bit."* This prevents the common RL disaster where one lucky experience makes the agent think it should ALWAYS turn left at full speed.

---

## The Architecture

Two brains, one body:

```
                    4 x 84 x 84 grayscale frames
                              |
                    +---------v---------+
                    |   Shared CNN      |
                    |   3 conv layers   |
                    |   + linear(512)   |    1.78M parameters
                    +---------+---------+
                              |
                     +--------+--------+
                     |                 |
              +------v------+   +------v------+
              | Actor Head  |   | Critic Head |
              | "what to do"|   | "how good   |
              | steer/gas/  |   |  is this    |
              | brake       |   |  situation?"|
              +-------------+   +-------------+
```

**CarRacing** uses the CNN above. **Highway environments** use a simpler 2-layer MLP (256 hidden units) since observations are already structured vectors, not pixels.

<details>
<summary><b>Full layer breakdown (click to expand)</b></summary>

| Layer | Output | Params |
|:------|:-------|-------:|
| Conv2d(4, 32, 8, stride=4) | 32 x 20 x 20 | 8,224 |
| Conv2d(32, 64, 4, stride=2) | 64 x 9 x 9 | 32,832 |
| Conv2d(64, 64, 3, stride=1) | 64 x 7 x 7 | 36,928 |
| Linear(3136, 512) | 512 | 1,606,144 |
| Actor: Linear(512, 3) | 3 | 1,539 |
| Critic: Linear(512, 1) | 1 | 513 |
| log_std (learnable) | 3 | 3 |
| **Total** | | **1,686,183** |

</details>

---

## The Training Journey

This is what 10 hours of GPU time looks like:

| Step | Reward | What's happening |
|-----:|-------:|:-----------------|
| 50K | -8 | Spinning in circles. Learning that walls hurt. |
| 500K | 1 | Stopped actively trying to die. Progress! |
| 1.5M | 23 | Discovered that going forward is good, actually |
| 2.5M | 99 | Can follow straight roads. Turns are still scary. |
| 3.5M | 206 | Starting to handle turns. Sometimes. |
| 4.2M | **631** | First near-complete laps! |
| 4.7M | **753** | Consistent full laps. Target of 700 smashed. |
| 4.9M | **812** | Best eval checkpoint. Near human-level. |
| 5M+ | **874** | Fine-tuned. Peak single episode: **940.4** |

The classic hockey-stick curve: 3 million steps of "is this thing even learning?" followed by rapid improvement once the agent figures out how to chain skills together.

---

## Same Algorithm, Four Different Worlds

The cool part: **the exact same PPO code** works across all four environments. The only thing that changes is the observation size and whether actions are discrete or continuous.

| | CarRacing | Highway | Roundabout | Parking |
|:--|:----------|:--------|:-----------|:--------|
| **Sees** | 4x84x84 pixels | 5x5 kinematics | 5x5 kinematics | 18-dim goal vector |
| **Does** | Steer + gas + brake | 5 discrete choices | 5 discrete choices | Steer + throttle |
| **Reward** | Dense (per tile) | Dense (speed) | Dense (progress) | Sparse (distance) |
| **Traffic?** | Just you | 10 cars | 10 cars | Parked cars |
| **Hard part** | Vision + control | Speed + safety | Timing | Precision |

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/anmol0705/CarRacing-v2-PPO-Agent.git
cd carracing-ppo
pip install -r requirements.txt

# Train CarRacing (~10 hrs on GPU)
python scripts/train.py

# Train all highway scenarios (~1 hr)
python scripts/train_highway.py

# Run the dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
carracing-ppo/
|-- configs/
|   |-- default.yaml         # CarRacing hyperparameters
|   +-- finetune.yaml        # Fine-tuning config
|-- src/
|   |-- model.py             # CNN ActorCritic (pixels)
|   |-- ppo.py               # GAE + clipped PPO update
|   |-- trainer.py           # Training loop with W&B
|   |-- highway_trainer.py   # MLP PPO for highway-env
|   +-- env_utils.py         # Wrappers, VecEnv factory
|-- scripts/
|   |-- train.py             # CarRacing entry point
|   |-- train_highway.py     # Highway/Roundabout/Parking
|   |-- record_showcase.py   # Best-episode HUD recording
|   +-- record_highway.py    # Highway-env GIF recording
|-- dashboard/
|   +-- app.py               # Streamlit live demo
+-- assets/                  # GIFs, CSVs, PNGs
```

---

## Bugs That Almost Broke Everything

| Bug | What happened | Fix |
|:----|:-------------|:----|
| **Gradient death** | `tanh` on large values = zero gradients forever | Centered tanh scaling + tiny init gain (0.01) |
| **log_std frozen** | Clamped at boundary, optimizer couldn't move it | Widened clamp range [-2.5, 0.5] |
| **Python 3.13 crash** | box2d SWIG wrapper rejected float32 | Custom `Float64Action` wrapper |
| **Value loss spikes** | Value clipping was counterproductive | Removed clipping, simple MSE works better |
| **10-epoch overfit** | Too many gradient steps per batch | Reduced to 4 epochs + KL early stopping |

---

## Key Design Decisions

**Why PPO over DQN?** &mdash; CarRacing has continuous actions (how much to steer, not just left/right). DQN only works with discrete actions. PPO handles both.

**Why frame stacking?** &mdash; One frame is a photograph. You can't tell if the car is moving or which direction. Four frames give the network velocity and acceleration for free.

**Why 8 parallel envs?** &mdash; PPO needs decorrelated samples. Running 8 tracks simultaneously means experiences in the same batch come from different situations, which makes training much more stable.

**Why entropy bonus?** &mdash; Without it, the agent quickly decides "I'll just always turn left" and stops exploring. The entropy term keeps the policy uncertain enough to discover better strategies.

---

## Built With

**PyTorch** &middot; **Gymnasium** &middot; **highway-env** &middot; **Hydra** &middot; **W&B** &middot; **Streamlit** &middot; **Plotly**

Trained on **AWS EC2 g4dn.xlarge** (NVIDIA T4, 16GB VRAM)

---

<div align="center">
<sub>
Every line of the RL algorithm is hand-written. No Stable-Baselines3 shortcuts.
<br>
Built as a portfolio project to demonstrate deep RL fundamentals.
</sub>
</div>
