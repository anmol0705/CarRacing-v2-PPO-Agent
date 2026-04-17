"""Streamlit dashboard for CarRacing PPO agent — demo, training curves, architecture.

This dashboard is designed to run on Streamlit Community Cloud without
gymnasium, torch, or pygame — it uses only prerecorded GIFs and CSV metrics.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve asset paths relative to this file (works both locally and on Streamlit Cloud)
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="CarRacing PPO Agent",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d5a8e;
    }
    .metric-card h2 { color: #64b5f6; margin: 0; font-size: 2.2em; }
    .metric-card p { color: #90a4ae; margin: 0; font-size: 0.9em; }
    .hero-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #64b5f6, #42a5f5, #1e88e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle { color: #90a4ae; font-size: 1.1em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ---- Hero Section ----
st.markdown('<p class="hero-title">CarRacing-v2 PPO Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">From-scratch PPO implementation achieving near-human driving '
    'performance from raw pixels</p>',
    unsafe_allow_html=True,
)

# Key metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        '<div class="metric-card"><h2>864.7</h2><p>Median Reward (50 eps)</p></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><h2>933.3</h2><p>Max Episode Reward</p></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        '<div class="metric-card"><h2>66%</h2><p>Episodes > 700</p></div>',
        unsafe_allow_html=True,
    )
with m4:
    st.markdown(
        '<div class="metric-card"><h2>5M</h2><p>Training Steps (~9.6 hrs)</p></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ---- Tabs ----
tab_demo, tab_training, tab_arch, tab_about = st.tabs([
    "Agent Demo", "Training Curves", "Architecture", "About",
])


# ==== TAB 1: DEMO ====
with tab_demo:
    col_gif, col_info = st.columns([2, 1])
    with col_gif:
        best_gif = ASSETS / "best_agent.gif"
        if best_gif.exists():
            st.image(str(best_gif), caption="Best episode: 928 reward (full lap completion)")
        else:
            st.warning("best_agent.gif not found — run scripts/record_final_assets.py locally")

    with col_info:
        st.markdown("### Performance Summary")
        st.markdown("""
| Metric | Value |
|:-------|------:|
| Best eval | **811.9** |
| Median (50 eps) | **864.7** |
| Mean (50 eps) | 632.1 |
| Max single episode | **933.3** |
| Episodes > 700 | 66% |
        """)
        st.markdown("---")
        st.markdown("### What you're seeing")
        st.markdown(
            "The agent takes in 4 stacked grayscale frames and outputs continuous "
            "steering, gas, and brake values. After 5M training steps across 8 "
            "parallel environments, it learned to complete full laps at near-human "
            "performance (~900)."
        )

    st.markdown("### Training Progression")
    prog_gif = ASSETS / "progression.gif"
    if prog_gif.exists():
        st.image(str(prog_gif), caption="250K → 1M → 2.5M → 4M → 4.9M steps")

    st.markdown("### Before vs After")
    col_before, col_after = st.columns(2)
    with col_before:
        early_gif = ASSETS / "ep_0pct.gif"
        if early_gif.exists():
            st.image(str(early_gif), caption="Early training (~1M steps)")
    with col_after:
        late_gif = ASSETS / "ep_100pct.gif"
        if late_gif.exists():
            st.image(str(late_gif), caption="Fully trained (best checkpoint)")


# ==== TAB 2: TRAINING CURVES ====
with tab_training:
    eval_csv = ASSETS / "eval_metrics.csv"
    train_csv = ASSETS / "training_metrics.csv"

    if eval_csv.exists() and train_csv.exists():
        eval_df = pd.read_csv(eval_csv)
        train_df = pd.read_csv(train_csv)

        # Eval reward over time
        st.markdown("### Evaluation Reward Over Training")
        eval_chart = eval_df[["step", "eval_reward"]].copy()
        eval_chart = eval_chart.rename(columns={"eval_reward": "Eval Reward"})
        eval_chart["step"] = eval_chart["step"] / 1e6
        eval_chart = eval_chart.rename(columns={"step": "Steps (millions)"})
        st.line_chart(eval_chart, x="Steps (millions)", y="Eval Reward", height=400)

        # Mark best checkpoint
        best_idx = eval_df["eval_reward"].idxmax()
        best_row = eval_df.iloc[best_idx]
        st.success(
            f"Best checkpoint: step {int(best_row['step']):,} — "
            f"eval reward {best_row['eval_reward']:.1f} +/- {best_row['eval_std']:.1f}"
        )

        # Two columns for entropy and training reward
        col_ent, col_pol = st.columns(2)

        with col_ent:
            st.markdown("### Entropy")
            ent_df = train_df[["step", "entropy"]].copy()
            ent_df["step"] = ent_df["step"] / 1e6
            ent_df = ent_df.rename(columns={"step": "Steps (M)", "entropy": "Entropy"})
            st.line_chart(ent_df, x="Steps (M)", y="Entropy", height=300)

        with col_pol:
            st.markdown("### Episode Reward (Training)")
            rew_df = train_df[["step", "ep_reward_mean"]].copy()
            rew_df["step"] = rew_df["step"] / 1e6
            rew_df = rew_df.rename(
                columns={"step": "Steps (M)", "ep_reward_mean": "Mean Episode Reward"}
            )
            st.line_chart(rew_df, x="Steps (M)", y="Mean Episode Reward", height=300)

        # Learning rate and KL
        col_lr, col_kl = st.columns(2)

        with col_lr:
            st.markdown("### Learning Rate")
            lr_df = train_df[["step", "lr"]].copy()
            lr_df["step"] = lr_df["step"] / 1e6
            lr_df = lr_df.rename(columns={"step": "Steps (M)", "lr": "Learning Rate"})
            st.line_chart(lr_df, x="Steps (M)", y="Learning Rate", height=250)

        with col_kl:
            st.markdown("### Approx KL Divergence")
            kl_df = train_df[["step", "approx_kl"]].copy()
            kl_df["step"] = kl_df["step"] / 1e6
            kl_df = kl_df.rename(columns={"step": "Steps (M)", "approx_kl": "KL"})
            st.line_chart(kl_df, x="Steps (M)", y="KL", height=250)

    else:
        st.warning(
            "Training metrics not found. Run `python scripts/export_metrics.py` locally "
            "to generate assets/training_metrics.csv and assets/eval_metrics.csv"
        )


# ==== TAB 3: ARCHITECTURE ====
with tab_arch:
    col_arch, col_hyper = st.columns([3, 2])

    with col_arch:
        st.markdown("### CNN Actor-Critic Architecture")
        st.code("""
Observation: 4 stacked grayscale frames (4 x 84 x 84)
                      |
                      v
+-----------------------------------------+
|          Shared CNN Backbone            |
|                                         |
|  Conv2d(4->32, 8x8, stride 4)  -> ReLU |
|  Conv2d(32->64, 4x4, stride 2) -> ReLU |
|  Conv2d(64->64, 3x3, stride 1) -> ReLU |
|  Flatten -> Linear(3136->512)   -> ReLU |
+--------------+--------------+-----------+
               |              |
        +------+              +------+
        v                            v
+---------------+           +----------------+
|  Actor Head   |           |  Critic Head   |
|               |           |                |
| Linear(512->3)|           | Linear(512->1) |
| + tanh scale  |           | -> V(s)        |
| + learned std |           +----------------+
| -> Normal dist|
| -> [steer,    |
|    gas, brake]|
+---------------+

Total parameters: 1,686,183
        """, language=None)

        st.markdown("### Layer Details")
        st.dataframe(
            pd.DataFrame([
                {"Layer": "Conv2d(4, 32, 8, s=4)", "Output": "32 x 20 x 20", "Params": "8,224"},
                {"Layer": "Conv2d(32, 64, 4, s=2)", "Output": "64 x 9 x 9", "Params": "32,832"},
                {"Layer": "Conv2d(64, 64, 3, s=1)", "Output": "64 x 7 x 7", "Params": "36,928"},
                {"Layer": "Linear(3136, 512)", "Output": "512", "Params": "1,606,144"},
                {"Layer": "Actor: Linear(512, 3)", "Output": "3", "Params": "1,539"},
                {"Layer": "Critic: Linear(512, 1)", "Output": "1", "Params": "513"},
                {"Layer": "log_std (learnable)", "Output": "3", "Params": "3"},
            ]),
            hide_index=True,
            use_container_width=True,
        )

    with col_hyper:
        st.markdown("### Hyperparameters")
        st.dataframe(
            pd.DataFrame([
                {"Parameter": "n_envs", "Value": "8", "Purpose": "Parallel environments"},
                {"Parameter": "rollout_steps", "Value": "256", "Purpose": "Steps before update"},
                {"Parameter": "n_epochs", "Value": "4", "Purpose": "Gradient steps/rollout"},
                {"Parameter": "minibatch_size", "Value": "256", "Purpose": "SGD batch size"},
                {"Parameter": "lr", "Value": "2.5e-4", "Purpose": "LR (linear decay to 10%)"},
                {"Parameter": "gamma", "Value": "0.99", "Purpose": "Discount factor"},
                {"Parameter": "gae_lambda", "Value": "0.95", "Purpose": "GAE lambda"},
                {"Parameter": "clip_eps", "Value": "0.2", "Purpose": "PPO clipping range"},
                {"Parameter": "ent_coef", "Value": "0.01", "Purpose": "Entropy bonus"},
                {"Parameter": "target_kl", "Value": "0.02", "Purpose": "KL early stopping"},
            ]),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Key Design Choices")
        st.markdown("""
- **Centered tanh scaling** for action space mapping
- **Learnable log_std** (init=-1.0, clamp [-2.5, 0.5])
- **Reward normalization** (running std, no mean shift)
- **Orthogonal weight init** with per-head gains
- **No value clipping** (simple MSE works better)
        """)


# ==== TAB 4: ABOUT ====
with tab_about:
    st.markdown("### About This Project")
    st.markdown("""
This is a **from-scratch implementation** of Proximal Policy Optimization (PPO)
that learns to drive in OpenAI Gymnasium's CarRacing-v2 environment from raw pixels.

**No pretrained models. No Stable-Baselines3.** Every line of the algorithm — GAE,
clipped surrogate objective, entropy bonus, reward normalization — is hand-written
in PyTorch.

#### Training Journey

The agent trained for **5 million steps** (~9.6 hours on an NVIDIA T4 GPU) across
8 parallel environments. The learning curve shows a characteristic hockey-stick
pattern:

- **0-2M steps:** Slow improvement while learning basic track-following
- **2-4M steps:** Rapid improvement as steering + throttle + braking combine
- **4-5M steps:** Fine-tuning to near-human performance (median 865)

#### Bugs I Fixed

| Bug | Fix |
|:----|:----|
| Action gradient saturation | Centered tanh scaling + small init gain |
| log_std stuck at clamp boundary | Widened range, init in middle |
| Value loss instability | Removed value clipping |
| Over-optimization per rollout | Reduced epochs from 10 to 4 |
| box2d Python 3.13 crash | Custom Float64Action wrapper |

#### Built With

**PyTorch** · **Gymnasium** · **Hydra** · **Weights & Biases** · **Streamlit**
    """)

    st.markdown("### Run It Yourself")
    st.code("""
# Clone and install
git clone https://github.com/anmol0705/CarRacing-v2-PPO-Agent.git
cd CarRacing-v2-PPO-Agent
pip install -r requirements.txt gymnasium[box2d] torch

# Train (~10 hours on T4 GPU)
python scripts/train.py

# Evaluate
python scripts/eval_detailed.py

# Record GIFs
python scripts/record_hero.py
    """, language="bash")


# ---- Footer ----
st.divider()
st.markdown(
    '<p style="text-align:center; color:#546e7a; font-size:0.85em;">'
    'Built from scratch with PyTorch · No Stable-Baselines3 · '
    '<a href="https://github.com/anmol0705/CarRacing-v2-PPO-Agent" '
    'style="color:#64b5f6;">GitHub</a></p>',
    unsafe_allow_html=True,
)
