"""Streamlit dashboard for CarRacing PPO agent — demo, training curves, architecture."""

import glob
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
tab_demo, tab_training, tab_arch, tab_run = st.tabs([
    "Agent Demo", "Training Curves", "Architecture", "Run Episode",
])


# ==== TAB 1: DEMO ====
with tab_demo:
    col_gif, col_info = st.columns([2, 1])
    with col_gif:
        best_gif = Path("assets/best_agent.gif")
        if best_gif.exists():
            st.image(str(best_gif), caption="Best episode: 928 reward (full lap)")
        else:
            st.warning("assets/best_agent.gif not found. Run scripts/record_final_assets.py")

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
    prog_gif = Path("assets/progression.gif")
    if prog_gif.exists():
        st.image(str(prog_gif), caption="250K → 1M → 2.5M → 4M → 4.9M steps")


# ==== TAB 2: TRAINING CURVES ====
with tab_training:
    eval_csv = Path("assets/eval_metrics.csv")
    train_csv = Path("assets/training_metrics.csv")

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
            f"eval reward {best_row['eval_reward']:.1f} ± {best_row['eval_std']:.1f}"
        )

        # Two columns for entropy and policy loss
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
            "Training metrics not found. Run `python scripts/export_metrics.py` "
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
┌───────────────────────────────────────────┐
│           Shared CNN Backbone             │
│                                           │
│  Conv2d(4→32, 8x8, stride 4)  → ReLU     │
│  Conv2d(32→64, 4x4, stride 2) → ReLU     │
│  Conv2d(64→64, 3x3, stride 1) → ReLU     │
│  Flatten → Linear(3136→512)    → ReLU     │
└──────────────┬──────────────┬─────────────┘
               |              |
        ┌──────┘              └──────┐
        v                            v
┌───────────────┐           ┌────────────────┐
│  Actor Head   │           │  Critic Head   │
│               │           │                │
│ Linear(512→3) │           │ Linear(512→1)  │
│ + tanh scale  │           │ → V(s)         │
│ + learned std │           └────────────────┘
│ → Normal dist │
│ → [steer,     │
│    gas, brake] │
└───────────────┘

Total parameters: 1,686,183
        """)

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


# ==== TAB 4: RUN EPISODE ====
with tab_run:
    st.markdown("### Run a Live Episode")

    checkpoint_files = sorted(glob.glob("checkpoints/*.pt"))
    if not checkpoint_files:
        st.warning("No checkpoints found in checkpoints/")
    else:
        col_ctrl, col_space = st.columns([1, 2])
        with col_ctrl:
            selected_ckpt = st.selectbox("Checkpoint", checkpoint_files, index=len(checkpoint_files) - 1)
            seed = st.number_input("Environment seed", value=0, min_value=0, max_value=9999)
            run_episode = st.button("Run Episode", type="primary")

        if run_episode:
            import time
            import torch
            from src.model import ActorCritic
            from src.env_utils import make_env

            @st.cache_resource
            def load_model(ckpt_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = ActorCritic().to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                return model

            model = load_model(selected_ckpt)
            device = next(model.parameters()).device

            env_fn = make_env(seed=int(seed), render_mode="rgb_array")
            env = env_fn()
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            rewards_over_time = []

            col_main, col_actions = st.columns([3, 1])
            frame_placeholder = col_main.empty()
            steer_text = col_actions.empty()
            gas_text = col_actions.empty()
            brake_text = col_actions.empty()
            reward_text = col_actions.empty()
            step_text = col_actions.empty()
            chart_placeholder = st.empty()

            while not done:
                obs_tensor = torch.as_tensor(
                    np.array(obs), dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = model.get_greedy_action(obs_tensor)
                action_np = action.cpu().numpy().squeeze(0)

                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                total_reward += float(reward)
                step_count += 1
                rewards_over_time.append(total_reward)

                frame = env.render()
                if frame is not None:
                    frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                steer_text.metric("Steer", f"{action_np[0]:+.2f}")
                gas_text.metric("Gas", f"{action_np[1]:.2f}")
                brake_text.metric("Brake", f"{action_np[2]:.2f}")
                reward_text.metric("Total Reward", f"{total_reward:.1f}")
                step_text.metric("Step", str(step_count))

                time.sleep(1.0 / 20.0)

            env.close()

            st.subheader("Cumulative Reward")
            st.line_chart(pd.DataFrame({"Reward": rewards_over_time}))
            if total_reward > 700:
                st.success(f"Episode complete: {total_reward:.1f} reward in {step_count} steps")
            else:
                st.info(f"Episode complete: {total_reward:.1f} reward in {step_count} steps")
        else:
            st.info("Select a checkpoint and click 'Run Episode' to watch the agent drive in real time.")


# ---- Footer ----
st.divider()
st.markdown(
    '<p style="text-align:center; color:#546e7a; font-size:0.85em;">'
    'Built from scratch with PyTorch &middot; No Stable-Baselines3 &middot; '
    '<a href="https://github.com/YOUR_USERNAME/carracing-ppo" '
    'style="color:#64b5f6;">GitHub</a></p>',
    unsafe_allow_html=True,
)
