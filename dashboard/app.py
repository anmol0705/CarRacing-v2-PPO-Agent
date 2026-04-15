"""Streamlit dashboard for CarRacing PPO agent replay and monitoring."""

import glob
import sys
import os
import time

import numpy as np
import streamlit as st
import torch

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env_utils import make_env
from src.model import ActorCritic


st.set_page_config(page_title="CarRacing PPO", layout="wide")
st.title("CarRacing-v2 PPO Agent")

# --- Sidebar: checkpoint selector ---
st.sidebar.header("Checkpoint")
checkpoint_files = sorted(glob.glob("checkpoints/*.pt"))
if not checkpoint_files:
    st.sidebar.warning("No checkpoints found in checkpoints/")
    st.stop()

selected_ckpt = st.sidebar.selectbox("Select checkpoint", checkpoint_files)
run_episode = st.sidebar.button("Run episode")

# --- Load model ---
@st.cache_resource
def load_model(ckpt_path: str) -> ActorCritic:
    """Load a model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if run_episode:
    model = load_model(selected_ckpt)
    device = next(model.parameters()).device

    env_fn = make_env(seed=42, render_mode="rgb_array")
    env = env_fn()

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    rewards_over_time: list[float] = []

    # Layout
    col_main, col_controls = st.columns([3, 1])
    frame_placeholder = col_main.empty()
    steer_bar = col_controls.empty()
    gas_bar = col_controls.empty()
    brake_bar = col_controls.empty()
    reward_text = col_controls.empty()
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

        # Render frame
        frame = env.render()
        if frame is not None:
            frame_placeholder.image(frame, caption=f"Step {step_count}", channels="RGB")

        # Action bars (map to 0-1 range for progress bars)
        steer_val = (float(action_np[0]) + 1.0) / 2.0  # [-1,1] -> [0,1]
        gas_val = float(np.clip(action_np[1], 0, 1))
        brake_val = float(np.clip(action_np[2], 0, 1))

        steer_bar.metric("Steer", f"{action_np[0]:.2f}")
        gas_bar.metric("Gas", f"{action_np[1]:.2f}")
        brake_bar.metric("Brake", f"{action_np[2]:.2f}")
        reward_text.metric("Total Reward", f"{total_reward:.1f}")

        time.sleep(1.0 / 20.0)  # ~20 fps

    env.close()

    # Final reward chart
    st.subheader("Cumulative Reward")
    st.line_chart(rewards_over_time)
    st.success(f"Episode complete: {total_reward:.1f} reward in {step_count} steps")
else:
    st.info("Select a checkpoint and click 'Run episode' to watch the agent drive.")
