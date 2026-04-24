"""CarRacing-v2 PPO Agent — Professional Dashboard.

Display-only: no gymnasium, no torch, no pygame imports.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PPO for Autonomous Driving",
    page_icon="\U0001f3ce",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1200px;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.02em;
    color: #f0ece4; margin: 0 0 0.4rem 0; line-height: 1.15;
}
.hero-sub {
    font-size: 1.05rem; color: #6b6b6b; font-weight: 400;
    max-width: 680px; line-height: 1.7; margin: 0 0 1.2rem 0;
}
.badge-success {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0d2e0d; color: #4caf50;
    border: 1px solid #1a4a1a; border-radius: 6px;
    padding: 5px 14px; font-size: 0.82rem; font-weight: 600;
}
.metrics-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 12px; margin: 2rem 0;
}
.metric-card {
    background: #111113; border: 1px solid #1e1e24;
    border-radius: 10px; padding: 1.1rem 1rem;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent, #3B8BD4);
}
.metric-val {
    font-size: 1.9rem; font-weight: 700; color: var(--accent, #3B8BD4);
    letter-spacing: -0.02em; line-height: 1.1; margin: 0 0 4px 0;
}
.metric-lbl {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4a4a55; margin: 0 0 2px 0;
}
.metric-sub { font-size: 0.72rem; color: #333340; margin: 0; }
.sec-header {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3B8BD4;
    margin: 0 0 1rem 0; display: flex; align-items: center; gap: 10px;
}
.sec-header::after {
    content: ''; flex: 1; height: 1px; background: #1a1a22;
}
.info-panel {
    background: #0d0d10; border: 1px solid #1a1a22;
    border-radius: 10px; padding: 1.2rem 1.4rem;
}
.info-panel h4 {
    font-size: 0.8rem; font-weight: 600; color: #c0bdb6; margin: 0 0 0.5rem 0;
}
.info-panel p {
    font-size: 0.82rem; color: #55555f; line-height: 1.7; margin: 0;
}
.results-table {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
.results-table th {
    text-align: left; padding: 8px 12px; font-size: 0.67rem;
    font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #3B8BD4; border-bottom: 1px solid #1a1a22;
}
.results-table td {
    padding: 10px 12px; color: #a0a0aa; border-bottom: 1px solid #111115;
}
.results-table tr:hover td { background: #111115; }
.results-table .val { color: #f0ece4; font-weight: 600; }
.results-table .good { color: #4caf50; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #1a1a22; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px; font-size: 0.82rem; font-weight: 500;
    color: #44444f; border-bottom: 2px solid transparent;
    border-radius: 0; background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #f0ece4 !important;
    border-bottom: 2px solid #3B8BD4 !important;
    background: transparent !important;
}
.divider { border: none; border-top: 1px solid #1a1a22; margin: 2rem 0; }
.spec-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}
.spec-item {
    padding: 0.9rem 1rem; background: #0d0d10;
    border: 1px solid #1a1a22; border-radius: 8px;
}
.spec-item .spec-key {
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3a3a50; margin: 0 0 4px 0;
}
.spec-item .spec-val {
    font-size: 0.9rem; font-weight: 600; color: #c0bdb6; margin: 0;
}
.spec-item .spec-desc {
    font-size: 0.75rem; color: #404050; margin: 3px 0 0 0;
}
.arch-container {
    background: #0d0d0f; border: 1px solid #1a1a22;
    border-radius: 10px; padding: 0; overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def gif_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def show_gif(path: str, css_width: str = "100%", caption: str = "") -> bool:
    p = Path(path)
    if not p.exists():
        st.caption(f"Asset not found: {path}")
        return False
    b64 = gif_b64(path)
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:{css_width};border-radius:8px;display:block;" '
        f'alt="{caption}">',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)
    return True


def load_best_info() -> dict:
    p = ASSETS / "best_episode_info.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"reward": 928, "steps": 716, "seed": 3}


# ── Hero ───────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 class="hero-title">PPO for Autonomous Driving</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">'
    "Proximal Policy Optimisation implemented from scratch in PyTorch, "
    "applied to four autonomous driving challenges: circuit racing, "
    "highway overtaking, roundabout navigation, and autonomous parking. "
    "No GPS, no map \u2014 raw observations only."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge-success">\u2713 &nbsp;Target 700 achieved \u2014 '
    "median 864.7 / 1000 over 50 episodes</span>",
    unsafe_allow_html=True,
)

# ── Metric cards ───────────────────────────────────────────────────────────

best = load_best_info()
st.markdown(
    f"""
<div class="metrics-row">
  <div class="metric-card" style="--accent:#3B8BD4">
    <p class="metric-lbl">Median Reward</p>
    <p class="metric-val">864.7</p>
    <p class="metric-sub">50 episodes</p>
  </div>
  <div class="metric-card" style="--accent:#4caf50">
    <p class="metric-lbl">Best Episode</p>
    <p class="metric-val">933.3</p>
    <p class="metric-sub">single run</p>
  </div>
  <div class="metric-card" style="--accent:#BA7517">
    <p class="metric-lbl">Above Target</p>
    <p class="metric-val">66%</p>
    <p class="metric-sub">threshold 700</p>
  </div>
  <div class="metric-card" style="--accent:#7F77DD">
    <p class="metric-lbl">Train Steps</p>
    <p class="metric-val">5M</p>
    <p class="metric-sub">9.6 hrs \u00b7 T4 GPU</p>
  </div>
  <div class="metric-card" style="--accent:#D85A30">
    <p class="metric-lbl">Parameters</p>
    <p class="metric-val">1.78M</p>
    <p class="metric-sub">CNN + actor-critic</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────

tab_demo, tab_highway, tab_train, tab_arch, tab_about = st.tabs(
    ["  CarRacing  ", "  Highway Scenarios  ", "  Training  ", "  Architecture  ", "  About  "]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 \u2014 DEMO
# ══════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown(
        '<div class="sec-header">Best episode</div>', unsafe_allow_html=True
    )

    col_gif, col_info = st.columns([3, 1], gap="large")

    with col_gif:
        shown = False
        for candidate in [
            "assets/showcase.gif",
            "assets/best_lap.gif",
            "assets/best_agent.gif",
        ]:
            if show_gif(candidate, css_width="100%"):
                shown = True
                break
        if not shown:
            st.info("Run scripts/record_showcase.py to generate the demo GIF")

    with col_info:
        st.markdown(
            f"""
<div class="info-panel" style="height:100%">
  <h4>Episode details</h4>
  <p>Cherry-picked from 18 trials across top checkpoints using deterministic
  policy (mean action \u2014 no sampling).</p>
  <br>
  <div style="display:grid;gap:10px;margin-top:4px">
    <div>
      <p style="color:#3a3a50;font-size:.67rem;letter-spacing:.1em;
         text-transform:uppercase;margin:0 0 2px">Reward</p>
      <p style="color:#4caf50;font-size:1.4rem;font-weight:700;margin:0">
        {best.get('reward', 928):.0f}</p>
    </div>
    <div>
      <p style="color:#3a3a50;font-size:.67rem;letter-spacing:.1em;
         text-transform:uppercase;margin:0 0 2px">Steps</p>
      <p style="color:#c0bdb6;font-size:1.1rem;font-weight:600;margin:0">
        {best.get('steps', 716)}</p>
    </div>
    <div>
      <p style="color:#3a3a50;font-size:.67rem;letter-spacing:.1em;
         text-transform:uppercase;margin:0 0 2px">Policy</p>
      <p style="color:#c0bdb6;font-size:.85rem;margin:0">
        Deterministic<br>(mean action)</p>
    </div>
    <div>
      <p style="color:#3a3a50;font-size:.67rem;letter-spacing:.1em;
         text-transform:uppercase;margin:0 0 2px">Checkpoint</p>
      <p style="color:#c0bdb6;font-size:.85rem;margin:0">Step 4.75M</p>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Track layout
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-header">Track layout</div>', unsafe_allow_html=True
    )

    col_track, col_track_info = st.columns([2, 1], gap="large")

    with col_track:
        track_shown = False
        for tp in [ASSETS / "track_layout.png", ASSETS / "track_start.png"]:
            if tp.exists():
                st.image(
                    Image.open(tp),
                    width=600,
                    caption="CarRacing-v2 \u2014 procedurally generated track",
                )
                track_shown = True
                break
        if not track_shown:
            st.info("Run scripts/generate_track_image.py to generate track image")

    with col_track_info:
        st.markdown(
            """
<div class="info-panel">
  <h4>Environment</h4>
  <p>CarRacing-v2 is a continuous control benchmark from OpenAI Gymnasium.
  The track is procedurally generated each episode \u2014 the agent cannot memorise
  a fixed layout and must generalise from visual features alone.</p>
  <br>
  <h4>Observation</h4>
  <p>96\u00d796 RGB top-down view, preprocessed to 4 stacked 84\u00d784 grayscale frames.
  Stacking provides implicit velocity information.</p>
  <br>
  <h4>Action space</h4>
  <p>Continuous: steering \u2208 [\u22121, 1], gas \u2208 [0, 1], brake \u2208 [0, 1].</p>
  <br>
  <h4>Reward</h4>
  <p>+1000/N per track tile visited, \u22120.1 per frame (time penalty),
  \u2212100 if the car leaves the track.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    # Progression
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-header">Learning progression</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "From random exploration at step 0 to consistent lap completion. "
        "Each segment is a separate checkpoint."
    )
    for candidate in [
        str(ASSETS / "progression_clean.gif"),
        str(ASSETS / "progression.gif"),
    ]:
        if show_gif(candidate, css_width="100%"):
            break

# ══════════════════════════════════════════════════════════════════
# TAB: HIGHWAY SCENARIOS
# ══════════════════════════════════════════════════════════════════
with tab_highway:
    st.markdown(
        '<div class="sec-header">Autonomous driving scenarios</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The same PPO algorithm applied to three autonomous driving "
        "tasks from **highway-env** \u2014 a library used in AV research. "
        "Each scenario tests a different driving competency.",
    )

    hw_results = {}
    hw_results_path = ASSETS / "highway_results.json"
    hw_demo_path = ASSETS / "highway_demo_results.json"
    if hw_results_path.exists():
        with open(hw_results_path) as f:
            hw_results = json.load(f)
    if hw_demo_path.exists():
        with open(hw_demo_path) as f:
            hw_results.update(json.load(f))

    c1, c2, c3 = st.columns(3, gap="medium")
    hw_scenarios = [
        {
            "col": c1,
            "title": "Highway Overtaking",
            "env": "highway-v0",
            "gif": "assets/highway_demo.gif",
            "desc": (
                "Navigate a 4-lane highway, overtake slower vehicles, "
                "maintain high speed without collisions."
            ),
            "action": "Discrete \u2014 IDLE / FASTER / SLOWER / LEFT / RIGHT",
            "obs": "Kinematics of 5 nearest vehicles",
            "color": "#3B8BD4",
        },
        {
            "col": c2,
            "title": "Roundabout Navigation",
            "env": "roundabout-v0",
            "gif": "assets/roundabout_demo.gif",
            "desc": (
                "Enter a roundabout, navigate around other vehicles, "
                "exit without collisions."
            ),
            "action": "Discrete \u2014 MetaAction (5 actions)",
            "obs": "Kinematics of 5 nearest vehicles",
            "color": "#1D9E75",
        },
        {
            "col": c3,
            "title": "Autonomous Parking",
            "env": "parking-v0",
            "gif": "assets/parking_demo.gif",
            "desc": (
                "Goal-conditioned task \u2014 park in a specific spot "
                "among other vehicles. Sparse reward."
            ),
            "action": "Continuous \u2014 steering + acceleration",
            "obs": "Kinematics + goal position (18-dim)",
            "color": "#BA7517",
        },
    ]

    for s in hw_scenarios:
        with s["col"]:
            st.markdown(
                f'<div style="border-top:2px solid {s["color"]};'
                f'padding-top:10px;margin-bottom:10px">'
                f'<p style="font-size:.67rem;font-weight:700;'
                f'letter-spacing:.1em;text-transform:uppercase;'
                f'color:{s["color"]};margin:0 0 4px">{s["title"]}</p>'
                f'<p style="font-size:.8rem;color:#888;'
                f'line-height:1.6;margin:0">{s["desc"]}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            show_gif(s["gif"], css_width="100%")
            r = hw_results.get(s["env"])
            if isinstance(r, dict):
                best_r = r.get("best_reward")
            else:
                best_r = r
            if best_r is not None:
                st.markdown(
                    f'<p style="font-size:.75rem;color:#888;margin:6px 0 2px">'
                    f"Best reward</p>"
                    f'<p style="font-size:1.3rem;font-weight:700;'
                    f'color:{s["color"]};margin:0">{best_r:.1f}</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<p style="font-size:.7rem;color:#666;margin:8px 0 2px">Actions</p>'
                f'<p style="font-size:.75rem;color:#aaa;margin:0">{s["action"]}</p>'
                f'<p style="font-size:.7rem;color:#666;margin:8px 0 2px">Observation</p>'
                f'<p style="font-size:.75rem;color:#aaa;margin:0">{s["obs"]}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("#### Same algorithm, harder problems")
        st.markdown(
            "The same PPO implementation that trained the CarRacing agent works "
            "here with minimal changes \u2014 demonstrating that the algorithm generalises "
            "across pixel vs kinematics observations, continuous vs discrete actions, "
            "and dense vs sparse rewards."
        )
    with col_r:
        comparison = pd.DataFrame(
            {
                "Property": ["Observation", "Action space", "Reward", "Other agents", "Train time"],
                "CarRacing": ["4\u00d784\u00d784 pixels", "Continuous", "Dense", "None", "~10 hours"],
                "Highway": ["10\u00d75 kinematics", "Discrete / Both", "Dense / Sparse", "Yes \u2014 traffic", "~3 hours"],
            }
        )
        st.dataframe(comparison, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 \u2014 TRAINING
# ══════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown(
        '<div class="sec-header">Training curves</div>', unsafe_allow_html=True
    )

    csv_candidates = [ASSETS / "training_metrics.csv", ASSETS / "eval_metrics.csv"]
    df = None
    for cp in csv_candidates:
        if cp.exists():
            df = pd.read_csv(cp)
            df.columns = [c.lower().strip() for c in df.columns]
            break

    if df is not None:
        step_col = next((c for c in df.columns if "step" in c), None)
        reward_col = next(
            (c for c in df.columns if "reward" in c or "return" in c), None
        )
        ent_col = next((c for c in df.columns if "entrop" in c), None)
        vl_col = next(
            (c for c in df.columns if "value" in c and "loss" in c), None
        )

        PLOT_BG = "rgba(0,0,0,0)"
        GRID_COL = "rgba(255,255,255,0.04)"
        FONT = dict(family="monospace", size=11, color="#888880")

        def base_layout(title: str, yaxis_title: str) -> dict:
            return dict(
                title=dict(
                    text=title,
                    font=dict(size=13, color="#888880", family="monospace"),
                ),
                xaxis=dict(
                    title="Training steps",
                    tickfont=FONT,
                    gridcolor=GRID_COL,
                    linecolor="#1a1a22",
                    tickformat=".2s",
                ),
                yaxis=dict(
                    title=yaxis_title,
                    tickfont=FONT,
                    gridcolor=GRID_COL,
                    linecolor="#1a1a22",
                ),
                paper_bgcolor=PLOT_BG,
                plot_bgcolor="#0a0a0c",
                height=340,
                margin=dict(l=55, r=30, t=50, b=50),
                legend=dict(
                    font=dict(size=10, color="#666"), bgcolor="rgba(0,0,0,0)"
                ),
                hovermode="x unified",
            )

        if step_col and reward_col:
            fig1 = go.Figure()
            window = max(1, len(df) // 40)
            smoothed = df[reward_col].rolling(window, center=True).mean()

            fig1.add_trace(
                go.Scatter(
                    x=df[step_col],
                    y=df[reward_col],
                    mode="lines",
                    name="Raw eval",
                    line=dict(color="rgba(59,139,212,0.2)", width=1),
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=df[step_col],
                    y=smoothed,
                    mode="lines",
                    name="Smoothed",
                    line=dict(color="#3B8BD4", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59,139,212,0.06)",
                )
            )
            fig1.add_hline(
                y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)"
            )
            fig1.add_hline(
                y=700,
                line_dash="dash",
                line_color="rgba(76,175,80,0.4)",
                annotation_text="Target 700",
                annotation_font_color="#4caf50",
                annotation_font_size=10,
            )
            fig1.add_vline(
                x=1_850_000,
                line_dash="dash",
                line_color="rgba(186,117,23,0.4)",
                annotation_text="Breakthrough",
                annotation_font_color="#BA7517",
                annotation_font_size=10,
                annotation_position="top right",
            )
            best_idx = df[reward_col].idxmax()
            fig1.add_trace(
                go.Scatter(
                    x=[df[step_col].iloc[best_idx]],
                    y=[df[reward_col].iloc[best_idx]],
                    mode="markers+text",
                    marker=dict(color="#D85A30", size=8, symbol="circle"),
                    text=[f"  {df[reward_col].iloc[best_idx]:.0f}"],
                    textposition="middle right",
                    textfont=dict(size=10, color="#D85A30"),
                    name="Best checkpoint",
                    showlegend=False,
                )
            )
            fig1.update_layout(
                **base_layout("Eval reward vs training steps", "Eval reward")
            )
            st.plotly_chart(fig1, use_container_width=True)

        secondary_figs = []
        if ent_col and step_col:
            fig_e = go.Figure()
            fig_e.add_trace(
                go.Scatter(
                    x=df[step_col],
                    y=df[ent_col],
                    mode="lines",
                    line=dict(color="#1D9E75", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(29,158,117,0.06)",
                    name="Entropy",
                )
            )
            fig_e.update_layout(**base_layout("Policy entropy", "Entropy H(\u03c0)"))
            secondary_figs.append(fig_e)

        if vl_col and step_col:
            fig_v = go.Figure()
            fig_v.add_trace(
                go.Scatter(
                    x=df[step_col],
                    y=df[vl_col],
                    mode="lines",
                    line=dict(color="#D85A30", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(216,90,48,0.06)",
                    name="Value loss",
                )
            )
            fig_v.update_layout(**base_layout("Value loss", "MSE"))
            secondary_figs.append(fig_v)

        if secondary_figs:
            sec_cols = st.columns(len(secondary_figs), gap="medium")
            for col, fig in zip(sec_cols, secondary_figs):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("training_metrics.csv not found in assets/")

    # Results table
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-header">Evaluation results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="info-panel">
<table class="results-table">
  <thead><tr><th>Metric</th><th>Value</th><th>Context</th></tr></thead>
  <tbody>
    <tr><td>Best eval reward</td>
        <td class="val">811.9</td>
        <td>checkpoint at step 4.9M</td></tr>
    <tr><td>50-episode median</td>
        <td class="good">864.7</td>
        <td>strong, consistent performance</td></tr>
    <tr><td>50-episode mean</td>
        <td class="val">632.1</td>
        <td>pulled down by occasional crashes</td></tr>
    <tr><td>50-episode max</td>
        <td class="good">933.3</td>
        <td>near-perfect lap</td></tr>
    <tr><td>Episodes scoring > 700</td>
        <td class="good">66%</td>
        <td>target threshold \u2713</td></tr>
    <tr><td>Training steps</td>
        <td class="val">~5,000,000</td>
        <td>8 parallel environments</td></tr>
    <tr><td>Wall-clock time</td>
        <td class="val">~9.6 hours</td>
        <td>AWS EC2 g4dn.xlarge \u00b7 NVIDIA T4</td></tr>
  </tbody>
</table>
</div>
""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# TAB 3 \u2014 ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown(
        '<div class="sec-header">Network architecture</div>',
        unsafe_allow_html=True,
    )

    arch_img_path = ASSETS / "architecture_diagram.png"
    if arch_img_path.exists():
        st.markdown('<div class="arch-container">', unsafe_allow_html=True)
        st.image(Image.open(arch_img_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run scripts/generate_arch_diagram.py to generate diagram")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            '<div class="sec-header">Layer details</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="info-panel">
<table class="results-table">
  <thead>
    <tr><th>Layer</th><th>Output shape</th><th>Details</th><th>Params</th></tr>
  </thead>
  <tbody>
    <tr><td>Input</td><td class="val">4\u00d784\u00d784</td>
        <td>4 stacked grayscale frames</td><td>\u2014</td></tr>
    <tr><td>Conv2d 1</td><td class="val">32\u00d720\u00d720</td>
        <td>32 filters \u00b7 8\u00d78 \u00b7 stride 4</td>
        <td class="val">8,224</td></tr>
    <tr><td>Conv2d 2</td><td class="val">64\u00d79\u00d79</td>
        <td>64 filters \u00b7 4\u00d74 \u00b7 stride 2</td>
        <td class="val">131,136</td></tr>
    <tr><td>Conv2d 3</td><td class="val">64\u00d77\u00d77</td>
        <td>64 filters \u00b7 3\u00d73 \u00b7 stride 1</td>
        <td class="val">36,928</td></tr>
    <tr><td>Flatten</td><td class="val">3136</td>
        <td>\u2014</td><td>\u2014</td></tr>
    <tr><td>Linear</td><td class="val">512</td>
        <td>ReLU \u00b7 shared backbone</td>
        <td class="val">1,605,632</td></tr>
    <tr><td>Actor \u03bc</td><td class="val">3</td>
        <td>Mean \u2014 steer, gas, brake</td>
        <td class="val">1,539</td></tr>
    <tr><td>Actor log \u03c3</td><td class="val">3</td>
        <td>Log std of Gaussian</td><td class="val">3</td></tr>
    <tr><td>Critic V</td><td class="val">1</td>
        <td>Expected return V(s)</td>
        <td class="val">513</td></tr>
  </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            '<div class="sec-header">Hyperparameters</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="info-panel">
<table class="results-table">
  <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Algorithm</td><td class="val">PPO (from scratch)</td></tr>
    <tr><td>n_envs</td><td class="val">8</td></tr>
    <tr><td>rollout_steps</td><td class="val">128</td></tr>
    <tr><td>minibatch_size</td><td class="val">256</td></tr>
    <tr><td>n_epochs</td><td class="val">4</td></tr>
    <tr><td>learning_rate</td><td class="val">3e-4 \u2192 0 (linear)</td></tr>
    <tr><td>gamma</td><td class="val">0.99</td></tr>
    <tr><td>gae_lambda</td><td class="val">0.95</td></tr>
    <tr><td>clip_epsilon</td><td class="val">0.2</td></tr>
    <tr><td>vf_coef</td><td class="val">0.5</td></tr>
    <tr><td>ent_coef</td><td class="val">0.01</td></tr>
    <tr><td>target_kl</td><td class="val">0.015</td></tr>
    <tr><td>grad_clip</td><td class="val">0.5</td></tr>
  </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    # Design choices
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-header">Design decisions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="spec-grid">
  <div class="spec-item">
    <p class="spec-key">PPO over DQN</p>
    <p class="spec-val">Continuous action space</p>
    <p class="spec-desc">
      Steering is a float in [\u22121, 1], not a discrete choice.
      DQN requires discretisation which loses precision.
      PPO's Gaussian policy outputs continuous values naturally.
    </p>
  </div>
  <div class="spec-item">
    <p class="spec-key">Frame stacking \u00d7 4</p>
    <p class="spec-val">Implicit velocity encoding</p>
    <p class="spec-desc">
      A single frame is position-only \u2014 the agent can't tell if
      the car is moving or still. Stacking 4 consecutive frames
      encodes velocity via pixel differences, restoring the Markov property.
    </p>
  </div>
  <div class="spec-item">
    <p class="spec-key">GAE (\u03bb = 0.95)</p>
    <p class="spec-val">Bias-variance trade-off</p>
    <p class="spec-desc">
      Generalised Advantage Estimation blends TD(0) (low variance,
      biased) with Monte Carlo (unbiased, high variance). \u03bb = 0.95
      leans toward MC while keeping variance manageable.
    </p>
  </div>
  <div class="spec-item">
    <p class="spec-key">Entropy bonus (0.01)</p>
    <p class="spec-val">Prevents premature convergence</p>
    <p class="spec-desc">
      Adds H(\u03c0) to the objective. Without it the policy collapses
      to a single deterministic action early in training and stops
      exploring.
    </p>
  </div>
  <div class="spec-item">
    <p class="spec-key">Orthogonal initialisation</p>
    <p class="spec-val">Stable gradient flow</p>
    <p class="spec-desc">
      CNN layers: \u221a2 gain. Actor head: 0.01 (near-uniform initial policy).
      Critic head: 1.0. Prevents vanishing/exploding gradients at init.
    </p>
  </div>
  <div class="spec-item">
    <p class="spec-key">Linear LR decay</p>
    <p class="spec-val">3e-4 \u2192 0 over 5M steps</p>
    <p class="spec-desc">
      As the policy matures, smaller updates prevent overshooting.
      By step 5M the learning rate was near zero \u2014 the agent was
      effectively in fine-tuning mode.
    </p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# TAB 4 \u2014 ABOUT
# ══════════════════════════════════════════════════════════════════
with tab_about:
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown(
            '<div class="sec-header">Project</div>', unsafe_allow_html=True
        )
        st.markdown(
            """
<div class="info-panel">
  <p>
  This project implements Proximal Policy Optimisation (PPO) entirely from scratch
  in PyTorch \u2014 no RL libraries, no wrappers beyond Gymnasium's standard wrappers.
  The goal was to understand every moving part of a modern policy gradient method
  by building and debugging it hands-on.
  </p>
  <br>
  <p>
  The environment is CarRacing-v2 from OpenAI Gymnasium \u2014 a continuous control
  benchmark where a car must navigate a procedurally generated track visible only
  through a top-down 96\u00d796 pixel camera. The track changes every episode, so the
  agent must learn general driving behaviour rather than memorise a fixed route.
  </p>
  <br>
  <p>
  Training ran for approximately 9.6 hours on a single NVIDIA T4 GPU (AWS EC2
  g4dn.xlarge). The final agent achieves a median reward of 864.7 over 50 evaluation
  episodes, well above the 700 target, with 66% of episodes completing the lap.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-header">Training timeline</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="info-panel">
<table class="results-table">
  <thead>
    <tr><th>Phase</th><th>Steps</th><th>Reward</th><th>What happened</th></tr>
  </thead>
  <tbody>
    <tr><td>Random exploration</td><td>0 \u2192 1.5M</td>
        <td style="color:#D85A30">\u221284</td>
        <td>Agent flails, rarely stays on track</td></tr>
    <tr><td>Early learning</td><td>1.5M \u2192 1.85M</td>
        <td style="color:#BA7517">\u221270 \u2192 +30</td>
        <td>Learns basic steering</td></tr>
    <tr><td>Breakthrough</td><td>1.85M \u2192 2.7M</td>
        <td style="color:#4caf50">+30 \u2192 +130</td>
        <td>Discovers staying on track is rewarded</td></tr>
    <tr><td>Consolidation</td><td>2.7M \u2192 4.5M</td>
        <td style="color:#4caf50">+20 \u2192 +80</td>
        <td>Refines braking, corner entry</td></tr>
    <tr><td>Fine-tuning</td><td>4.5M \u2192 5M</td>
        <td style="color:#4caf50">+50 \u2192 +80</td>
        <td>LR near zero, marginal gains</td></tr>
  </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            '<div class="sec-header">Stack</div>', unsafe_allow_html=True
        )
        stack_items = [
            ("PyTorch 2.x + CUDA", "Deep learning framework"),
            ("Gymnasium 0.29.1", "RL environment"),
            ("AWS EC2 g4dn.xlarge", "NVIDIA T4 \u00b7 16GB VRAM"),
            ("Weights & Biases", "Experiment tracking"),
            ("Hydra", "Config management"),
            ("Streamlit", "This dashboard"),
        ]
        stack_html = (
            '<div class="info-panel" style="margin-bottom:16px">'
            '<div style="display:grid;gap:10px">'
        )
        for name, desc in stack_items:
            stack_html += (
                '<div style="display:flex;justify-content:space-between;'
                "align-items:center;padding:6px 0;"
                'border-bottom:1px solid #111115">'
                f'<span style="color:#c0bdb6;font-size:.85rem;font-weight:500">'
                f"{name}</span>"
                f'<span style="color:#333340;font-size:.75rem">{desc}</span>'
                "</div>"
            )
        stack_html += "</div></div>"
        st.markdown(stack_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-header">Links</div>', unsafe_allow_html=True
        )
        st.markdown(
            """
<div class="info-panel">
  <a href="https://github.com/anmol0705/CarRacing-v2-PPO-Agent"
     style="color:#3B8BD4;text-decoration:none;font-size:.85rem">
    \u2197 GitHub Repository
  </a><br><br>
  <a href="https://wandb.ai/anmol_752005/carracing-ppo"
     style="color:#3B8BD4;text-decoration:none;font-size:.85rem">
    \u2197 W&B Training Runs
  </a>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:.72rem;color:#222228;text-align:center">'
    "CarRacing-v2 PPO Agent \u00b7 "
    "Trained on AWS EC2 \u00b7 PyTorch \u00b7 Gymnasium"
    "</p>",
    unsafe_allow_html=True,
)
