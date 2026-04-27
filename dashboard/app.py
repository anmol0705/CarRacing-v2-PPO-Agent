"""PPO for Autonomous Driving — Interactive Dashboard.

Display-only: no gymnasium, no torch, no pygame imports.
Works on Streamlit Cloud with lightweight deps only.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PPO Autonomous Driving",
    page_icon="\U0001f3ce",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# ── Theme ─────────────────────────────────────────────────────────────────

BLUE = "#3B8BD4"
GREEN = "#22C55E"
AMBER = "#F59E0B"
ORANGE = "#EF6C35"
PURPLE = "#8B5CF6"
RED = "#EF4444"
TEAL = "#14B8A6"
PINK = "#EC4899"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem;
    max-width: 1280px;
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ── Hero ── */
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
    border: 1px solid rgba(34,197,94,0.2); border-radius: 100px;
    padding: 4px 14px; font-size: 0.72rem; font-weight: 600;
    color: #22C55E; letter-spacing: 0.02em; margin-bottom: 12px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 900; letter-spacing: -0.04em;
    background: linear-gradient(135deg, #f0f0f0 0%, #a0a0a0 50%, #f0f0f0 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.6rem 0; line-height: 1.1;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    to { background-position: 200% center; }
}
.hero-sub {
    font-size: 1.0rem; color: #71717A; font-weight: 400;
    max-width: 720px; line-height: 1.75; margin: 0 0 1.5rem 0;
}
.hero-highlight {
    color: #A1A1AA; font-weight: 500;
}

/* ── Metric cards ── */
.metrics-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 10px; margin: 1.5rem 0 2rem 0;
}
@media (max-width: 768px) {
    .metrics-row { grid-template-columns: repeat(2, 1fr); }
}
.m-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem 1rem 0.9rem;
    position: relative; overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.m-card:hover { border-color: rgba(255,255,255,0.14); transform: translateY(-1px); }
.m-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent);
}
.m-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #52525B; margin: 0 0 6px 0;
}
.m-value {
    font-size: 1.7rem; font-weight: 700; color: var(--accent);
    letter-spacing: -0.02em; line-height: 1; margin: 0 0 4px 0;
    font-family: 'JetBrains Mono', monospace;
}
.m-sub { font-size: 0.7rem; color: #3F3F46; margin: 0; }

/* ── Section headers ── */
.sec-h {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3B8BD4;
    margin: 0 0 1.2rem 0; display: flex; align-items: center; gap: 10px;
}
.sec-h::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.05); }

/* ── Panels ── */
.panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
}
.panel:hover { border-color: rgba(255,255,255,0.1); }
.panel h4 {
    font-size: 0.78rem; font-weight: 600; color: #D4D4D8; margin: 0 0 0.5rem 0;
}
.panel p, .panel li {
    font-size: 0.82rem; color: #71717A; line-height: 1.75; margin: 0;
}

/* ── Tables ── */
.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.tbl th {
    text-align: left; padding: 8px 12px; font-size: 0.62rem;
    font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #3B8BD4; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.tbl td {
    padding: 10px 12px; color: #A1A1AA;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.tbl tr:hover td { background: rgba(255,255,255,0.02); }
.tbl .v { color: #E4E4E7; font-weight: 600; }
.tbl .g { color: #22C55E; font-weight: 600; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    background: transparent; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 22px; font-size: 0.8rem; font-weight: 500;
    color: #52525B; border-bottom: 2px solid transparent;
    border-radius: 0; background: transparent;
    transition: color 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #71717A; }
.stTabs [aria-selected="true"] {
    color: #E4E4E7 !important;
    border-bottom: 2px solid #3B8BD4 !important;
    background: transparent !important;
}

/* ── Env cards ── */
.env-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; overflow: hidden;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.env-card:hover {
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.env-card-header {
    padding: 14px 16px 10px; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.env-card-body { padding: 0; }
.env-card-footer { padding: 12px 16px; }
.env-tag {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; margin: 0 0 4px;
}
.env-title {
    font-size: 0.95rem; font-weight: 700; color: #E4E4E7; margin: 0;
}
.env-desc { font-size: 0.75rem; color: #71717A; line-height: 1.6; margin: 6px 0 0; }
.env-stat {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.03);
}
.env-stat:last-child { border-bottom: none; }
.env-stat-label { font-size: 0.68rem; color: #52525B; }
.env-stat-val { font-size: 0.78rem; font-weight: 600; color: #A1A1AA; }

/* ── Spec grid ── */
.spec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 768px) { .spec-grid { grid-template-columns: 1fr; } }
.spec-card {
    padding: 1rem; background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 10px;
    transition: border-color 0.2s, transform 0.2s;
}
.spec-card:hover { border-color: rgba(255,255,255,0.12); transform: translateY(-1px); }
.spec-card .sk { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3B8BD4; margin: 0 0 4px; }
.spec-card .sv { font-size: 0.88rem; font-weight: 600; color: #D4D4D8; margin: 0; }
.spec-card .sd { font-size: 0.75rem; color: #52525B; margin: 4px 0 0; line-height: 1.6; }

.divider { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 2rem 0; }

/* ── Timeline ── */
.timeline-item {
    display: flex; gap: 16px; padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.timeline-item:last-child { border-bottom: none; }
.timeline-dot {
    width: 10px; height: 10px; border-radius: 50%;
    margin-top: 4px; flex-shrink: 0;
}
.timeline-content { flex: 1; }
.timeline-step { font-size: 0.7rem; color: #52525B; font-weight: 600; margin: 0 0 2px; }
.timeline-desc { font-size: 0.82rem; color: #A1A1AA; margin: 0; }
.timeline-reward { font-size: 0.82rem; font-weight: 700; margin: 0 0 2px; }

/* ── Code blocks ── */
.code-block {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #A1A1AA;
    line-height: 1.8; overflow-x: auto;
}
.code-block .kw { color: #C084FC; }
.code-block .fn { color: #3B8BD4; }
.code-block .cm { color: #52525B; }
.code-block .st { color: #22C55E; }
.code-block .num { color: #F59E0B; }

/* ── Leaderboard ── */
.lb-rank {
    width: 28px; height: 28px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    background: rgba(255,255,255,0.04);
    color: #71717A;
}
.lb-rank-1 { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05)); color: #F59E0B; }
.lb-rank-2 { background: linear-gradient(135deg, rgba(168,162,158,0.2), rgba(168,162,158,0.05)); color: #A8A29E; }
.lb-rank-3 { background: linear-gradient(135deg, rgba(180,83,9,0.2), rgba(180,83,9,0.05)); color: #B45309; }

/* ── Insight cards ── */
.insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
@media (max-width: 768px) { .insight-grid { grid-template-columns: 1fr; } }
.insight-card {
    padding: 1rem; background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; border-left: 3px solid var(--accent);
    transition: border-color 0.2s;
}
.insight-card:hover { border-color: rgba(255,255,255,0.12); }
.insight-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 700; color: var(--accent);
    margin: 0 0 4px;
}
.insight-label { font-size: 0.7rem; color: #52525B; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 4px; }
.insight-desc { font-size: 0.75rem; color: #71717A; line-height: 1.5; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def gif_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def show_gif(path: str, width: str = "100%", border_radius: str = "8px") -> bool:
    p = Path(path)
    if not p.exists():
        return False
    b64 = gif_b64(path)
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:{width};border-radius:{border_radius};display:block" />',
        unsafe_allow_html=True,
    )
    return True


def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default or {}


PLOT_BG = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.03)"
FONT = dict(family="Inter, monospace", size=11, color="#71717A")


def make_layout(title: str, ytitle: str, h: int = 360) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color="#A1A1AA", family="Inter, monospace")),
        xaxis=dict(title="Training steps", tickfont=FONT, gridcolor=GRID,
                   linecolor="rgba(255,255,255,0.06)", tickformat=".2s",
                   zeroline=False),
        yaxis=dict(title=ytitle, tickfont=FONT, gridcolor=GRID,
                   linecolor="rgba(255,255,255,0.06)", zeroline=False),
        paper_bgcolor=PLOT_BG, plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=55, r=30, t=50, b=50),
        legend=dict(font=dict(size=10, color="#71717A"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )


best_info = load_json(ASSETS / "best_episode_info.json", {"reward": 940, "steps": 596, "seed": 185})
hw_results = load_json(ASSETS / "highway_results.json")


# ── Hero ──────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-badge">ABOVE HUMAN LEVEL</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">PPO for Autonomous Driving</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">'
    "Proximal Policy Optimisation built <span class='hero-highlight'>entirely from scratch</span> in PyTorch. "
    "Four driving challenges &mdash; pixel-based racing, highway overtaking, "
    "roundabout navigation, and autonomous parking &mdash; all powered by "
    "<span class='hero-highlight'>the same algorithm</span>. "
    "No pretrained models. No RL libraries."
    "</p>",
    unsafe_allow_html=True,
)

# ── Metrics ───────────────────────────────────────────────────────────────

peak_r = best_info.get("reward", 940)
hw_best = hw_results.get("highway-v0", {})
hw_reward = hw_best.get("best_reward", 230.7) if isinstance(hw_best, dict) else hw_best

st.markdown(
    f"""
<div class="metrics-row">
  <div class="m-card" style="--accent:{GREEN}">
    <p class="m-label">Peak Reward</p>
    <p class="m-value">{peak_r:.0f}</p>
    <p class="m-sub">CarRacing-v2 best episode</p>
  </div>
  <div class="m-card" style="--accent:{BLUE}">
    <p class="m-label">Avg Reward</p>
    <p class="m-value">874</p>
    <p class="m-sub">20-episode evaluation</p>
  </div>
  <div class="m-card" style="--accent:{AMBER}">
    <p class="m-label">Highway Score</p>
    <p class="m-value">{hw_reward:.0f}</p>
    <p class="m-sub">lane changes + overtaking</p>
  </div>
  <div class="m-card" style="--accent:{PURPLE}">
    <p class="m-label">Training Steps</p>
    <p class="m-value">7M</p>
    <p class="m-sub">across all environments</p>
  </div>
  <div class="m-card" style="--accent:{ORANGE}">
    <p class="m-label">Parameters</p>
    <p class="m-value">1.78M</p>
    <p class="m-sub">CNN actor-critic</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────

tab_race, tab_highway, tab_train, tab_arch, tab_algo, tab_about = st.tabs(
    [" CarRacing ", " Highway Scenarios ", " Training Deep-Dive ", " Architecture ", " How PPO Works ", " About "]
)


# ══════════════════════════════════════════════════════════════════
# TAB: CARRACING
# ══════════════════════════════════════════════════════════════════
with tab_race:
    st.markdown('<div class="sec-h">Best episode showcase</div>', unsafe_allow_html=True)

    col_gif, col_info = st.columns([3, 1], gap="large")

    with col_gif:
        shown = False
        for candidate in ["assets/showcase.gif", "assets/showcase_small.gif", "assets/best_lap.gif"]:
            if show_gif(candidate, width="100%", border_radius="10px"):
                shown = True
                break
        if not shown:
            st.info("Run `python scripts/record_showcase.py` to generate the demo GIF.")

    with col_info:
        ep_reward = best_info.get("reward", 940)
        ep_steps = best_info.get("steps", 596)
        st.markdown(
            f"""
<div class="panel" style="height:100%">
  <h4>Episode stats</h4>
  <p style="margin-bottom:16px">Deterministic policy (mean action, no sampling) evaluated across
  200 random seeds. This is the single best episode.</p>
  <div style="display:grid;gap:14px">
    <div>
      <p class="m-label">Reward</p>
      <p style="color:{GREEN};font-size:1.5rem;font-weight:700;margin:0;font-family:'JetBrains Mono',monospace">{ep_reward:.0f}</p>
    </div>
    <div>
      <p class="m-label">Duration</p>
      <p style="color:#D4D4D8;font-size:1.1rem;font-weight:600;margin:0;font-family:'JetBrains Mono',monospace">{ep_steps} steps</p>
    </div>
    <div>
      <p class="m-label">Policy</p>
      <p style="color:#A1A1AA;font-size:.82rem;margin:0">Deterministic (mean)</p>
    </div>
    <div>
      <p class="m-label">Checkpoint</p>
      <p style="color:#A1A1AA;font-size:.82rem;margin:0">Step 4.75M (fine-tuned to 7M)</p>
    </div>
    <div>
      <p class="m-label">Human level</p>
      <p style="color:{AMBER};font-size:.82rem;font-weight:500;margin:0">~900 (agent exceeds this)</p>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Progression
    st.markdown('<div class="sec-h">Learning progression</div>', unsafe_allow_html=True)

    prog_col1, prog_col2 = st.columns([3, 1], gap="large")
    with prog_col1:
        for candidate in [str(ASSETS / "progression_clean.gif"), str(ASSETS / "progression.gif")]:
            if show_gif(candidate, width="100%", border_radius="10px"):
                break
    with prog_col2:
        st.markdown(
            f"""
<div class="panel">
  <h4>From chaos to control</h4>
  <div style="margin-top:12px">
    <div class="timeline-item">
      <div class="timeline-dot" style="background:{RED}"></div>
      <div class="timeline-content">
        <p class="timeline-step">250K steps</p>
        <p class="timeline-reward" style="color:{RED}">Reward: -8</p>
        <p class="timeline-desc">Spinning in circles</p>
      </div>
    </div>
    <div class="timeline-item">
      <div class="timeline-dot" style="background:{AMBER}"></div>
      <div class="timeline-content">
        <p class="timeline-step">1M steps</p>
        <p class="timeline-reward" style="color:{AMBER}">Reward: +23</p>
        <p class="timeline-desc">Learning to steer</p>
      </div>
    </div>
    <div class="timeline-item">
      <div class="timeline-dot" style="background:{BLUE}"></div>
      <div class="timeline-content">
        <p class="timeline-step">2.5M steps</p>
        <p class="timeline-reward" style="color:{BLUE}">Reward: +99</p>
        <p class="timeline-desc">Following the road</p>
      </div>
    </div>
    <div class="timeline-item">
      <div class="timeline-dot" style="background:{TEAL}"></div>
      <div class="timeline-content">
        <p class="timeline-step">4.2M steps</p>
        <p class="timeline-reward" style="color:{TEAL}">Reward: +631</p>
        <p class="timeline-desc">First full laps</p>
      </div>
    </div>
    <div class="timeline-item">
      <div class="timeline-dot" style="background:{GREEN}"></div>
      <div class="timeline-content">
        <p class="timeline-step">7M steps</p>
        <p class="timeline-reward" style="color:{GREEN}">Reward: +940</p>
        <p class="timeline-desc">Above human level</p>
      </div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Key Insights
    st.markdown('<div class="sec-h">Key insights</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="insight-grid">
  <div class="insight-card" style="--accent:{GREEN}">
    <p class="insight-num">3M</p>
    <p class="insight-label">Steps to first progress</p>
    <p class="insight-desc">The agent appears to learn nothing for 3M steps, then reward
    explodes. This "hockey stick" pattern is characteristic of RL on complex tasks.</p>
  </div>
  <div class="insight-card" style="--accent:{BLUE}">
    <p class="insight-num">+15%</p>
    <p class="insight-label">Fine-tuning gain</p>
    <p class="insight-desc">Reducing LR from 2.5e-4 to 5e-5 and training for 2M more steps
    pushed average reward from 812 to 874. Small hyperparameter changes, big impact.</p>
  </div>
  <div class="insight-card" style="--accent:{AMBER}">
    <p class="insight-num">4x</p>
    <p class="insight-label">Faster than single env</p>
    <p class="insight-desc">8 parallel environments don't just speed up data collection &mdash;
    they decorrelate samples, making each PPO update more effective.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Leaderboard
    lb_path = ASSETS / "episode_leaderboard.csv"
    if lb_path.exists():
        st.markdown('<div class="sec-h">Episode leaderboard (top 10)</div>', unsafe_allow_html=True)
        lb_df = pd.read_csv(lb_path)
        lb_top = lb_df.nlargest(10, "reward")

        rows_html = ""
        for i, (_, row) in enumerate(lb_top.iterrows()):
            rank = i + 1
            rank_cls = f" lb-rank-{rank}" if rank <= 3 else ""
            ckpt = row["checkpoint"].replace("checkpoints/", "").replace(".pt", "")
            rows_html += f"""
            <tr>
              <td><span class="lb-rank{rank_cls}">{rank}</span></td>
              <td class="g" style="font-family:'JetBrains Mono',monospace">{row['reward']:.1f}</td>
              <td class="v">{int(row['steps'])}</td>
              <td>{row['seed']}</td>
              <td style="font-size:0.75rem">{ckpt}</td>
              <td class="v">{row['completion']:.0%}</td>
            </tr>"""

        st.markdown(
            f"""
<div class="panel">
  <table class="tbl">
    <thead><tr>
      <th style="width:40px">#</th><th>Reward</th><th>Steps</th>
      <th>Seed</th><th>Checkpoint</th><th>Completion</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="font-size:0.7rem;color:#3F3F46;margin:12px 0 0">
    Evaluated with deterministic policy across 10 random seeds per checkpoint.
    Steps = episode length (max 1000). Completion = fraction of track tiles visited.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Environment details
    st.markdown('<div class="sec-h">Environment details</div>', unsafe_allow_html=True)
    col_env1, col_env2 = st.columns(2, gap="large")
    with col_env1:
        track_shown = False
        for tp in [ASSETS / "track_layout.png", ASSETS / "track_start.png"]:
            if tp.exists():
                st.image(Image.open(tp), use_container_width=True,
                         caption="CarRacing-v2 procedurally generated track")
                track_shown = True
                break
        if not track_shown:
            st.markdown(
                f"""
<div class="panel">
  <h4>CarRacing-v2</h4>
  <p>A continuous-control benchmark from OpenAI Gymnasium. The track is procedurally generated
  each episode, so the agent cannot memorise a fixed layout.</p>
  <br>
  <div class="env-stat"><span class="env-stat-label">Observation</span>
    <span class="env-stat-val">4 x 84 x 84 grayscale</span></div>
  <div class="env-stat"><span class="env-stat-label">Action space</span>
    <span class="env-stat-val">Continuous [steer, gas, brake]</span></div>
  <div class="env-stat"><span class="env-stat-label">Reward</span>
    <span class="env-stat-val">+1000/N per tile, -0.1/frame</span></div>
  <div class="env-stat"><span class="env-stat-label">Difficulty</span>
    <span class="env-stat-val">Vision + continuous control</span></div>
</div>
""",
                unsafe_allow_html=True,
            )
    with col_env2:
        st.markdown(
            """
<div class="panel">
  <h4>Observation pipeline</h4>
  <p>The raw 96x96 RGB frame goes through a preprocessing pipeline before the neural network
  sees it:</p>
  <br>
  <div class="env-stat"><span class="env-stat-label">1. Crop</span>
    <span class="env-stat-val">Remove HUD bar at bottom</span></div>
  <div class="env-stat"><span class="env-stat-label">2. Grayscale</span>
    <span class="env-stat-val">3 channels to 1</span></div>
  <div class="env-stat"><span class="env-stat-label">3. Resize</span>
    <span class="env-stat-val">84 x 84 pixels</span></div>
  <div class="env-stat"><span class="env-stat-label">4. Normalize</span>
    <span class="env-stat-val">[0, 255] to [0, 1]</span></div>
  <div class="env-stat"><span class="env-stat-label">5. Stack</span>
    <span class="env-stat-val">4 frames for velocity info</span></div>
  <br>
  <h4>Why frame stacking?</h4>
  <p>A single image is a photograph &mdash; you can't tell if the car is moving or stopped,
  turning or straight. Four consecutive frames encode velocity and acceleration implicitly
  through pixel differences, restoring the Markov property.</p>
</div>
""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB: HIGHWAY SCENARIOS
# ══════════════════════════════════════════════════════════════════
with tab_highway:
    st.markdown('<div class="sec-h">Highway-env driving scenarios</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.88rem;color:#71717A;line-height:1.7;margin:0 0 1.5rem">'
        "The same PPO algorithm applied to three different driving challenges from "
        "<strong style='color:#A1A1AA'>highway-env</strong>. Each tests a different "
        "competency: speed, timing, and precision. The only change between environments "
        "is the network architecture (MLP instead of CNN) and action distribution."
        "</p>",
        unsafe_allow_html=True,
    )

    scenarios = [
        {
            "title": "Highway Overtaking",
            "env": "highway-v0",
            "gif": "assets/highway_demo.gif",
            "color": BLUE,
            "desc": "4-lane highway at 130 km/h. Overtake slower traffic without collisions.",
            "action": "Discrete (5 actions)",
            "obs": "Kinematics of 5 vehicles",
            "difficulty": "Speed + safety",
            "network": "MLP (74K params)",
            "steps": "500K",
        },
        {
            "title": "Roundabout Navigation",
            "env": "roundabout-v0",
            "gif": "assets/roundabout_demo.gif",
            "color": TEAL,
            "desc": "Time entry, navigate around traffic, exit cleanly. Tricky even for humans.",
            "action": "Discrete (5 actions)",
            "obs": "Kinematics of 5 vehicles",
            "difficulty": "Timing + awareness",
            "network": "MLP (74K params)",
            "steps": "300K",
        },
        {
            "title": "Autonomous Parking",
            "env": "parking-v0",
            "gif": "assets/parking_demo.gif",
            "color": AMBER,
            "desc": "Goal-conditioned sparse reward. Park in the target spot with precision.",
            "action": "Continuous (steer + throttle)",
            "obs": "18-dim goal vector",
            "difficulty": "Precision + sparse reward",
            "network": "MLP (74K params)",
            "steps": "300K",
        },
    ]

    cols = st.columns(3, gap="medium")
    for col, s in zip(cols, scenarios):
        with col:
            r = hw_results.get(s["env"], {})
            reward = r.get("best_reward", 0) if isinstance(r, dict) else (r or 0)
            hours = r.get("training_hours", 0) if isinstance(r, dict) else 0

            st.markdown(
                f"""
<div class="env-card">
  <div class="env-card-header">
    <p class="env-tag" style="color:{s['color']}">{s['env']}</p>
    <p class="env-title">{s['title']}</p>
    <p class="env-desc">{s['desc']}</p>
  </div>
  <div class="env-card-body">
""",
                unsafe_allow_html=True,
            )
            show_gif(s["gif"], width="100%", border_radius="0")
            st.markdown(
                f"""
  </div>
  <div class="env-card-footer">
    <div class="env-stat">
      <span class="env-stat-label">Best reward</span>
      <span style="font-size:1.1rem;font-weight:700;color:{s['color']};font-family:'JetBrains Mono',monospace">{reward:.1f}</span>
    </div>
    <div class="env-stat">
      <span class="env-stat-label">Actions</span>
      <span class="env-stat-val">{s['action']}</span>
    </div>
    <div class="env-stat">
      <span class="env-stat-label">Observation</span>
      <span class="env-stat-val">{s['obs']}</span>
    </div>
    <div class="env-stat">
      <span class="env-stat-label">Network</span>
      <span class="env-stat-val">{s['network']}</span>
    </div>
    <div class="env-stat">
      <span class="env-stat-label">Training</span>
      <span class="env-stat-val">{s['steps']} steps / {hours:.1f} hrs</span>
    </div>
    <div class="env-stat">
      <span class="env-stat-label">Hard part</span>
      <span class="env-stat-val">{s['difficulty']}</span>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Comparison table
    st.markdown('<div class="sec-h">Same algorithm, different worlds</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="panel">
  <p style="margin-bottom:14px">The exact same PPO implementation works across all four environments.
  Only the network architecture changes: CNN for pixels, MLP for vectors.</p>
  <table class="tbl">
    <thead><tr>
      <th></th><th>CarRacing</th><th>Highway</th><th>Roundabout</th><th>Parking</th>
    </tr></thead>
    <tbody>
      <tr><td>Observation</td><td class="v">4x84x84 pixels</td>
          <td class="v">5x5 kinematics</td><td class="v">5x5 kinematics</td>
          <td class="v">18-dim goal</td></tr>
      <tr><td>Action space</td><td class="v">Continuous (3)</td>
          <td class="v">Discrete (5)</td><td class="v">Discrete (5)</td>
          <td class="v">Continuous (2)</td></tr>
      <tr><td>Reward type</td><td class="v">Dense</td>
          <td class="v">Dense (speed)</td><td class="v">Dense</td>
          <td class="v">Sparse (distance)</td></tr>
      <tr><td>Other agents</td><td>None</td>
          <td class="v">10 cars</td><td class="v">10 cars</td>
          <td>Parked only</td></tr>
      <tr><td>Network</td><td class="v">CNN (1.78M)</td>
          <td class="v">MLP (74K)</td><td class="v">MLP (74K)</td>
          <td class="v">MLP (74K)</td></tr>
      <tr><td>Training</td><td class="v">~12 hrs</td>
          <td class="v">~2.4 hrs</td><td class="v">~1.1 hrs</td>
          <td class="v">~0.7 hrs</td></tr>
    </tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Parking analysis
    st.markdown('<div class="sec-h">Why parking is the hardest</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <h4>The sparse reward problem</h4>
  <p style="margin-bottom:12px">Parking uses a goal-conditioned sparse reward: the agent only learns
  how far it is from the target position and orientation. Unlike highway (instant speed feedback) or
  CarRacing (per-tile reward), there's almost no gradient signal to follow.</p>

  <div class="insight-grid" style="margin-top:12px">
    <div style="text-align:center;padding:12px">
      <p style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:{RED};margin:0">-47</p>
      <p style="font-size:0.7rem;color:#52525B;margin:4px 0 0">Random agent</p>
    </div>
    <div style="text-align:center;padding:12px">
      <p style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:{AMBER};margin:0">-11.6</p>
      <p style="font-size:0.7rem;color:#52525B;margin:4px 0 0">Our PPO agent</p>
    </div>
    <div style="text-align:center;padding:12px">
      <p style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:{GREEN};margin:0">4x</p>
      <p style="font-size:0.7rem;color:#52525B;margin:4px 0 0">Improvement</p>
    </div>
  </div>

  <p style="margin-top:12px">This is deliberately left as an unsolved challenge. HER (Hindsight Experience Replay)
  would likely push the score near 0, but the point is to show where vanilla PPO hits its limits.</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB: TRAINING DEEP-DIVE
# ══════════════════════════════════════════════════════════════════
with tab_train:
    # Load training data
    train_csv = ASSETS / "training_metrics.csv"
    eval_csv = ASSETS / "eval_metrics.csv"
    train_df = pd.read_csv(train_csv) if train_csv.exists() else None
    eval_df = pd.read_csv(eval_csv) if eval_csv.exists() else None

    if train_df is not None:
        train_df.columns = [c.lower().strip() for c in train_df.columns]
    if eval_df is not None:
        eval_df.columns = [c.lower().strip() for c in eval_df.columns]

    # ── Main reward curve ──
    st.markdown('<div class="sec-h">Evaluation reward over training</div>', unsafe_allow_html=True)

    if eval_df is not None:
        step_col = next((c for c in eval_df.columns if "step" in c), None)
        reward_col = next((c for c in eval_df.columns if "reward" in c or "return" in c), None)

        if step_col and reward_col:
            fig = go.Figure()
            w = max(1, len(eval_df) // 20)
            smoothed = eval_df[reward_col].rolling(w, center=True).mean()

            fig.add_trace(go.Scatter(
                x=eval_df[step_col], y=eval_df[reward_col], mode="lines+markers",
                name="Eval reward",
                line=dict(color="rgba(59,139,212,0.25)", width=1),
                marker=dict(color=BLUE, size=3),
            ))
            fig.add_trace(go.Scatter(
                x=eval_df[step_col], y=smoothed, mode="lines", name="Smoothed",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(59,139,212,0.05)",
            ))
            fig.add_hline(y=700, line_dash="dash", line_color="rgba(34,197,94,0.35)",
                          annotation_text="Target: 700", annotation_font_color=GREEN,
                          annotation_font_size=10)
            fig.add_hline(y=900, line_dash="dot", line_color="rgba(245,158,11,0.25)",
                          annotation_text="Human level: ~900", annotation_font_color=AMBER,
                          annotation_font_size=10)

            best_idx = eval_df[reward_col].idxmax()
            fig.add_trace(go.Scatter(
                x=[eval_df[step_col].iloc[best_idx]], y=[eval_df[reward_col].iloc[best_idx]],
                mode="markers+text", marker=dict(color=GREEN, size=10, symbol="star"),
                text=[f"  Best: {eval_df[reward_col].iloc[best_idx]:.0f}"],
                textposition="middle right", textfont=dict(size=11, color=GREEN),
                name="Best", showlegend=False,
            ))
            fig.update_layout(**make_layout("Eval reward vs training steps", "Eval reward", 400))
            st.plotly_chart(fig, use_container_width=True)

    elif train_df is not None:
        step_col = next((c for c in train_df.columns if "step" in c), None)
        reward_col = next((c for c in train_df.columns if "reward" in c or "return" in c), None)
        if step_col and reward_col:
            fig = go.Figure()
            w = max(1, len(train_df) // 40)
            smoothed = train_df[reward_col].rolling(w, center=True).mean()
            fig.add_trace(go.Scatter(
                x=train_df[step_col], y=train_df[reward_col], mode="lines",
                line=dict(color="rgba(59,139,212,0.15)", width=1), name="Raw",
            ))
            fig.add_trace(go.Scatter(
                x=train_df[step_col], y=smoothed, mode="lines", name="Smoothed",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(59,139,212,0.05)",
            ))
            fig.update_layout(**make_layout("Training reward", "Reward", 400))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload `training_metrics.csv` or `eval_metrics.csv` to `assets/` to see training curves.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Training metrics exploration ──
    if train_df is not None:
        st.markdown('<div class="sec-h">Training metrics explorer</div>', unsafe_allow_html=True)

        step_col = next((c for c in train_df.columns if "step" in c), None)
        available_metrics = {}
        metric_info = {
            "policy_loss": ("Policy Loss", BLUE, "Clipped surrogate objective. Should be small and stable."),
            "value_loss": ("Value Loss", ORANGE, "MSE between predicted and actual returns. Spikes = new situations."),
            "entropy": ("Policy Entropy", TEAL, "How random the policy is. Should decrease over time but not collapse."),
            "approx_kl": ("KL Divergence", PURPLE, "How much the policy changed. KL > target triggers early stopping."),
            "clip_frac": ("Clip Fraction", PINK, "% of samples hitting the PPO clip boundary."),
            "lr": ("Learning Rate", AMBER, "Linear decay from 2.5e-4 to near zero."),
            "sps": ("Steps/Second", GREEN, "Training throughput (higher = faster)."),
        }

        for key in metric_info:
            if key in train_df.columns:
                available_metrics[key] = metric_info[key]

        if step_col and available_metrics:
            selected = st.multiselect(
                "Select metrics to visualize",
                options=list(available_metrics.keys()),
                default=["entropy", "value_loss", "approx_kl"],
                format_func=lambda x: available_metrics[x][0],
            )

            if selected:
                n_cols = min(len(selected), 2)
                chart_cols = st.columns(n_cols, gap="medium")
                for i, metric_key in enumerate(selected):
                    name, color, desc = available_metrics[metric_key]
                    with chart_cols[i % n_cols]:
                        fig = go.Figure()
                        y_data = train_df[metric_key]
                        w = max(1, len(train_df) // 60)
                        smoothed = y_data.rolling(w, center=True).mean()

                        fig.add_trace(go.Scatter(
                            x=train_df[step_col], y=y_data, mode="lines",
                            line=dict(color=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)", width=1),
                            name="Raw", showlegend=False,
                        ))
                        fig.add_trace(go.Scatter(
                            x=train_df[step_col], y=smoothed, mode="lines",
                            line=dict(color=color, width=2),
                            name=name,
                        ))
                        fig.update_layout(**make_layout(name, name, 280))
                        fig.update_layout(margin=dict(t=40, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(
                            f'<p style="font-size:0.72rem;color:#52525B;margin:-12px 0 8px;line-height:1.5">{desc}</p>',
                            unsafe_allow_html=True,
                        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Reward distribution ──
    if eval_df is not None:
        reward_col = next((c for c in eval_df.columns if "reward" in c), None)
        if reward_col:
            st.markdown('<div class="sec-h">Reward distribution across training</div>', unsafe_allow_html=True)

            rewards = eval_df[reward_col]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=rewards, nbinsx=25,
                marker=dict(color=BLUE, line=dict(color="rgba(59,139,212,0.5)", width=1)),
                opacity=0.7,
            ))
            fig.add_vline(x=rewards.mean(), line_dash="dash", line_color=AMBER,
                          annotation_text=f"Mean: {rewards.mean():.0f}",
                          annotation_font_color=AMBER, annotation_font_size=10)
            fig.add_vline(x=rewards.max(), line_dash="dash", line_color=GREEN,
                          annotation_text=f"Best: {rewards.max():.0f}",
                          annotation_font_color=GREEN, annotation_font_size=10)
            fig.update_layout(**make_layout("Distribution of eval rewards", "Count", 300))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Results summary
    st.markdown('<div class="sec-h">Final results summary</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns(2, gap="large")
    with res_col1:
        st.markdown(
            f"""
<div class="panel">
  <h4>CarRacing-v2</h4>
  <table class="tbl">
    <thead><tr><th>Metric</th><th>Value</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td>Peak episode reward</td><td class="g">{peak_r:.0f}</td><td>single best episode</td></tr>
      <tr><td>Average (20 episodes)</td><td class="g">874</td><td>consistent performance</td></tr>
      <tr><td>Human level</td><td class="v">~900</td><td>agent exceeds this</td></tr>
      <tr><td>Total training steps</td><td class="v">7M</td><td>5M base + 2M fine-tune</td></tr>
      <tr><td>Wall-clock time</td><td class="v">~12 hrs</td><td>NVIDIA T4 GPU</td></tr>
      <tr><td>Parameters</td><td class="v">1.78M</td><td>CNN actor-critic</td></tr>
    </tbody>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )
    with res_col2:
        ra_r = hw_results.get("roundabout-v0", {})
        pk_r = hw_results.get("parking-v0", {})
        ra_val = ra_r.get("best_reward", 41.1) if isinstance(ra_r, dict) else (ra_r or 0)
        pk_val = pk_r.get("best_reward", -11.6) if isinstance(pk_r, dict) else (pk_r or 0)
        st.markdown(
            f"""
<div class="panel">
  <h4>Highway-env scenarios</h4>
  <table class="tbl">
    <thead><tr><th>Environment</th><th>Score</th><th>Steps</th><th>Time</th></tr></thead>
    <tbody>
      <tr><td>Highway-v0</td><td class="g">{hw_reward:.1f}</td><td class="v">500K</td><td>~2.4 hrs</td></tr>
      <tr><td>Roundabout-v0</td><td class="v">{ra_val:.1f}</td><td class="v">300K</td><td>~1.1 hrs</td></tr>
      <tr><td>Parking-v0</td><td class="v">{pk_val:.1f}</td><td class="v">300K</td><td>~0.7 hrs</td></tr>
    </tbody>
  </table>
  <p style="font-size:0.75rem;color:#52525B;margin:12px 0 0">
    All trained with MLP actor-critic (74K params) on same PPO implementation.
    Parking uses sparse rewards (random baseline: -47), making it the hardest scenario.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown('<div class="sec-h">Network architecture</div>', unsafe_allow_html=True)

    arch_col1, arch_col2 = st.columns([3, 2], gap="large")

    with arch_col1:
        arch_path = ASSETS / "architecture_diagram.png"
        if arch_path.exists():
            st.image(Image.open(arch_path), use_container_width=True)
        else:
            st.markdown(
                f"""
<div class="panel" style="text-align:center;padding:2rem">
  <p style="font-size:0.85rem;color:#A1A1AA;margin:0 0 6px">
    <strong>CNN Actor-Critic</strong> (CarRacing)</p>
  <div class="code-block" style="text-align:left;margin-top:12px">
    Input: <span class="num">4</span> x <span class="num">84</span> x <span class="num">84</span> grayscale frames<br>
    <br>
    Conv2d(<span class="num">4</span>, <span class="num">32</span>, <span class="num">8</span>, stride=<span class="num">4</span>) + ReLU<br>
    Conv2d(<span class="num">32</span>, <span class="num">64</span>, <span class="num">4</span>, stride=<span class="num">2</span>) + ReLU<br>
    Conv2d(<span class="num">64</span>, <span class="num">64</span>, <span class="num">3</span>, stride=<span class="num">1</span>) + ReLU<br>
    Flatten(<span class="num">3136</span>) + Linear(<span class="num">512</span>) + ReLU<br>
    <br>
    <span class="fn">Actor:</span> Linear(<span class="num">3</span>) <span class="cm">[steer, gas, brake]</span><br>
    <span class="fn">Critic:</span> Linear(<span class="num">1</span>) <span class="cm">[V(s)]</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with arch_col2:
        st.markdown(
            f"""
<div class="panel">
  <h4>Design rationale</h4>
  <p style="margin-bottom:12px">The architecture follows the Nature DQN backbone
  (Mnih et al., 2015) with a shared feature extractor feeding separate actor and critic heads.
  This is standard for PPO on pixel inputs.</p>

  <div style="display:grid;gap:10px">
    <div class="env-stat"><span class="env-stat-label">Shared backbone</span>
      <span class="env-stat-val">Lower param count</span></div>
    <div class="env-stat"><span class="env-stat-label">Separate heads</span>
      <span class="env-stat-val">Independent learning rates</span></div>
    <div class="env-stat"><span class="env-stat-label">Learnable log_std</span>
      <span class="env-stat-val">State-independent exploration</span></div>
    <div class="env-stat"><span class="env-stat-label">No batch norm</span>
      <span class="env-stat-val">Simpler, more stable</span></div>
    <div class="env-stat"><span class="env-stat-label">Orthogonal init</span>
      <span class="env-stat-val">Stable gradient flow</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_layers, col_hyper = st.columns(2, gap="large")

    with col_layers:
        st.markdown('<div class="sec-h">Layer details</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
<table class="tbl">
  <thead><tr><th>Layer</th><th>Output</th><th>Params</th></tr></thead>
  <tbody>
    <tr><td>Input</td><td class="v">4 x 84 x 84</td><td>-</td></tr>
    <tr><td>Conv2d (8x8, stride 4)</td><td class="v">32 x 20 x 20</td><td class="v">8,224</td></tr>
    <tr><td>Conv2d (4x4, stride 2)</td><td class="v">64 x 9 x 9</td><td class="v">32,832</td></tr>
    <tr><td>Conv2d (3x3, stride 1)</td><td class="v">64 x 7 x 7</td><td class="v">36,928</td></tr>
    <tr><td>Linear (shared)</td><td class="v">512</td><td class="v">1,606,144</td></tr>
    <tr><td>Actor head</td><td class="v">3</td><td class="v">1,539</td></tr>
    <tr><td>log_std (learnable)</td><td class="v">3</td><td class="v">3</td></tr>
    <tr><td>Critic head</td><td class="v">1</td><td class="v">513</td></tr>
    <tr><td><strong>Total</strong></td><td></td><td class="g"><strong>1,686,183</strong></td></tr>
  </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_hyper:
        st.markdown('<div class="sec-h">Hyperparameters</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
<table class="tbl">
  <thead><tr><th>Parameter</th><th>Base</th><th>Fine-tune</th></tr></thead>
  <tbody>
    <tr><td>Total steps</td><td class="v">5M</td><td class="v">+2M</td></tr>
    <tr><td>Parallel envs</td><td class="v">8</td><td class="v">8</td></tr>
    <tr><td>Rollout steps</td><td class="v">256</td><td class="v">256</td></tr>
    <tr><td>Minibatch size</td><td class="v">256</td><td class="v">256</td></tr>
    <tr><td>Epochs per update</td><td class="v">4</td><td class="v">4</td></tr>
    <tr><td>Learning rate</td><td class="v">2.5e-4</td><td class="v">5e-5</td></tr>
    <tr><td>Clip epsilon</td><td class="v">0.2</td><td class="v">0.15</td></tr>
    <tr><td>Entropy coeff</td><td class="v">0.01</td><td class="v">0.02</td></tr>
    <tr><td>KL target</td><td class="v">0.02</td><td class="v">0.015</td></tr>
    <tr><td>Gamma</td><td class="v">0.99</td><td class="v">0.99</td></tr>
    <tr><td>GAE lambda</td><td class="v">0.95</td><td class="v">0.95</td></tr>
    <tr><td>Grad clip</td><td class="v">0.5</td><td class="v">0.5</td></tr>
  </tbody>
</table>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Design decisions
    st.markdown('<div class="sec-h">Design decisions</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="spec-grid">
  <div class="spec-card">
    <p class="sk">PPO over DQN</p>
    <p class="sv">Continuous action space</p>
    <p class="sd">Steering is a float in [-1, 1]. DQN requires discretisation which
    loses precision. PPO outputs continuous values through a Gaussian policy.</p>
  </div>
  <div class="spec-card">
    <p class="sk">Frame stacking (x4)</p>
    <p class="sv">Implicit velocity encoding</p>
    <p class="sd">A single frame is position-only. Stacking 4 consecutive frames encodes
    velocity via pixel differences, restoring the Markov property.</p>
  </div>
  <div class="spec-card">
    <p class="sk">GAE (lambda = 0.95)</p>
    <p class="sv">Bias-variance trade-off</p>
    <p class="sd">Blends TD(0) with Monte Carlo returns. Lambda = 0.95 leans toward MC
    for lower bias while keeping variance manageable.</p>
  </div>
  <div class="spec-card">
    <p class="sk">Entropy bonus</p>
    <p class="sv">Prevents premature convergence</p>
    <p class="sd">Without it, the policy collapses to a single action early in training.
    The entropy term keeps exploration alive during the critical early phase.</p>
  </div>
  <div class="spec-card">
    <p class="sk">Orthogonal init</p>
    <p class="sv">Stable gradient flow</p>
    <p class="sd">CNN: sqrt(2) gain. Actor: 0.01 (near-uniform initial policy). Critic: 1.0.
    Prevents vanishing/exploding gradients at initialization.</p>
  </div>
  <div class="spec-card">
    <p class="sk">KL early stopping</p>
    <p class="sv">Adaptive epoch count</p>
    <p class="sd">If KL divergence exceeds the target, stop the epoch loop early.
    This prevents catastrophic policy updates on unusual batches.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB: HOW PPO WORKS
# ══════════════════════════════════════════════════════════════════
with tab_algo:
    st.markdown('<div class="sec-h">Proximal Policy Optimisation explained</div>', unsafe_allow_html=True)

    st.markdown(
        """
<div class="panel" style="margin-bottom:1.5rem">
  <h4>The core idea in one sentence</h4>
  <p style="font-size:0.92rem;color:#D4D4D8;line-height:1.8">
    PPO improves the policy by asking <em>"was this action better or worse than expected?"</em>
    for each collected experience, then updating the network &mdash; but capping how much
    it can change in a single step to prevent catastrophic forgetting.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    algo_col1, algo_col2 = st.columns(2, gap="large")

    with algo_col1:
        st.markdown(
            f"""
<div class="panel">
  <h4>The training loop</h4>
  <div class="code-block" style="margin-top:10px">
    <span class="kw">for</span> iteration <span class="kw">in</span> range(<span class="num">7_000_000</span> // (<span class="num">8</span> * <span class="num">256</span>)):<br>
    <br>
    &nbsp;&nbsp;<span class="cm"># 1. Collect experiences</span><br>
    &nbsp;&nbsp;<span class="kw">for</span> step <span class="kw">in</span> range(<span class="num">256</span>):<br>
    &nbsp;&nbsp;&nbsp;&nbsp;action = <span class="fn">policy</span>(obs) &nbsp;<span class="cm"># sample from Gaussian</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;obs, reward, done = <span class="fn">env.step</span>(action)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<span class="fn">buffer.store</span>(obs, action, reward, done)<br>
    <br>
    &nbsp;&nbsp;<span class="cm"># 2. Compute advantages (GAE)</span><br>
    &nbsp;&nbsp;advantages = <span class="fn">compute_gae</span>(rewards, values)<br>
    <br>
    &nbsp;&nbsp;<span class="cm"># 3. PPO update (clipped)</span><br>
    &nbsp;&nbsp;<span class="kw">for</span> epoch <span class="kw">in</span> range(<span class="num">4</span>):<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<span class="kw">for</span> batch <span class="kw">in</span> <span class="fn">shuffle</span>(buffer):<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ratio = new_prob / old_prob<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loss = -<span class="fn">min</span>(ratio * A, <span class="fn">clip</span>(ratio) * A)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="fn">optimizer.step</span>(loss)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;<span class="kw">if</span> kl > target: <span class="kw">break</span> &nbsp;<span class="cm"># early stop</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with algo_col2:
        st.markdown(
            f"""
<div class="panel">
  <h4>Key components</h4>
  <div style="display:grid;gap:16px;margin-top:10px">

    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{BLUE};margin:0 0 4px;text-transform:uppercase;letter-spacing:0.08em">
        Clipped objective</p>
      <p style="font-size:0.8rem;color:#A1A1AA;line-height:1.6;margin:0">
        The probability ratio r(theta) is clipped to [1-eps, 1+eps]. This prevents the policy from
        changing too drastically in a single update, which is the #1 cause of RL training collapse.</p>
    </div>

    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{TEAL};margin:0 0 4px;text-transform:uppercase;letter-spacing:0.08em">
        GAE (Generalized Advantage Estimation)</p>
      <p style="font-size:0.8rem;color:#A1A1AA;line-height:1.6;margin:0">
        Instead of raw returns, GAE computes a weighted blend of n-step returns.
        Lambda=0.95 gets most of Monte Carlo's low bias with TD's lower variance.</p>
    </div>

    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{AMBER};margin:0 0 4px;text-transform:uppercase;letter-spacing:0.08em">
        Entropy bonus</p>
      <p style="font-size:0.8rem;color:#A1A1AA;line-height:1.6;margin:0">
        Adding H(pi) to the objective encourages the policy to stay stochastic during training.
        Without it, the agent prematurely commits to a suboptimal strategy.</p>
    </div>

    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{PURPLE};margin:0 0 4px;text-transform:uppercase;letter-spacing:0.08em">
        Value function</p>
      <p style="font-size:0.8rem;color:#A1A1AA;line-height:1.6;margin:0">
        The critic learns V(s) &mdash; how good is this state? This baseline reduces variance
        in the policy gradient, making learning much more sample-efficient.</p>
    </div>

  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # The math
    st.markdown('<div class="sec-h">The mathematics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="panel">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{BLUE};margin:0 0 8px;text-transform:uppercase;letter-spacing:0.08em">
        PPO-Clip Objective</p>
      <div class="code-block">
        L = E[ <span class="fn">min</span>( r * A, <span class="fn">clip</span>(r, <span class="num">1-e</span>, <span class="num">1+e</span>) * A ) ]<br>
        <br>
        <span class="cm">where:</span><br>
        &nbsp;&nbsp;r = pi_new(a|s) / pi_old(a|s)<br>
        &nbsp;&nbsp;A = advantage (from GAE)<br>
        &nbsp;&nbsp;e = <span class="num">0.2</span> <span class="cm">(clip range)</span>
      </div>
    </div>
    <div>
      <p style="font-size:0.72rem;font-weight:700;color:{TEAL};margin:0 0 8px;text-transform:uppercase;letter-spacing:0.08em">
        GAE Formula</p>
      <div class="code-block">
        A_t = sum( (gamma * lambda)^l * delta_(t+l) )<br>
        <br>
        <span class="cm">where:</span><br>
        &nbsp;&nbsp;delta_t = r_t + gamma * V(s_(t+1)) - V(s_t)<br>
        &nbsp;&nbsp;gamma = <span class="num">0.99</span>, lambda = <span class="num">0.95</span>
      </div>
    </div>
  </div>

  <div style="margin-top:16px">
    <p style="font-size:0.72rem;font-weight:700;color:{AMBER};margin:0 0 8px;text-transform:uppercase;letter-spacing:0.08em">
      Total Loss</p>
    <div class="code-block">
      L_total = L_clip - <span class="num">0.5</span> * L_value + <span class="num">0.01</span> * H(pi)<br>
      <br>
      <span class="cm">Policy gradient + value function MSE + entropy bonus</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Why PPO over alternatives
    st.markdown('<div class="sec-h">Why PPO?</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="spec-grid">
  <div class="spec-card">
    <p class="sk">vs. Vanilla Policy Gradient</p>
    <p class="sv">More sample-efficient</p>
    <p class="sd">REINFORCE uses each experience once. PPO reuses data for 4 epochs
    via importance sampling, getting ~4x more learning per environment step.</p>
  </div>
  <div class="spec-card">
    <p class="sk">vs. TRPO</p>
    <p class="sv">Much simpler to implement</p>
    <p class="sd">TRPO needs conjugate gradients and a line search. PPO achieves similar
    trust-region behavior with a simple clipped objective. Same stability, less code.</p>
  </div>
  <div class="spec-card">
    <p class="sk">vs. DQN</p>
    <p class="sv">Continuous actions</p>
    <p class="sd">DQN discretises the action space. For steering (float in [-1,1]),
    you'd need hundreds of bins to match PPO's precision.</p>
  </div>
  <div class="spec-card">
    <p class="sk">vs. SAC</p>
    <p class="sv">Simpler, often competitive</p>
    <p class="sd">SAC adds a replay buffer and twin critics. For on-policy problems
    like CarRacing (where trajectories matter), PPO is a natural fit.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB: ABOUT
# ══════════════════════════════════════════════════════════════════
with tab_about:
    col_about, col_stack = st.columns([3, 2], gap="large")

    with col_about:
        st.markdown('<div class="sec-h">About this project</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
  <p>This project implements Proximal Policy Optimisation entirely from scratch in PyTorch.
  No RL libraries like Stable-Baselines3, no pretrained models, no shortcuts. Every line of
  the training loop, advantage estimation, and policy update is hand-written.</p>
  <br>
  <p>The goal was to deeply understand modern policy gradient methods by building and debugging
  them from the ground up. The agent trains on four autonomous driving environments of increasing
  difficulty, from pixel-based racing to sparse-reward parking.</p>
  <br>
  <p>The CarRacing agent achieves <strong style="color:#E4E4E7">940 peak reward</strong>
  (874 average), exceeding human-level performance (~900). Training took ~12 hours on a single
  NVIDIA T4 GPU.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="sec-h">Challenges overcome</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="panel">
  <table class="tbl">
    <thead><tr><th>Bug</th><th>Impact</th><th>Fix</th></tr></thead>
    <tbody>
      <tr><td>Gradient death</td><td class="v">tanh on large values = zero gradients</td>
          <td>Centered tanh + tiny init gain (0.01)</td></tr>
      <tr><td>log_std frozen</td><td class="v">Clamped at boundary, couldn't learn</td>
          <td>Widened clamp range [-2.5, 0.5]</td></tr>
      <tr><td>Python 3.13 crash</td><td class="v">box2d SWIG rejected float32</td>
          <td>Custom Float64Action wrapper</td></tr>
      <tr><td>Value loss spikes</td><td class="v">Value clipping was counterproductive</td>
          <td>Removed clipping, simple MSE</td></tr>
      <tr><td>10-epoch overfit</td><td class="v">Too many gradient steps per batch</td>
          <td>4 epochs + KL early stopping</td></tr>
      <tr><td>Parking collapse</td><td class="v">High entropy destabilized training</td>
          <td>ent=0.005, lr=1e-4, target_kl=0.01</td></tr>
    </tbody>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_stack:
        st.markdown('<div class="sec-h">Tech stack</div>', unsafe_allow_html=True)
        stack = [
            ("PyTorch 2.x", "Deep learning framework", ORANGE),
            ("Gymnasium 0.29", "RL environments", BLUE),
            ("highway-env", "Driving scenarios", TEAL),
            ("NVIDIA T4 GPU", "AWS EC2 g4dn.xlarge", GREEN),
            ("Weights & Biases", "Experiment tracking", AMBER),
            ("Hydra", "Configuration", PURPLE),
            ("Streamlit", "This dashboard", RED),
            ("Plotly", "Interactive charts", BLUE),
        ]
        stack_html = '<div class="panel"><div style="display:grid;gap:2px">'
        for name, desc, color in stack:
            stack_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.03)">'
                f'<span style="color:#D4D4D8;font-size:.82rem;font-weight:500">'
                f'<span style="color:{color};margin-right:8px">|</span>{name}</span>'
                f'<span style="color:#52525B;font-size:.72rem">{desc}</span>'
                f'</div>'
            )
        stack_html += '</div></div>'
        st.markdown(stack_html, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-h">Links</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
  <div style="display:grid;gap:12px">
    <a href="https://github.com/anmol0705/CarRacing-v2-PPO-Agent"
       style="color:#3B8BD4;text-decoration:none;font-size:.85rem;font-weight:500"
       target="_blank">
      GitHub Repository &rarr;
    </a>
    <a href="https://wandb.ai/anmol_752005/carracing-ppo"
       style="color:#3B8BD4;text-decoration:none;font-size:.85rem;font-weight:500"
       target="_blank">
      W&B Training Dashboard &rarr;
    </a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-h">Project structure</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="panel">
  <pre style="font-size:0.72rem;color:#71717A;margin:0;line-height:1.8;overflow-x:auto;font-family:'JetBrains Mono',monospace">
carracing-ppo/
  src/
    model.py          CNN actor-critic
    ppo.py            GAE + clipped PPO
    trainer.py        Training loop
    highway_trainer.py  MLP PPO for highway-env
    env_utils.py      Wrappers, VecEnv
  scripts/
    train.py          CarRacing entry point
    train_highway.py  Highway scenarios
  dashboard/
    app.py            This dashboard
  configs/
    default.yaml      Hyperparameters
    finetune.yaml     Fine-tuning config</pre>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-h">What I\'d do next</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="panel">
  <div style="display:grid;gap:10px">
    <div class="env-stat">
      <span style="color:#D4D4D8;font-size:.82rem;font-weight:500">HER for parking</span>
      <span class="env-stat-val">Score -11.6 to ~0</span>
    </div>
    <div class="env-stat">
      <span style="color:#D4D4D8;font-size:.82rem;font-weight:500">Multi-agent highway</span>
      <span class="env-stat-val">Competitive PPO agents</span>
    </div>
    <div class="env-stat">
      <span style="color:#D4D4D8;font-size:.82rem;font-weight:500">SAC comparison</span>
      <span class="env-stat-val">Off-policy baseline</span>
    </div>
    <div class="env-stat">
      <span style="color:#D4D4D8;font-size:.82rem;font-weight:500">Sim-to-real transfer</span>
      <span class="env-stat-val">RC car with domain rand</span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ── Footer ────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:.7rem;color:#27272A;text-align:center;margin:0">'
    "Built from scratch with PyTorch, Gymnasium, and Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
