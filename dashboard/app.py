"""Streamlit dashboard for CarRacing PPO agent — display-only, no gymnasium/torch deps.

Shows prerecorded GIFs, training curves from CSV, architecture details, and project info.
Imports: streamlit, pandas, plotly, PIL, pathlib, base64 only.
"""

import base64
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CarRacing-v2 PPO Agent",
    page_icon="\U0001f3ce\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1080px}
.mcard{background:#161616;border:1px solid #2a2a2a;border-radius:10px;
       padding:1rem 1.2rem;text-align:center}
.mv{font-size:2rem;font-weight:600;color:#3B8BD4;margin:0}
.ml{font-size:.72rem;color:#666;text-transform:uppercase;letter-spacing:.07em}
.ms{font-size:.7rem;color:#444;margin-top:1px}
.badge{background:#0d2e0d;color:#4caf50;border-radius:6px;
       padding:3px 10px;font-size:.8rem;font-weight:600}
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def gif_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def show_gif(path: str, caption: str = "", width: int = 700) -> None:
    p = Path(path)
    if not p.exists():
        st.warning(f"Asset not found: {path}")
        return
    b64 = gif_to_b64(path)
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:{width}px;max-width:100%;border-radius:8px;" '
        f'alt="{caption}">',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


# ── header ────────────────────────────────────────────────────────────────────

st.markdown("# \U0001f3ce\ufe0f CarRacing-v2 PPO Agent")
st.markdown(
    "A deep RL agent that learned to drive from raw pixels \u2014 "
    "no GPS, no map, no hand-coded rules. "
    "Trained with Proximal Policy Optimization over ~5M environment steps."
)
st.markdown(
    '<span class="badge">\u2705 Target 700 achieved \u2014 median 864.7 / 1000</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── metrics ──────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    ("864.7", "Median reward", "50 episodes"),
    ("933.3", "Best episode", "single run"),
    ("66%", "Above 700", "target threshold"),
    ("~5M", "Train steps", "~9.6 hrs on T4"),
    ("1.7M", "Parameters", "CNN + actor-critic"),
]
for col, (val, lbl, sub) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(
            f'<div class="mcard">'
            f'<p class="mv">{val}</p>'
            f'<p class="ml">{lbl}</p>'
            f'<p class="ms">{sub}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("")

# ── tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["\U0001f3ac Demo", "\U0001f4c8 Training", "\U0001f9e0 Architecture", "\U0001f4cb About"]
)

# ── TAB 1: DEMO ──────────────────────────────────────────────────────────────

with tab1:
    st.markdown("### Best agent \u2014 full lap")
    st.markdown(
        "Best episode cherry-picked from 18 trials across top checkpoints. "
        "Deterministic policy (mean action, no sampling) \u2014 "
        "reward **928**, completed in 716 steps."
    )

    best_candidates = [
        ASSETS / "showcase.gif",
        ASSETS / "best_lap.gif",
        ASSETS / "best_agent.gif",
    ]
    shown = False
    for c in best_candidates:
        if c.exists():
            show_gif(
                str(c),
                caption="Best episode \u2014 deterministic policy, seed selected from 18 trials",
                width=680,
            )
            shown = True
            break
    if not shown:
        st.info("showcase.gif not found \u2014 run scripts/record_showcase.py on EC2")

    st.markdown("---")
    st.markdown("### Learning progression")
    st.markdown(
        "From random flailing at step 0 to consistent lap completion by step 4.9M. "
        "Each clip is a separate checkpoint showing how behaviour evolved."
    )

    prog_candidates = [ASSETS / "progression_clean.gif", ASSETS / "progression.gif"]
    for c in prog_candidates:
        if c.exists():
            show_gif(
                str(c),
                caption="Progression: step 500K \u2192 750K \u2192 4.9M \u2192 5M",
                width=680,
            )
            break

    demo_clip = ASSETS / "demo_clip.gif"
    if demo_clip.exists():
        st.markdown("---")
        st.markdown("### Highlight clip")
        show_gif(
            str(demo_clip), caption="10-second highlight \u2014 smoothest segment", width=500
        )

# ── TAB 2: TRAINING ──────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Training curves")

    csv_path = ASSETS / "training_metrics.csv"
    if not csv_path.exists():
        csv_path = ASSETS / "eval_metrics.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.caption(
            f"Loaded {len(df):,} rows from {csv_path.name} \u2014 columns: {list(df.columns)}"
        )

        df.columns = [c.lower().strip() for c in df.columns]

        step_col = next((c for c in df.columns if "step" in c), None)
        reward_col = next(
            (c for c in df.columns if "reward" in c or "return" in c), None
        )
        entropy_col = next((c for c in df.columns if "entrop" in c), None)
        vloss_col = next(
            (c for c in df.columns if "value" in c and "loss" in c), None
        )

        if step_col and reward_col:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df[step_col],
                    y=df[reward_col],
                    mode="lines",
                    name="Eval reward",
                    line=dict(color="#3B8BD4", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(59,139,212,0.08)",
                )
            )
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.add_vline(
                x=1_850_000,
                line_dash="dash",
                line_color="rgba(255,180,50,0.5)",
                annotation_text="Breakthrough",
                annotation_position="top right",
            )
            best_row = df.loc[df[reward_col].idxmax()]
            fig.add_trace(
                go.Scatter(
                    x=[best_row[step_col]],
                    y=[best_row[reward_col]],
                    mode="markers+text",
                    marker=dict(color="#D85A30", size=10),
                    text=[f"  Best: {best_row[reward_col]:.0f}"],
                    textposition="middle right",
                    name="Best checkpoint",
                    showlegend=False,
                )
            )
            fig.update_layout(
                title="Eval reward vs training steps",
                xaxis_title="Steps",
                yaxis_title="Eval reward",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=360,
                margin=dict(l=50, r=30, t=50, b=50),
            )
            st.plotly_chart(fig, width="stretch")

            cols_available = []
            if entropy_col:
                cols_available.append((entropy_col, "Entropy", "#1D9E75"))
            if vloss_col:
                cols_available.append((vloss_col, "Value loss", "#D85A30"))

            if cols_available:
                fig2 = go.Figure()
                for col, name, color in cols_available:
                    fig2.add_trace(
                        go.Scatter(
                            x=df[step_col],
                            y=df[col],
                            mode="lines",
                            name=name,
                            line=dict(color=color, width=1.2),
                        )
                    )
                fig2.update_layout(
                    title="Entropy & value loss",
                    xaxis_title="Steps",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=50, r=30, t=50, b=50),
                )
                st.plotly_chart(fig2, width="stretch")
        else:
            st.write("Columns found:", list(df.columns))
            st.dataframe(df.head(20))
    else:
        st.warning("training_metrics.csv not found in assets/")

    st.markdown("---")
    st.markdown("### Eval results summary")
    results = {
        "Metric": [
            "Best eval reward",
            "50-episode median",
            "50-episode mean",
            "50-episode max",
            "Episodes > 700",
            "Training steps",
            "Training time",
        ],
        "Value": ["811.9", "864.7", "632.1", "933.3", "66%", "~5,000,000", "~9.6 hrs"],
        "Notes": [
            "checkpoint at step 4.9M",
            "strong, consistent performance",
            "pulled down by occasional crashes",
            "near-perfect lap",
            "target was 700",
            "on AWS EC2 g4dn.xlarge",
            "NVIDIA T4 GPU",
        ],
    }
    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

# ── TAB 3: ARCHITECTURE ─────────────────────────────────────────────────────

with tab3:
    st.markdown("### Model architecture")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**CNN feature extractor**")
        arch = pd.DataFrame(
            {
                "Layer": [
                    "Input",
                    "Conv2d 1",
                    "Conv2d 2",
                    "Conv2d 3",
                    "Flatten",
                    "Linear",
                ],
                "Shape out": [
                    "4\u00d784\u00d784",
                    "32\u00d720\u00d720",
                    "64\u00d79\u00d79",
                    "64\u00d77\u00d77",
                    "3136",
                    "512",
                ],
                "Details": [
                    "4 stacked grayscale frames",
                    "32 filters, 8\u00d78, stride 4",
                    "64 filters, 4\u00d74, stride 2",
                    "64 filters, 3\u00d73, stride 1",
                    "\u2014",
                    "ReLU activation",
                ],
                "Params": [
                    "\u2014",
                    "8,224",
                    "131,136",
                    "36,928",
                    "\u2014",
                    "1,605,632",
                ],
            }
        )
        st.dataframe(arch, width="stretch", hide_index=True)

        st.markdown("**Actor & critic heads** (shared backbone)")
        heads = pd.DataFrame(
            {
                "Head": ["Actor \u03bc", "Actor log \u03c3", "Critic V"],
                "Output": ["3", "3", "1"],
                "Meaning": [
                    "Mean steering/gas/brake",
                    "Log std of Gaussian policy",
                    "Expected return from state",
                ],
            }
        )
        st.dataframe(heads, width="stretch", hide_index=True)

    with c2:
        st.markdown("**Hyperparameters**")
        hp = pd.DataFrame(
            {
                "Parameter": [
                    "Algorithm",
                    "n_envs",
                    "rollout_steps",
                    "minibatch_size",
                    "n_epochs",
                    "learning_rate",
                    "gamma",
                    "gae_lambda",
                    "clip_epsilon",
                    "vf_coef",
                    "ent_coef",
                    "target_kl",
                    "total_steps",
                ],
                "Value": [
                    "PPO",
                    "8",
                    "128",
                    "256",
                    "4",
                    "3e-4 \u2192 0 (linear)",
                    "0.99",
                    "0.95",
                    "0.2",
                    "0.5",
                    "0.01",
                    "0.015",
                    "5,000,000",
                ],
            }
        )
        st.dataframe(hp, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Why these choices?")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**PPO over DQN**")
        st.markdown(
            "CarRacing has *continuous* actions \u2014 steering is a float "
            "in [-1, 1], not a discrete left/right choice. DQN only handles "
            "discrete actions. PPO uses a Gaussian policy that naturally "
            "outputs continuous values."
        )
    with e2:
        st.markdown("**Frame stacking**")
        st.markdown(
            "A single frame contains no velocity information \u2014 you can't "
            "tell if the car is moving or still. Stacking 4 consecutive "
            "frames gives the network implicit motion through pixel "
            "differences, restoring the Markov property."
        )
    with e3:
        st.markdown("**GAE advantage**")
        st.markdown(
            "We can't wait until race end to assign credit. GAE (\u03bb=0.95) "
            "blends short-horizon TD estimates (low variance, biased) with "
            "long-horizon returns (unbiased, high variance) \u2014 getting the "
            "best of both worlds."
        )

# ── TAB 4: ABOUT ─────────────────────────────────────────────────────────────

with tab4:
    st.markdown("### Project overview")
    st.markdown(
        """
This project implements **Proximal Policy Optimization (PPO)** from scratch in PyTorch
to train an agent that drives in OpenAI Gymnasium's CarRacing-v2 environment using
only raw pixel observations.

**Environment:** CarRacing-v2 (Gymnasium 0.29.1)
- Observation: 96\u00d796 RGB top-down view (preprocessed to 4\u00d784\u00d784 grayscale stack)
- Action space: continuous \u2014 steering \u2208 [-1,1], gas \u2208 [0,1], brake \u2208 [0,1]
- Reward: +1000/N per track tile visited, -0.1 per frame, -100 if off-track

**Training infrastructure:** AWS EC2 g4dn.xlarge (NVIDIA T4 GPU, 16GB VRAM)

**Key implementation details:**
- Orthogonal weight initialisation (\u221a2 for CNN, 0.01 for actor, 1.0 for critic)
- Linear learning rate decay 3e-4 \u2192 0 over all training steps
- Gradient clipping at 0.5
- Value function clipping matching policy clip range
- Entropy bonus (0.01) to prevent premature convergence
    """
    )

    st.markdown("---")
    st.markdown("### Training timeline")
    timeline = pd.DataFrame(
        {
            "Phase": [
                "Random exploration",
                "Early learning",
                "Breakthrough",
                "Consolidation",
                "Final polish",
            ],
            "Steps": [
                "0 \u2192 1.5M",
                "1.5M \u2192 1.85M",
                "1.85M \u2192 2.7M",
                "2.7M \u2192 4.5M",
                "4.5M \u2192 5M",
            ],
            "Avg reward": [
                "\u221284 \u2192 \u221270",
                "\u221270 \u2192 +30",
                "+30 \u2192 +130",
                "+20 \u2192 +80",
                "+50 \u2192 +80",
            ],
            "What happened": [
                "Agent flails randomly, occasionally stays on track",
                "Learns to steer, still crashes frequently",
                "Discovers that staying on track is rewarded \u2014 hockey stick begins",
                "Refines steering, learns to brake on corners",
                "LR near zero, minor fine-tuning",
            ],
        }
    )
    st.dataframe(timeline, width="stretch", hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Built with**")
        for tech in [
            "PyTorch 2.x + CUDA",
            "Gymnasium 0.29.1",
            "Hydra (config)",
            "Weights & Biases",
            "AWS EC2 g4dn.xlarge",
        ]:
            st.markdown(f"- {tech}")
    with col2:
        st.markdown("**Links**")
        st.markdown(
            "- [GitHub Repository](https://github.com/anmol0705/CarRacing-v2-PPO-Agent)"
        )
        st.markdown(
            "- [Weights & Biases Run](https://wandb.ai/anmol_752005/carracing-ppo)"
        )

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "CarRacing-v2 PPO Agent \u00b7 Trained on AWS EC2 \u00b7 Built with PyTorch & Gymnasium"
)
