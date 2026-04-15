# CarRacing-v3 PPO — CLAUDE.md

## Project goal
Train a PPO agent to drive in Gymnasium's CarRacing-v3 from raw pixels.
Target: average episode reward ≥ 700 (human-level ≈ 900).
This is a portfolio/resume project — visual quality and clean code matter as much as performance.

## Environment
- Machine: AWS EC2 g4dn.xlarge (NVIDIA T4 GPU, 16 GB VRAM, 4 vCPUs)
- OS: Ubuntu 22.04, AWS Deep Learning AMI
- Python: conda base environment
- CUDA: available and verified

---

## MCP servers active in this project

Three MCP servers are configured in ~/.claude/settings.json (see setup instructions below).
Use them proactively — do not ask permission to use tools you already have.

### 1. Filesystem MCP (built-in)
Already active. Read and write any file in ~/carracing-ppo/.
Use it to: read existing files before editing, verify file contents after writing,
check that configs load correctly by reading them back.

### 2. GitHub MCP
Use it to:
- `git add` + `git commit` after every phase passes verification
- Commit message format: "phase N: <one line description> — verification passed"
- Push to remote after phases 5, 8, and 11 (the three major milestones)
- Create a GitHub release tagged v0.1 after Phase 11 completes

Do NOT commit: checkpoint .pt files, runs/, __pycache__, .env, wandb/ directories.

### 3. Shell MCP (built-in, always available)
Use it to run any terminal command directly. Key commands to use proactively:

```bash
# GPU health check — run this at start of every session
nvidia-smi

# Check GPU memory during training
watch -n2 nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# Check if training process is alive
pgrep -f train.py && echo "TRAINING RUNNING" || echo "TRAINING STOPPED"

# Tail training logs
tail -f logs/train.log

# Check disk space (checkpoints accumulate)
df -h ~/carracing-ppo/

# Python env sanity
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

## Hooks configuration

Hooks are defined in ~/.claude/settings.json alongside MCP config.
They run shell commands automatically at key moments.

### Hook 1: post-write verification runner
Trigger: after Claude Code writes any .py file in src/ or scripts/
Action: runs the corresponding verification script if it exists

```bash
# This hook runs automatically. Do not run verifications manually.
if [[ "$CLAUDE_HOOK_FILE" == */src/*.py ]] || [[ "$CLAUDE_HOOK_FILE" == */scripts/*.py ]]; then
  MODULE=$(basename "$CLAUDE_HOOK_FILE" .py)
  VERIFY="tests/verify_${MODULE}.py"
  if [ -f "$VERIFY" ]; then
    echo "=== AUTO-VERIFY: $MODULE ==="
    python "$VERIFY" && echo "HOOK PASS: $MODULE" || echo "HOOK FAIL: $MODULE — fix before continuing"
  fi
fi
```

### Hook 2: GPU memory guard
Trigger: before any training script is run
Action: checks available GPU memory, warns if below 4 GB

```bash
python -c "
import torch
if torch.cuda.is_available():
    free = torch.cuda.mem_get_info()[0] / 1e9
    if free < 4.0:
        print(f'WARNING: Only {free:.1f}GB GPU memory free. Consider reducing batch size.')
    else:
        print(f'GPU memory OK: {free:.1f}GB free')
"
```

### Hook 3: auto-format on write
Trigger: after writing any .py file
Action: runs black formatter silently

```bash
black "$CLAUDE_HOOK_FILE" --quiet 2>/dev/null || true
```

### Hook 4: checkpoint size monitor
Trigger: after any file is written to checkpoints/
Action: prints total checkpoint directory size

```bash
du -sh ~/carracing-ppo/checkpoints/ 2>/dev/null && \
  find ~/carracing-ppo/checkpoints/ -name "*.pt" | wc -l | xargs echo "checkpoint files:"
```

---

## settings.json setup (run this once before starting)

Create or update ~/.claude/settings.json with the following. This configures both
MCPs and hooks in one place:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_PAT_HERE"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/ubuntu/carracing-ppo"]
    }
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash /home/ubuntu/carracing-ppo/.claude_hooks/post_write.sh \"$CLAUDE_TOOL_INPUT_PATH\""
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash /home/ubuntu/carracing-ppo/.claude_hooks/pre_bash.sh \"$CLAUDE_TOOL_INPUT_COMMAND\""
          }
        ]
      }
    ]
  },
  "env": {
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_REGION": "us-east-1",
    "ANTHROPIC_MODEL": "us.anthropic.claude-opus-4-6-20250514-v1:0"
  }
}
```

After writing settings.json, create the hooks directory and scripts:
```bash
mkdir -p ~/carracing-ppo/.claude_hooks
```

Then write the hook scripts (Claude Code will do this in Phase 0).

---

## Repo structure
```
carracing-ppo/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── .claude_hooks/
│   ├── post_write.sh       # auto-verify + auto-format
│   └── pre_bash.sh         # GPU memory guard
├── configs/
│   └── default.yaml        # ALL hyperparams live here, nothing hardcoded
├── src/
│   ├── __init__.py
│   ├── env_utils.py        # wrappers + VecEnv factory
│   ├── model.py            # ActorCritic CNN
│   ├── ppo.py              # GAE + PPO update
│   ├── trainer.py          # main training loop
│   └── evaluate.py         # greedy eval + GIF recorder
├── scripts/
│   ├── train.py            # entry point
│   ├── record_gif.py       # progression GIF builder
│   └── make_heatmap.py     # value heatmap video
├── tests/
│   ├── verify_env_utils.py
│   ├── verify_model.py
│   ├── verify_ppo.py
│   ├── verify_evaluate.py
│   └── verify_trainer.py
├── dashboard/
│   └── app.py              # Streamlit live demo
├── logs/                   # training logs (gitignored)
├── checkpoints/            # .pt files (gitignored)
├── runs/                   # W&B local (gitignored)
└── assets/                 # GIFs for README (committed)
```

---

## Environment spec
- Env ID: `CarRacing-v3` (continuous action space)
- Observation after wrappers: `(4, 84, 84)` float32 in [0, 1]
- Action: `Box(3,)` — [steer ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]
- Reward: +1000/N per new tile visited, −0.1/frame, −100 on out-of-bounds
- Episode termination: all tiles visited OR 1000 steps OR car leaves track

## Wrappers (in this exact order)
1. `GrayScaleObservation`
2. `ResizeObservation(84, 84)`
3. `FrameStack(4)`
4. `NormalizeObservation` (divides by 255)

VecEnv: `gym.vector.AsyncVectorEnv` with n_envs=8 workers.

---

## Model architecture
```
ActorCritic(nn.Module)
  shared CNN:
    Conv2d(4,  32, kernel=8, stride=4) → ReLU
    Conv2d(32, 64, kernel=4, stride=2) → ReLU
    Conv2d(64, 64, kernel=3, stride=1) → ReLU
    Flatten → Linear(64*7*7, 512) → ReLU

  actor:
    mu_head:  Linear(512, 3) → tanh
    log_std:  nn.Parameter(zeros(3)), clamped to [-2, 2]

  critic:
    value_head: Linear(512, 1)

  forward(obs) → (action, log_prob, value, entropy)
    std = log_std.exp()
    dist = Normal(mu, std)
    action = dist.sample().clamp(-1, 1)
    log_prob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
```

Weight init: orthogonal_ with gain=sqrt(2) for CNN, gain=0.01 for mu_head, gain=1.0 for value_head.

---

## Hyperparameters (configs/default.yaml — source of truth)
```yaml
env:
  n_envs: 8
  frame_stack: 4
  obs_size: 84

training:
  total_timesteps: 5_000_000
  rollout_steps: 128
  n_epochs: 4
  minibatch_size: 256
  lr: 3.0e-4
  lr_schedule: linear
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.015

logging:
  eval_interval: 50_000
  eval_episodes: 10
  checkpoint_interval: 100_000
  wandb_project: carracing-ppo
  gif_interval: 500_000
```

---

## PPO loss (exact)
```python
ratio = (new_log_prob - old_log_prob).exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
value_loss = F.mse_loss(value_pred.squeeze(), returns)
loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
```
Normalize advantages before the update: `(adv - adv.mean()) / (adv.std() + 1e-8)`
Early-stop epoch if `approx_kl > target_kl`.

---

## W&B logging — log these exact keys every update step
- `train/ep_reward_mean` — mean episode reward (only when episodes complete)
- `train/ep_len_mean`
- `train/policy_loss`
- `train/value_loss`
- `train/entropy`
- `train/approx_kl`
- `train/lr`
- `eval/mean_reward` — only at eval checkpoints
- `eval/std_reward`

---

## GitHub workflow (automated via GitHub MCP)
After each phase verification passes:
```
git add -A
git commit -m "phase N: <description> — verification passed"
```
Push after phases 5, 8, 11.
Tag v0.1 after Phase 11: `git tag -a v0.1 -m "initial training pipeline complete"`

---

## Code standards
- Type hints on all functions
- Docstring on every class and public method
- No magic numbers — all hyperparams from Hydra config
- Black formatting (enforced by post-write hook)
- Files stay under 200 lines — split if larger
- `if __name__ == '__main__'` guards on all scripts

## What NOT to do
- Do not use Stable-Baselines3 for the core algorithm
- Do not hardcode any hyperparameter in Python files
- Do not commit .pt checkpoint files
- Do not use discrete action wrappers
- Do not skip frame stacking

## Interview talking points — be ready to explain these
1. Why PPO over DQN for continuous actions?
2. What does GAE do? (bias-variance tradeoff)
3. Why frame stacking? (Markov property, velocity info)
4. What does entropy bonus prevent? (premature policy collapse)
5. Why 8 parallel envs? (decorrelated samples for PPO)
6. What was the hardest bug you fixed? (log this as you go)
