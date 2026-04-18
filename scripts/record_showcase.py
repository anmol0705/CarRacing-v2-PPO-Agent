"""Record the best episode with professional HUD overlay.

Reads best_episode_info.json for checkpoint + seed, then re-runs
the episode deterministically and renders 600x480 frames with
sidebar (actions, steering history) and bottom HUD bar.
"""

import json
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env_utils import Float64Action, NormalizeObservation
from src.model import ActorCritic

with open("assets/best_episode_info.json") as f:
    info = json.load(f)

CHECKPOINT = info["checkpoint"]
SEED = info["seed"]
DEVICE = torch.device("cpu")
OUT_PATH = "assets/showcase.gif"
FPS = 30
MAX_STEPS = 1200

print(f"Recording: ckpt={CHECKPOINT}, seed={SEED}, expected reward={info['reward']:.1f}")

model = ActorCritic().to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state["model_state_dict"])
model.eval()


def make_wrapped_env(seed: int) -> gym.Env:
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    env = Float64Action(env)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    env = NormalizeObservation(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


env_w = make_wrapped_env(SEED)
obs_w, _ = env_w.reset(seed=SEED)

env_r = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
env_r = Float64Action(env_r)
obs_r, _ = env_r.reset(seed=SEED)

frames_raw = []
actions_log = []
rewards_log = []
total_reward = 0.0

try:
    for step in range(MAX_STEPS):
        obs_t = torch.FloatTensor(np.array(obs_w)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = model.get_greedy_action(obs_t)
        action_np = action.cpu().numpy()[0]

        obs_w, rw, tw, trw, _ = env_w.step(action_np)
        obs_r, rr, tr, trr, _ = env_r.step(action_np)
        total_reward += rw

        frame = env_r.render()
        frames_raw.append(frame)
        actions_log.append(action_np.copy())
        rewards_log.append(total_reward)

        if tw or trw:
            break
        if step % 100 == 0:
            print(f"  Step {step:4d} | reward {total_reward:7.1f}")
finally:
    env_w.close()
    env_r.close()

print(f"Episode done: {len(frames_raw)} steps, reward {total_reward:.1f}")

# ── HUD rendering ────────────────────────────────────────────────────────────

CANVAS_W, CANVAS_H = 600, 480
GAME_W, GAME_H = 480, 380
SIDEBAR_X = GAME_W + 8
SIDEBAR_W = CANVAS_W - GAME_W - 16

BG = (12, 12, 16)
PANEL = (20, 22, 28)
ACCENT = (59, 139, 212)
GREEN = (76, 175, 80)
AMBER = (255, 180, 50)
RED = (220, 80, 60)
WHITE = (230, 230, 230)
GRAY = (100, 100, 110)
DIMGRAY = (35, 35, 45)
DARKBG = (30, 30, 40)


def draw_bar(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
             frac: float, color: tuple, bg: tuple = DARKBG) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    if frac > 0:
        draw.rectangle([x, y, x + int(w * min(frac, 1.0)), y + h], fill=color)


def draw_steer_bar(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
                   steer: float, color: tuple) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=DARKBG)
    mid = x + w // 2
    if steer >= 0:
        draw.rectangle([mid, y, mid + int(w / 2 * steer), y + h], fill=color)
    else:
        draw.rectangle([mid + int(w / 2 * steer), y, mid, y + h], fill=color)


def make_frame(raw_frame: np.ndarray, step_i: int, total_r: float,
               action: np.ndarray, all_actions: list) -> np.ndarray:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas)

    game_img = Image.fromarray(raw_frame).resize((GAME_W, GAME_H), Image.LANCZOS)
    canvas.paste(game_img, (0, 0))
    draw.rectangle([0, 0, GAME_W - 1, GAME_H - 1], outline=(40, 40, 50), width=1)

    sx = SIDEBAR_X
    sw = SIDEBAR_W

    draw.text((sx, 14), "PPO AGENT", fill=ACCENT)
    draw.text((sx, 30), "CarRacing-v2", fill=GRAY)
    draw.line([(sx, 48), (sx + sw, 48)], fill=DIMGRAY, width=1)

    draw.text((sx, 56), "REWARD", fill=GRAY)
    r_color = GREEN if total_r > 500 else AMBER if total_r > 0 else RED
    draw.text((sx, 70), f"{total_r:6.0f}", fill=r_color)

    draw.text((sx, 98), "STEP", fill=GRAY)
    draw.text((sx, 112), f"{step_i:5d}", fill=WHITE)

    draw.text((sx, 140), "LAP PROGRESS", fill=GRAY)
    progress = min(1.0, step_i / 900)
    p_color = GREEN if progress > 0.8 else ACCENT
    draw_bar(draw, sx, 156, sw, 6, progress, p_color)

    draw.line([(sx, 172), (sx + sw, 172)], fill=DIMGRAY, width=1)

    steer, gas, brake = float(action[0]), float(action[1]), float(action[2])

    draw.text((sx, 180), "STEERING", fill=GRAY)
    s_color = AMBER if abs(steer) > 0.5 else ACCENT
    draw_steer_bar(draw, sx, 196, sw, 6, steer, s_color)
    draw.text((sx, 208), f"{steer:+.2f}", fill=WHITE)

    draw.text((sx, 226), "GAS", fill=GRAY)
    draw_bar(draw, sx, 242, sw, 6, gas, GREEN)
    draw.text((sx, 254), f"{gas:.2f}", fill=WHITE)

    draw.text((sx, 272), "BRAKE", fill=GRAY)
    draw_bar(draw, sx, 288, sw, 6, brake, RED)
    draw.text((sx, 300), f"{brake:.2f}", fill=WHITE)

    draw.line([(sx, 318), (sx + sw, 318)], fill=DIMGRAY, width=1)

    draw.text((sx, 326), "STEER HISTORY", fill=GRAY)
    hist_y, hist_h = 342, 30
    hist = [a[0] for a in all_actions[-40:]]
    draw.rectangle([sx, hist_y, sx + sw, hist_y + hist_h], fill=(22, 22, 30))
    mid_y = hist_y + hist_h // 2
    draw.line([(sx, mid_y), (sx + sw, mid_y)], fill=(40, 40, 50), width=1)
    if len(hist) > 1:
        pts = []
        for j, s in enumerate(hist):
            hx = sx + int(j * sw / max(len(hist) - 1, 1))
            hy = mid_y - int(s * hist_h * 0.45)
            pts.append((hx, hy))
        draw.line(pts, fill=ACCENT, width=1)

    hy0 = GAME_H
    draw.rectangle([0, hy0, GAME_W, CANVAS_H], fill=PANEL)

    r_frac = min(1.0, max(0.0, total_r / 1000))
    r_col = GREEN if r_frac > 0.7 else AMBER if r_frac > 0.3 else RED
    draw_bar(draw, 0, hy0, GAME_W, 6, r_frac, r_col, bg=(25, 25, 35))

    hud_y = hy0 + 14
    labels = [
        (10, "REWARD", f"{total_r:.0f} / 1000", WHITE),
        (130, "MODEL", "PPO Agent", ACCENT),
        (260, "SEED", f"{SEED}", WHITE),
        (330, "STEP", f"{step_i}", WHITE),
        (400, "ENV", "CarRacing-v2", WHITE),
    ]
    for lx, label, value, val_color in labels:
        draw.text((lx, hud_y), label, fill=GRAY)
        draw.text((lx, hud_y + 14), value, fill=val_color)

    return np.array(canvas)


print(f"Rendering {len(frames_raw)} frames with HUD overlay...")
processed = []
for i in range(len(frames_raw)):
    frame = make_frame(
        frames_raw[i], i + 1, rewards_log[i],
        actions_log[i], actions_log[: i + 1],
    )
    processed.append(frame)
    if (i + 1) % 100 == 0:
        print(f"  Rendered {i + 1}/{len(frames_raw)}")

print(f"Saving {OUT_PATH}...")
imageio.mimsave(OUT_PATH, processed, fps=FPS, loop=0)
size_mb = Path(OUT_PATH).stat().st_size / 1e6
print(f"Done — {OUT_PATH} ({size_mb:.1f} MB, {len(processed)} frames)")
