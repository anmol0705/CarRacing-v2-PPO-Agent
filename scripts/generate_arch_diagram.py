"""Create a professional architecture diagram of the PPO network as a PNG."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
fig.patch.set_facecolor("#0d0d0f")
ax.set_facecolor("#0d0d0f")
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")

C_BLUE = "#3B8BD4"
C_GREEN = "#1D9E75"
C_AMBER = "#BA7517"
C_CORAL = "#D85A30"
C_PURPLE = "#7F77DD"
C_GRAY = "#444450"
C_TEXT = "#e8e4dc"
C_MUTED = "#888880"
C_BG2 = "#161618"


def draw_box(x, y, w, h, label, sublabel, color):
    shadow = FancyBboxPatch(
        (x + 0.04, y - 0.04), w, h,
        boxstyle="round,pad=0.05", linewidth=0,
        facecolor="#000000", alpha=0.4, zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05", linewidth=1,
        edgecolor=color, facecolor=C_BG2, alpha=0.95, zorder=2,
    )
    ax.add_patch(box)
    accent = FancyBboxPatch(
        (x + 0.05, y + h - 0.14), w - 0.1, 0.1,
        boxstyle="round,pad=0.02", linewidth=0,
        facecolor=color, alpha=0.9, zorder=3,
    )
    ax.add_patch(accent)
    ax.text(x + w / 2, y + h / 2 + 0.1, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C_TEXT, zorder=4)
    ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
            ha="center", va="center", fontsize=7,
            color=C_MUTED, zorder=4)


def arrow(x1, y1, x2, y2, color=C_MUTED):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=5)


for i in range(4):
    offset = i * 0.08
    rect = FancyBboxPatch(
        (0.3 + offset, 1.5 + offset), 1.0, 1.0,
        boxstyle="round,pad=0.02", linewidth=0.5,
        edgecolor=C_BLUE, facecolor="#1a2535", alpha=0.8, zorder=2 + i,
    )
    ax.add_patch(rect)
ax.text(0.9, 3.0, "Input", ha="center", fontsize=9, fontweight="bold", color=C_TEXT)
ax.text(0.9, 2.78, "4 × 84×84", ha="center", fontsize=7, color=C_MUTED)
ax.text(0.9, 2.58, "grayscale", ha="center", fontsize=7, color=C_MUTED)
ax.text(0.9, 2.38, "frames", ha="center", fontsize=7, color=C_MUTED)

arrow(1.42, 2.5, 1.85, 2.5)

layers = [
    (1.9, 1.8, 1.4, 1.4, "Conv2d 1", "32×8×8\nstride 4", C_GREEN),
    (3.5, 1.8, 1.4, 1.4, "Conv2d 2", "64×4×4\nstride 2", C_GREEN),
    (5.1, 1.8, 1.4, 1.4, "Conv2d 3", "64×3×3\nstride 1", C_GREEN),
    (6.7, 1.8, 1.4, 1.4, "Flatten", "3136-dim\nvector", C_GRAY),
    (8.3, 1.8, 1.4, 1.4, "Linear", "512-dim\nReLU", C_AMBER),
]
for x, y, w, h, lbl, sub, col in layers:
    draw_box(x, y, w, h, lbl, sub, col)

for i in range(len(layers) - 1):
    x1 = layers[i][0] + layers[i][2]
    x2 = layers[i + 1][0]
    arrow(x1, 2.5, x2, 2.5)

ax.text(5.45, 3.55, "Shared CNN Backbone", ha="center", fontsize=8,
        color=C_MUTED, style="italic")
ax.plot([1.9, 9.7], [3.42, 3.42], "--", color=C_GRAY, lw=0.5, alpha=0.4)

arrow(9.7, 2.5, 10.4, 3.3, color=C_CORAL)
arrow(9.7, 2.5, 10.4, 1.7, color=C_PURPLE)

draw_box(10.4, 2.9, 1.6, 1.0, "Actor", "μ, log σ\n3 actions", C_CORAL)
ax.annotate("", xy=(12.5, 3.5), xytext=(12.0, 3.4),
            arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.0), zorder=5)
out_box = FancyBboxPatch(
    (12.5, 3.1), 1.3, 0.8,
    boxstyle="round,pad=0.04", linewidth=0.8,
    edgecolor=C_CORAL, facecolor="#2a1510", zorder=2,
)
ax.add_patch(out_box)
ax.text(13.15, 3.55, "Actions", ha="center", fontsize=8, fontweight="bold", color=C_CORAL)
ax.text(13.15, 3.3, "steer, gas, brake", ha="center", fontsize=6.5, color=C_MUTED)

draw_box(10.4, 1.1, 1.6, 1.0, "Critic", "V(s)\nscalar", C_PURPLE)
ax.annotate("", xy=(12.5, 1.65), xytext=(12.0, 1.6),
            arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.0), zorder=5)
val_box = FancyBboxPatch(
    (12.5, 1.3), 1.3, 0.7,
    boxstyle="round,pad=0.04", linewidth=0.8,
    edgecolor=C_PURPLE, facecolor="#1a1535", zorder=2,
)
ax.add_patch(val_box)
ax.text(13.15, 1.72, "Value V(s)", ha="center", fontsize=8, fontweight="bold", color=C_PURPLE)
ax.text(13.15, 1.47, "expected return", ha="center", fontsize=6.5, color=C_MUTED)

legend_items = [
    (C_BLUE, "Input"), (C_GREEN, "Conv layers"), (C_AMBER, "FC layer"),
    (C_CORAL, "Actor head"), (C_PURPLE, "Critic head"),
]
for i, (col, lbl) in enumerate(legend_items):
    lx = 0.3 + i * 2.7
    ax.plot([lx], [0.35], "s", color=col, markersize=7)
    ax.text(lx + 0.15, 0.35, lbl, va="center", fontsize=7, color=C_MUTED)

ax.text(7.0, 4.7, "ActorCritic Network \u2014 1,782,275 Parameters",
        ha="center", fontsize=11, fontweight="bold", color=C_TEXT)

plt.tight_layout(pad=0.3)
plt.savefig("assets/architecture_diagram.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("Saved assets/architecture_diagram.png")
