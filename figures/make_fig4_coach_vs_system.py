from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# --- Data ---
# Source: field study, 2026-08. 7 self-recorded videos scored by the PPO
# agent and blind-rated by a coach who was unaware the system existed. The
# coach gave one score per swimmer (not per video): Swimmer A (author) = 40,
# Swimmer B = 20.
DATA = [
    # video,      swimmer,              system_score, coach_score
    ("IMG_4641", "Swimmer A (author)", 30, 40),
    ("IMG_4642", "Swimmer A (author)", 25, 40),
    ("IMG_4644", "Swimmer A (author)", 25, 40),
    ("IMG_4645", "Swimmer B", 30, 20),
    ("IMG_4646", "Swimmer B", 30, 20),
    ("IMG_4648", "Swimmer B", 30, 20),
    ("IMG_4650", "Swimmer B", 35, 20),
]

COLOR_A = "#2563EB"  # filled circles, Swimmer A
COLOR_B = "#DC2626"  # open squares, Swimmer B

# Group rows that land on the exact same (system_score, coach_score) point
# so only one marker is drawn per point, with a combined video-ID label.
points: dict[tuple[str, int, int], list[str]] = {}
for video, swimmer, sys_score, coach_score in DATA:
    key = (swimmer, sys_score, coach_score)
    points.setdefault(key, []).append(video.replace("IMG_", ""))

# Points that share the same coach_score sit on the same horizontal row and
# their labels would otherwise collide; stagger the vertical offset
# deterministically by x-position among points on that row.
by_row: dict[int, list[tuple[str, int, int]]] = {}
for swimmer, sys_score, coach_score in points:
    by_row.setdefault(coach_score, []).append((swimmer, sys_score, coach_score))
label_dy: dict[tuple[str, int, int], int] = {}
for coach_score, keys in by_row.items():
    for i, key in enumerate(sorted(keys, key=lambda k: k[1])):
        label_dy[key] = 22 if i % 2 else 10

fig, ax = plt.subplots(figsize=(7, 4.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, zorder=1)

plotted_a = False
plotted_b = False
for (swimmer, sys_score, coach_score), video_ids in points.items():
    if swimmer == "Swimmer A (author)":
        ax.scatter(
            sys_score, coach_score,
            marker="o", s=55, facecolor=COLOR_A, edgecolor=COLOR_A,
            zorder=3,
            label="Swimmer A (author)" if not plotted_a else None,
        )
        plotted_a = True
    else:
        ax.scatter(
            sys_score, coach_score,
            marker="s", s=55, facecolor="none", edgecolor=COLOR_B, linewidth=1.5,
            zorder=3,
            label="Swimmer B" if not plotted_b else None,
        )
        plotted_b = True
    ax.annotate(
        ", ".join(video_ids),
        xy=(sys_score, coach_score),
        xytext=(0, label_dy[(swimmer, sys_score, coach_score)]),
        textcoords="offset points",
        fontsize=7,
        ha="center",
        color="dimgray",
    )

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_yticks(np.arange(0, 101, 10))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.yaxis.grid(True, linestyle="--", color="lightgray", alpha=0.7)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.set_xlabel("System score (PPO agent)")
ax.set_ylabel("Blinded coach rating")
ax.set_title("System score vs. blinded coach rating (field study, n=7 videos)")

ax.legend(loc="lower right", fontsize=8, frameon=False)

out_path = ROOT / "figures" / "fig4_coach_vs_system.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")

print(f"Saved figure to: {out_path.resolve()}")
