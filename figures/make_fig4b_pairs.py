"""EE 结果图 Figure 4b：教练盲评分 vs 系统分配对哑铃图（per-video dumbbell plot）。

每一行代表一个视频，用一条浅灰细线连接该视频的教练盲评分（实心 marker）
与系统分（PPO agent，空心 marker）。按泳者分组：Swimmer A (author) 在上，
Swimmer B 在下，颜色/形状随泳者区分（A=蓝色圆，B=红色方）。

产出：
  figures/fig4b_coach_vs_system_pairs.png

数据来源：field study, 2026-08。7 个自录视频，系统分来自 PPO agent，
教练盲评分为教练在不知系统存在的情况下按泳者给出的统一评分
（Swimmer A (author) = 40，Swimmer B = 20）。

运行方式：
    venv_pose/bin/python figures/make_fig4b_pairs.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# --- Data ---
# Source: field study, 2026-08. Coach gave one score per swimmer (not per
# video): Swimmer A (author) = 40, Swimmer B = 20.
DATA = [
    # video,      swimmer,              system_score, coach_score
    ("4641", "Swimmer A (author)", 30, 40),
    ("4642", "Swimmer A (author)", 25, 40),
    ("4644", "Swimmer A (author)", 25, 40),
    ("4645", "Swimmer B", 30, 20),
    ("4646", "Swimmer B", 30, 20),
    ("4648", "Swimmer B", 30, 20),
    ("4650", "Swimmer B", 35, 20),
]

COLOR_A = "#2563EB"
COLOR_B = "#DC2626"

group_a = [row for row in DATA if row[1] == "Swimmer A (author)"]
group_b = [row for row in DATA if row[1] == "Swimmer B"]

# Build y-positions top-to-bottom: group A rows, one blank row, group B rows.
rows = []  # list of (y, video, swimmer, system_score, coach_score) or None for blank
y = len(group_a) + 1 + len(group_b)
for row in group_a:
    rows.append((y, *row))
    y -= 1
y -= 1  # blank gap row
for row in group_b:
    rows.append((y, *row))
    y -= 1

fig, ax = plt.subplots(figsize=(7, 3.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y_pos, video, swimmer, sys_score, coach_score in rows:
    color = COLOR_A if swimmer == "Swimmer A (author)" else COLOR_B
    marker = "o" if swimmer == "Swimmer A (author)" else "s"

    ax.plot([sys_score, coach_score], [y_pos, y_pos], color="lightgray", linewidth=1.5, zorder=1)
    ax.scatter(coach_score, y_pos, marker=marker, s=55, facecolor=color, edgecolor=color, zorder=3)
    ax.scatter(sys_score, y_pos, marker=marker, s=55, facecolor="none", edgecolor=color, linewidth=1.5, zorder=3)

ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows])  # video IDs (rows[i] = (y, video, swimmer, sys, coach))
ax.set_ylim(0.3, len(group_a) + len(group_b) + 1.7)

# Group labels on the left, colored by swimmer.
y_a_mid = np.mean([r[0] for r in rows[: len(group_a)]])
y_b_mid = np.mean([r[0] for r in rows[len(group_a):]])
ax.text(-0.14, y_a_mid, "Swimmer A (author)", transform=ax.get_yaxis_transform(),
        fontsize=9, color=COLOR_A, ha="right", va="center", rotation=90)
ax.text(-0.14, y_b_mid, "Swimmer B", transform=ax.get_yaxis_transform(),
        fontsize=9, color=COLOR_B, ha="right", va="center", rotation=90)

ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xlabel("Score (0–100)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.xaxis.grid(True, linestyle="--", color="lightgray", alpha=0.7)
ax.yaxis.grid(False)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

# Neutral gray legend handles so they don't compete with the swimmer colors.
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dimgray",
               markeredgecolor="dimgray", markersize=8, label="Blinded coach rating"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="dimgray", markeredgewidth=1.5, markersize=8,
               label="System score (PPO agent)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()

out_path = ROOT / "figures" / "fig4b_coach_vs_system_pairs.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")

print(f"Saved figure to: {out_path.resolve()}")
