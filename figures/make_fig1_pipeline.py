"""EE 架构图 Figure 1：蛙泳蹬腿评估系统六模块流水线示意图。

产出：
  figures/fig1_pipeline.png

纯 matplotlib 绘图，不依赖任何运行数据，可直接执行：
    venv_pose/bin/python figures/make_fig1_pipeline.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent

BLUE = "#2563EB"
DARK = "#1F2937"
GRAY = "#9CA3AF"
LIGHT_GRAY = "#E5E7EB"
BOX_FILL = "#F9FAFB"

# Module definitions: (name, subtitle, row, col, highlight)
MODULES = [
    ("VideoLoader", "Frame extraction &\nnormalisation", 0, 0, False),
    ("PoseEstimator", "HRNet joint detection\n(17 keypoints)", 0, 1, False),
    ("FeatureExtractor", "Biomechanical features\n(angles, symmetry)", 0, 2, False),
    ("DTWAligner", "Kick-cycle segmentation &\ntemplate alignment", 1, 2, False),
    ("RL Scoring Env +\nPPO Agent", "Single-step episode →\nscore 0–100", 1, 1, True),
    ("FeedbackGenerator", "Natural-language\nsuggestions", 1, 0, False),
]

BOX_W, BOX_H = 2.55, 1.15
COL_X = [0.6, 3.5, 6.4]
ROW_Y = [3.2, 1.0]

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(-0.6, 9.6)
ax.set_ylim(0.0, 4.9)
ax.axis("off")

box_centers = {}

for name, subtitle, row, col, highlight in MODULES:
    x, y = COL_X[col], ROW_Y[row]
    box_centers[(row, col)] = (x + BOX_W / 2, y + BOX_H / 2)

    if highlight:
        face, edge, lw = "#DBEAFE", BLUE, 2.4
        name_color = BLUE
    else:
        face, edge, lw = BOX_FILL, "#9CA3AF", 1.2
        name_color = DARK

    box = FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(box)

    cx, cy = x + BOX_W / 2, y + BOX_H / 2
    ax.text(
        cx, cy + 0.24, name,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=name_color,
        zorder=3,
    )
    ax.text(
        cx, cy - 0.22, subtitle,
        ha="center", va="center",
        fontsize=8, color="#6B7280",
        zorder=3,
        linespacing=1.35,
    )

    if highlight:
        ax.text(
            x + BOX_W - 0.06, y + BOX_H - 0.02, "★",
            ha="right", va="top", fontsize=9, color=BLUE, zorder=3,
        )


def draw_arrow(p0, p1, color=GRAY, lw=1.6):
    arrow = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, zorder=1,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


# Input label -> VideoLoader
x0, y0 = COL_X[0], ROW_Y[0]
ax.text(
    x0 + BOX_W / 2, ROW_Y[0] + BOX_H + 0.55, "Input video\n(side-view kick clip)",
    ha="center", va="center", fontsize=8.5, color=DARK, linespacing=1.3,
)
draw_arrow((x0 + BOX_W / 2, ROW_Y[0] + BOX_H + 0.28), (x0 + BOX_W / 2, ROW_Y[0] + BOX_H))

# Row 0: VideoLoader -> PoseEstimator -> FeatureExtractor
for c in range(2):
    x_a = COL_X[c] + BOX_W
    x_b = COL_X[c + 1]
    y = ROW_Y[0] + BOX_H / 2
    draw_arrow((x_a, y), (x_b, y))

# Fold down: FeatureExtractor (row0,col2) -> DTWAligner (row1,col2)
x_c = COL_X[2] + BOX_W / 2
draw_arrow((x_c, ROW_Y[0]), (x_c, ROW_Y[1] + BOX_H))

# Row 1: DTWAligner -> RL Scoring Env -> FeedbackGenerator (right to left)
for c in [2, 1]:
    x_a = COL_X[c]
    x_b = COL_X[c - 1] + BOX_W
    y = ROW_Y[1] + BOX_H / 2
    draw_arrow((x_a, y), (x_b, y))

# Output label <- FeedbackGenerator
x_f = COL_X[0] + BOX_W / 2
draw_arrow((x_f, ROW_Y[1]), (x_f, ROW_Y[1] - 0.28))
ax.text(
    x_f, ROW_Y[1] - 0.55, "Score + feedback",
    ha="center", va="center", fontsize=8.5, color=DARK, fontweight="bold",
)

# Legend
legend_y = 4.65
ax.add_patch(FancyBboxPatch(
    (0.6, legend_y - 0.09), 0.28, 0.18,
    boxstyle="round,pad=0.01,rounding_size=0.03",
    linewidth=2.0, edgecolor=BLUE, facecolor="#DBEAFE",
))
ax.text(1.0, legend_y, "Original contribution", ha="left", va="center", fontsize=8, color=DARK)

ax.add_patch(FancyBboxPatch(
    (3.6, legend_y - 0.09), 0.28, 0.18,
    boxstyle="round,pad=0.01,rounding_size=0.03",
    linewidth=1.0, edgecolor="#9CA3AF", facecolor=BOX_FILL,
))
ax.text(4.0, legend_y, "Built on existing libraries", ha="left", va="center", fontsize=8, color=DARK)

fig.tight_layout()
out = ROOT / "figures" / "fig1_pipeline.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out}")
