"""EE 结果图 Figure 2：PPO 训练学习曲线（validation MAE vs timesteps）。

产出：
  figures/fig2_learning_curve.png

依赖 numpy + matplotlib，读取 models/best_kinetics/evaluations.npz（SB3 EvalCallback 产物）。
    venv_pose/bin/python figures/make_fig2_learning_curve.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
EVAL_NPZ = ROOT / "models" / "best_kinetics" / "evaluations.npz"


def main():
    if not EVAL_NPZ.is_file():
        print(f"[跳过] 未找到 {EVAL_NPZ}")
        sys.exit(0)

    d = np.load(EVAL_NPZ)
    print(f"Keys: {list(d.files)}")
    for key in d.files:
        arr = d[key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 1:
            print(f"    first 3: {arr[:3]}")
            print(f"    last 3:  {arr[-3:]}")
        else:
            print(f"    first 3 rows (truncated): {arr[:3, :5]}")
            print(f"    last 3 rows (truncated):  {arr[-3:, :5]}")

    timesteps = d["timesteps"]
    results = d["results"]
    mae = -results.mean(axis=1) * 100.0

    print("timesteps -> MAE:")
    for t, m in zip(timesteps, mae):
        print(f"  {t} -> {m:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        timesteps, mae, color="#2563EB", lw=2, marker="o", markersize=4,
        label="Training eval (100 episodes)",
    )
    ax.axhline(
        y=4.44, color="#DC2626", ls="--", lw=1.2, zorder=1,
        label="Full evaluation MAE = 4.44 (all 107 test cycles)",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, axis="y", color="#E5E7EB", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Validation MAE (score points)")
    ax.set_title("PPO Training: Validation MAE vs. Timesteps")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax.annotate(
        f"{mae[-1]:.2f} (training eval, 100 episodes)",
        xy=(timesteps[-1], mae[-1]),
        xytext=(-70, 20),
        textcoords="offset points",
        fontsize=9,
        color="#1F2937",
        arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.8, shrinkA=0, shrinkB=4),
    )

    fig.tight_layout()
    out = ROOT / "figures" / "fig2_learning_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"MAE first={mae[0]:.2f} last={mae[-1]:.2f}")


if __name__ == "__main__":
    main()
