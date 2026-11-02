"""EE 结果图：原始运动学在多大程度上决定技术分（to-what-extent 的定量证据）。

产出：
  figures/fig_learnability_raw.png      —— 仅原始运动学预测技术分的可学习度（MAE vs 训练量）
  figures/fig_ppo_learning_curve.png    —— PPO 在 28 维原始观测上的验证集学习曲线

依赖 numpy + matplotlib（不需 sb3 即可出第一张；第二张读 EvalCallback 的 evaluations.npz）。
    venv_pose/bin/python figures/learnability_curve.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.swimxyz_loader import SwimXYZLoader
from src.feature_extractor import FeatureExtractor
from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES
from src.kick_env import BreaststrokeKickEnv

FIGDIR = ROOT / "figures"
EVAL_NPZ = ROOT / "models" / "raw28" / "best" / "evaluations.npz"
LABELS = "data/swimxyz/Breaststroke"
TEMPLATE = "data/templates/swimxyz_template.npy"
MAXSEQ = 250


def build_dataset():
    loader = SwimXYZLoader(LABELS, stroke="breaststroke")
    ext = FeatureExtractor(fps=30)
    aligner = DTWAligner(TEMPLATE)
    if aligner.template.shape[0] > _MAX_CYCLE_FRAMES:
        tc = aligner.segment_kick_cycles(aligner.template)
        if tc:
            lens = [c.shape[0] for c in tc]
            aligner.template = tc[int(np.argsort(lens)[len(lens) // 2])]
    cycles = []
    for p in loader.list_sequences()[:MAXSEQ]:
        try:
            f = ext.extract_sequence_features(loader.load_sequence(p))
        except Exception:
            continue
        if f.shape[0]:
            cycles.extend(aligner.segment_kick_cycles(f))
    env = BreaststrokeKickEnv(cycles, aligner, seed=0)
    OBS = np.asarray(env._cached_obs, dtype=np.float64)          # (M,28) 原始运动学
    TGT = np.asarray(env._cached_target_score, dtype=np.float64)  # (M,)
    return OBS, TGT


def knn_mae(Xtr, ytr, Xte, yte, k=7):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    preds = []
    for q in Zte:
        d = np.linalg.norm(Ztr - q, axis=1)
        nn = np.argsort(d)[:k]
        preds.append(round(np.mean(ytr[nn]) / 5) * 5)
    return float(np.mean(np.abs(np.array(preds) - yte)))


def fig_learnability(OBS, TGT):
    rng = np.random.default_rng(0)
    M = len(TGT)
    perm = rng.permutation(M)
    nte = int(M * 0.3)
    te, pool = perm[:nte], perm[nte:]
    Xte, yte = OBS[te], TGT[te]
    sizes = [s for s in [20, 40, 80, 160, 320, 640] if s <= len(pool)] + [len(pool)]
    sizes = sorted(set(sizes))
    maes = [knn_mae(OBS[pool[:s]], TGT[pool[:s]], Xte, yte) for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=130)
    ax.plot(sizes, maes, "o-", color="#d8572a", lw=2.2, ms=7,
            label="Learn score from RAW kinematics only (what the agent sees)")
    ax.axhline(0.0, ls="--", color="#2a9d8f", lw=2,
               label="Reference formula WITH DTW deviation (the 'answer') → MAE = 0")
    ax.set_xlabel("Number of training kick-cycles")
    ax.set_ylabel("Validation MAE  (score points, 0–100)")
    ax.set_title("How far do raw kinematics determine the technical score?\n"
                 "(non-zero gap = genuine learning, not reading the answer)", fontsize=11)
    ax.set_ylim(bottom=-0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.annotate(f"converges to ~{maes[-1]:.1f} pts",
                xy=(sizes[-1], maes[-1]), xytext=(-150, 28),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#d8572a"))
    fig.tight_layout()
    out = FIGDIR / "fig_learnability_raw.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[图] {out.name}  | 训练量={sizes}  MAE={[round(m,1) for m in maes]}")
    return maes[-1]


def fig_ppo_curve():
    if not EVAL_NPZ.is_file():
        print(f"[跳过 PPO 曲线] 未找到 {EVAL_NPZ}（训练完成后会生成）")
        return
    d = np.load(EVAL_NPZ)
    ts = d["timesteps"]
    results = d["results"]                 # (n_evals, n_episodes) rewards
    mae = -results.mean(axis=1) * 100.0    # reward=-|err|/100 → MAE = -mean_reward*100
    std = results.std(axis=1) * 100.0
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=130)
    ax.plot(ts, mae, "o-", color="#264653", lw=2.2, ms=6, label="PPO validation MAE")
    ax.fill_between(ts, mae - std, mae + std, color="#264653", alpha=0.15)
    ax.set_xlabel("PPO training timesteps")
    ax.set_ylabel("Validation MAE  (score points)")
    ax.set_title("PPO learns to score from 28-D raw kinematics\n"
                 "(MAE > 0 throughout → the agent is genuinely learning, not a calculator)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig_ppo_learning_curve.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"[图] {out.name}  | 最终 PPO val MAE={mae[-1]:.1f} 分")


if __name__ == "__main__":
    OBS, TGT = build_dataset()
    print(f"数据集: {len(TGT)} 个蹬腿周期, 观测维度={OBS.shape[1]}")
    fig_learnability(OBS, TGT)
    fig_ppo_curve()
