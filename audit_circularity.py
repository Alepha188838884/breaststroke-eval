"""审计：验证 reference_score 是（周期特征, 模板）的确定性闭式函数、不依赖任何外部
真值；并给出仅从 28 维原始运动学观测预测该分数的 kNN 基线（PPO 的可学习性参照）。

背景：RL 环境 BreaststrokeKickEnv 的观测（obs）只含 28 维原始运动学摘要（角度/
对称/角速度统计量），**不含**任何 DTW 派生量；参考分数（reward 的目标）则由
模板对齐后的 DTW 阶段偏差 + 对称性 + timing + 平滑度按固定公式算出。二者的关系
是：obs 完全看不到参考分公式的输入，PPO 必须单纯从原始运动学学习预测该公式。

本脚本做两件事：
    1. 从 env 缓存的 DTW 中间量（``_cached_overall_deviation`` /
       ``_cached_warp_amount``）与周期本身，复刻 ``_compute_reference_score``
       的计算，验证其与 env 实际给出的训练目标（``_cached_target_score``）
       完全一致（MAE≈0）——证明参考分数是确定性闭式函数，无外部真值参与。
    2. 用仅有的 28 维原始运动学 obs 做 kNN(k=5) 基线，衡量"原始动作能在多大
       程度上决定该参考分"，其残差正是 EE 研究问题 to-what-extent 的量化对象。

只需 gymnasium + numpy（不需 stable_baselines3）。
    venv_pose/bin/python audit_circularity.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.swimxyz_loader import SwimXYZLoader
from src.feature_extractor import FeatureExtractor
from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES
from src.kick_env import (
    BreaststrokeKickEnv,
    _ANGLE_DEV_MAX,
    _SYMMETRY_DEV_MAX,
    _TIMING_WARP_MAX,
    _SMOOTHNESS_MAX,
    _REFERENCE_WEIGHTS,
    _SYMMETRY_COLS,
    _VELOCITY_COLS,
    _MAX_SCORE,
    _SCORE_PER_ACTION,
)

LABELS = "data/swimxyz/Breaststroke"
TEMPLATE = "data/templates/swimxyz_template.npy"
MAXSEQ = 40

# ---- 1) 复刻训练数据：SwimXYZ → 特征 → 蹬腿周期 → 环境（含缓存的 obs 与 target）----
loader = SwimXYZLoader(LABELS, stroke="breaststroke")
paths = loader.list_sequences()[:MAXSEQ]
ext = FeatureExtractor(fps=30)
aligner = DTWAligner(TEMPLATE)
if aligner.template.shape[0] > _MAX_CYCLE_FRAMES:           # 模板若是多周期，切中位单周期
    tc = aligner.segment_kick_cycles(aligner.template)
    if tc:
        lens = [c.shape[0] for c in tc]
        aligner.template = tc[int(np.argsort(lens)[len(lens) // 2])]

cycles = []
for p in paths:
    try:
        feats = ext.extract_sequence_features(loader.load_sequence(p))
    except Exception:
        continue
    if feats.shape[0]:
        cycles.extend(aligner.segment_kick_cycles(feats))
print(f"序列={len(paths)}  分割出蹬腿周期={len(cycles)}")

env = BreaststrokeKickEnv(cycles, aligner, seed=0)
OBS = np.asarray(env._cached_obs, dtype=np.float64)           # (M, 28) 纯原始运动学
TGT = np.asarray(env._cached_target_score, dtype=np.float64)  # (M,)  RL 的训练目标
M = len(TGT)


# ---- 2) 从 env 缓存的 DTW 中间量 + 周期本身复刻 _compute_reference_score ----
# 不从观测反推：obs 里已经没有任何 DTW 派生量，反推不可行也不是本次要证明的东西。
# 这里做的是"用与 kick_env 完全相同的输入（cycle + 缓存的 overall_deviation /
# warp_amount）复刻同一公式"，验证该公式确定性地产出了训练目标——不依赖 obs。
def reconstruct_reference_score(
    cycle: np.ndarray, overall_dev: float, warp_amount: float
) -> float:
    angle = np.clip(overall_dev / _ANGLE_DEV_MAX, 0.0, 1.0)
    symmetry = np.clip(cycle[:, _SYMMETRY_COLS].mean() / _SYMMETRY_DEV_MAX, 0.0, 1.0)
    timing = np.clip(warp_amount / _TIMING_WARP_MAX, 0.0, 1.0)
    smoothness = np.clip(cycle[:, _VELOCITY_COLS].std() / _SMOOTHNESS_MAX, 0.0, 1.0)
    weighted = (
        angle * _REFERENCE_WEIGHTS["angle"]
        + symmetry * _REFERENCE_WEIGHTS["symmetry"]
        + timing * _REFERENCE_WEIGHTS["timing"]
        + smoothness * _REFERENCE_WEIGHTS["smoothness"]
    )
    score_raw = (1.0 - weighted) * _MAX_SCORE
    score_clipped = max(0.0, min(_MAX_SCORE, score_raw))
    return round(score_clipped / _SCORE_PER_ACTION) * _SCORE_PER_ACTION


REC = np.array(
    [
        reconstruct_reference_score(
            env.kick_cycles[i],
            env._cached_overall_deviation[i],
            env._cached_warp_amount[i],
        )
        for i in range(M)
    ]
)

err = np.abs(REC - TGT)
print("\n================ 证明：参考分数是确定性闭式函数（无外部真值） ================")
print(f"用『周期特征 + 缓存的 DTW 中间量』复刻公式 vs env 实际训练目标（{M} 个周期）：")
print(f"   完全相同的比例 = {100*np.mean(err < 1e-9):.1f}%")
print(f"   平均绝对误差 MAE = {err.mean():.4f} 分")
print(f"   最大误差 = {err.max():.1f} 分")
print(
    "→ 目标分数是（周期特征, 模板）的确定性闭式函数，计算全程无任何外部人工标注/"
    "真值参与——这不是训练学出来的关系，而是公式的定义本身。"
)

# ---- 3) 『零参数公式策略』直接输出该公式 → reward 完美，无需任何训练 ----
# RL reward = -|agent - target|/100；公式策略 agent=REC，故 reward≈0（满分）
formula_reward = -np.mean(err) / 100.0
print(f"\n零参数『公式策略』(直接输出复刻分) 的平均 reward = {formula_reward:.5f}  (满分=0)")
print("→ 不训练、不用神经网络即达最优；PPO 智能体最多与之打平，无法提供额外信息。")

# ---- 4) 目标分布：评估是否只是窄带 ----
uniq = sorted(set(TGT.tolist()))
print(f"\n目标分数分布: min={TGT.min():.0f} max={TGT.max():.0f} "
      f"使用档位={len(uniq)}/21  常见值={uniq[:8]}")

# ---- 5) obs 只含 28 维原始运动学（无 DTW 派生量）：kNN(k=5) 基线能预测到什么程度？----
RAW = OBS  # obs 本身就是 28 维原始运动学，无需再切片
mu, sd = RAW.mean(0), RAW.std(0) + 1e-9
Z = (RAW - mu) / sd
rng = np.random.default_rng(0)
perm = rng.permutation(M)
ntr = int(M * 0.7)
tr, te = perm[:ntr], perm[ntr:]


def knn_pred(q: np.ndarray, k: int = 5) -> float:
    d = np.linalg.norm(Z[tr] - q, axis=1)
    nn = tr[np.argsort(d)[:k]]
    return round(np.mean(TGT[nn]) / 5) * 5


pred = np.array([knn_pred(Z[i]) for i in te])
mae_raw = np.mean(np.abs(pred - TGT[te]))
formula_mae = err.mean()
print("\n================ 仅原始运动学（28 维 obs）能多大程度预测参考分？================")
print(f"公式复刻（周期特征 + 缓存 DTW 中间量）预测参考分 : MAE = {formula_mae:.4f} 分（≈0，因为它就是公式本身）")
print(f"仅原始运动学 obs 预测参考分(kNN, 70/30 划分)     : MAE = {mae_raw:.1f} 分（>0 → PPO 必须真正学习）")
print("→ 后者的残差正是 EE『to what extent』要量化的：原始动作能多大程度决定技术分。")
