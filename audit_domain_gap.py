"""审计合成(SwimXYZ)→真实(HRNet champion) 的域差，并验证可行的缓解措施。

输出：
  1) 10 维原始特征 real vs synthetic 的分位对比（定位差在哪）
  2) 角速度抖动：给角度加轻度平滑后，real 的速度 std 是否回落到合成量级
  3) 4 个评分因素分布 vs 当前标定阈值，并暴露『训练用 swimxyz_template / 部署用
     champion_template』的模板不一致效应

    venv_pose/bin/python audit_domain_gap.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.swimxyz_loader import SwimXYZLoader
from src.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES
from src.kick_env import BreaststrokeKickEnv

CHAMP = "data/processed/champion.npy"
CHAMP_TPL = "data/templates/champion_template.npy"
SX_TPL = "data/templates/swimxyz_template.npy"


def load_template(path):
    al = DTWAligner(path)
    if al.template.shape[0] > _MAX_CYCLE_FRAMES:
        tc = al.segment_kick_cycles(al.template)
        if tc:
            al.template = tc[int(np.argsort([c.shape[0] for c in tc])[len(tc) // 2])]
    return al


def pct(a):
    return f"P10={np.percentile(a,10):7.1f} P50={np.percentile(a,50):7.1f} P90={np.percentile(a,90):7.1f}"


# ---------- 数据 ----------
real = np.load(CHAMP)                       # (N,10) 真实 HRNet 特征（整段）
ld = SwimXYZLoader("data/swimxyz/Breaststroke", stroke="breaststroke")
ext = FeatureExtractor(fps=30)
syn_list = []
for p in ld.list_sequences()[:60]:
    try:
        f = ext.extract_sequence_features(ld.load_sequence(p))
    except Exception:
        continue
    if f.shape[0]:
        syn_list.append(f)
syn = np.concatenate(syn_list, 0)

print("=" * 78)
print("1) 10 维原始特征：真实 HRNet  vs  SwimXYZ 合成")
print("=" * 78)
for i, name in enumerate(FEATURE_NAMES):
    print(f"{name:28s} real[{pct(real[:,i])}]  syn[{pct(syn[:,i])}]")

# ---------- 2) 角速度抖动：加平滑后回落？ ----------
print("\n" + "=" * 78)
print("2) 角速度抖动来源：HRNet 角度未平滑 → 差分放大。加轻度平滑后 real 速度 std：")
print("=" * 78)


def vel_std_after_smooth(feat, win):
    ang = feat[:, :4].copy()
    if win > 1:
        ang = uniform_filter1d(ang, size=win, axis=0, mode="nearest")
    v = np.zeros_like(ang)
    v[1:-1] = (ang[2:] - ang[:-2]) / (2.0 / 30.0)
    v[0] = (ang[1] - ang[0]) * 30.0
    v[-1] = (ang[-1] - ang[-2]) * 30.0
    return float(np.std(v))


for win in [1, 3, 5, 7]:
    print(f"   平滑窗={win:2d}帧  real 速度std={vel_std_after_smooth(real, win):6.1f}  "
          f"(合成≈{np.std(syn[:,6:10]):.0f}, 当前 _SMOOTHNESS_MAX=120)")

# ---------- 3) 因素分布 vs 阈值 + 模板效应 ----------
print("\n" + "=" * 78)
print("3) 4 个评分因素分布（P50/P90/max）vs 当前阈值；并看模板不一致的影响")
print("=" * 78)


def factors(cycles, aligner, tag):
    # 刻意不传 score_scales（用合成默认尺度），目的正是暴露真实数据在合成标定下
    # 的因素饱和；勿"修复"为传 scale_max_real。
    env = BreaststrokeKickEnv(cycles, aligner, seed=0)
    d = env.factor_distribution()
    print(f"\n[{tag}]  cycles={len(env.kick_cycles)}  target档位用={d['target_score']['n_bins_used']}/21 "
          f"范围[{d['target_score']['min']:.0f},{d['target_score']['max']:.0f}]")
    for k, mx in [("angle_overall_deviation", 20), ("symmetry_mean", 18),
                  ("timing_warp", 0.9), ("smoothness_vel_std", 120)]:
        s = d[k]
        flag = "  <-- 顶到/超阈值(饱和)" if s["p90"] >= mx else ""
        print(f"   {k:24s} P50={s['p50']:7.2f} P90={s['p90']:7.2f} max={s['max']:7.2f}  阈值={mx}{flag}")


champ_al = load_template(CHAMP_TPL)
sx_al = load_template(SX_TPL)
real_cycles = champ_al.segment_kick_cycles(real)
syn_cycles = []
for f in syn_list:
    syn_cycles.extend(sx_al.segment_kick_cycles(f))

factors(real_cycles, champ_al, "真实 champion 周期 ↔ champion模板（部署路径）")
factors(syn_cycles[:300], sx_al, "合成 SwimXYZ 周期 ↔ swimxyz模板（训练路径）")
factors(syn_cycles[:300], champ_al, "合成 SwimXYZ 周期 ↔ champion模板（跨模板，暴露不一致）")
