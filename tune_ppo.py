"""【已弃用/存档】2026-06 SwimXYZ 域内 PPO 防坍缩调参实验存档。使用旧
swimxyz_template 与默认合成尺度，与现行主链路（train_agent --source kinetics +
champion 模板 + scale_max_real）不可比。仅作复现历史结果用，勿用其产出模型。
现行入口：python3 -m src.train_agent --source kinetics
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.train_agent import build_kick_cycles, _split_sequences
from src.kick_env import BreaststrokeKickEnv

LABELS = "data/swimxyz/Breaststroke"
TEMPLATE = "data/templates/swimxyz_template.npy"


def eval_stats(model, env):
    OBS = np.asarray(env._cached_obs)
    TGT = np.asarray(env._cached_target_score)
    pred = np.array([int(model.predict(o, deterministic=True)[0]) * 5 for o in OBS])
    mae = float(np.mean(np.abs(pred - TGT)))
    best_c = round(np.median(TGT) / 5) * 5
    mae_const = float(np.mean(np.abs(TGT - best_c)))
    return mae, mae_const, len(set(pred.tolist())), float(pred.std())


def main():
    per_seq, aligner = build_kick_cycles(LABELS, TEMPLATE, fps=30, max_sequences=200)
    tr, va = _split_sequences(per_seq, train_ratio=0.8, seed=42)
    train_env = BreaststrokeKickEnv(tr, aligner, seed=42)
    val_env = BreaststrokeKickEnv(va, aligner, seed=43)

    # 防坍缩配置：更大网络 + 正 entropy 保持探索 + 更大 rollout + 更长训练
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=1024, batch_size=128, n_epochs=15,
        learning_rate=5e-4, ent_coef=0.01, gamma=0.99,
        verbose=0, seed=0,
    )
    sched = [0, 2000, 5000, 10000, 20000, 40000, 70000, 110000, 160000]
    prev = 0
    print("step    val_MAE  const_MAE  #uniq  pred_std")
    rows = []
    for t in sched:
        if t > 0:
            model.learn(t - prev, reset_num_timesteps=False)
            prev = t
        mae, mae_c, nuniq, pstd = eval_stats(model, val_env)
        rows.append((t, mae))
        print(f"{t:6d}   {mae:6.2f}   {mae_c:7.2f}    {nuniq:3d}    {pstd:5.1f}")

    final_mae = rows[-1][1]
    _, mae_c, nuniq, _ = eval_stats(model, val_env)
    if nuniq > 1 and final_mae < mae_c - 1.0:
        print(f"\n✅ PPO 真在学：val MAE={final_mae:.2f} < 常数基线 {mae_c:.2f}，预测有 {nuniq} 种取值")
        model.save("models/raw28/ppo_kick_scorer.zip")
        print("已保存 models/raw28/ppo_kick_scorer.zip")
    else:
        print(f"\n⚠️ 仍坍缩或未超过常数基线（MAE={final_mae:.2f}, 常数={mae_c:.2f}, uniq={nuniq}）")
    np.savez(ROOT / "models" / "raw28" / "tune_curve.npz",
             ts=[r[0] for r in rows], maes=[r[1] for r in rows])


if __name__ == "__main__":
    main()
