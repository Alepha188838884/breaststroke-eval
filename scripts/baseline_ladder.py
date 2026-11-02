"""Baseline ladder：在 Kinetics 真实数据 video-level train/test 划分上，用**同一
口径**（同一批 28 维 obs / 同一参考分 target / 同一档位网格）对比
constant / kNN / MLP / PPO 四个 baseline 的 MAE，量化 PPO 相对朴素基线的提升
（方差解释比）。

复用现有代码，不重算 target/obs：
    - train/val 周期划分：src.train_agent.build_kick_cycles_kinetics
    - score_scales：src.train_agent._resolve_score_scales
    - 28 维观测 / 参考分：src.kick_env.BreaststrokeKickEnv 的
      ``_cached_obs`` / ``_cached_target_score``（预计算好的，逐位一致）
    - PPO 全量评估：src.train_agent._full_eval

运行：
    python3 scripts/baseline_ladder.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kick_env import BreaststrokeKickEnv
from src.train_agent import _full_eval, _resolve_score_scales, build_kick_cycles_kinetics

logger = logging.getLogger(__name__)

_SEED = 42
_LOG_PATH = ROOT / "baseline_ladder.log"


def grid(x: np.ndarray) -> np.ndarray:
    """把连续预测值 round 到 PPO 的动作档位网格（0,5,...,100）。"""
    return np.clip(np.round(np.asarray(x) / 5.0) * 5.0, 0, 100)


def main(config_path: str) -> None:
    lines: list[str] = []

    def log(msg: str) -> None:
        logger.info(msg)
        lines.append(msg)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})

    train_cycles, val_cycles, aligner = build_kick_cycles_kinetics(
        features_dir=paths.get("features_dir", "data/features"),
        meta_path=paths.get("dataset_meta", "data/training/dataset_meta.json"),
        template_path=paths.get("template", "data/templates/champion_template.npy"),
    )
    score_scales = _resolve_score_scales(cfg, "kinetics")

    train_env = BreaststrokeKickEnv(
        train_cycles, aligner, seed=_SEED, score_scales=score_scales
    )
    val_env = BreaststrokeKickEnv(
        val_cycles, aligner, seed=_SEED, score_scales=score_scales
    )

    Xtr = np.asarray(train_env._cached_obs)
    ytr = np.asarray(train_env._cached_target_score)
    Xte = np.asarray(val_env._cached_obs)
    yte = np.asarray(val_env._cached_target_score)

    log("=" * 78)
    log(f"n_train_cycles={len(ytr)}  n_val_cycles={len(yte)}")

    # ---- constant baseline（train 均值 / 中位数） ----
    const_mean_pred = grid(np.full_like(yte, ytr.mean()))
    mae_const_mean = float(np.mean(np.abs(const_mean_pred - yte)))
    const_median_pred = grid(np.full_like(yte, np.median(ytr)))
    mae_const_median = float(np.mean(np.abs(const_median_pred - yte)))

    # ---- kNN(k=5)：train 统计量标准化，与 audit_circularity.py 同口径 ----
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    knn = KNeighborsRegressor(n_neighbors=5).fit(Ztr, ytr)
    mae_knn = float(np.mean(np.abs(grid(knn.predict(Zte)) - yte)))

    # ---- MLP（监督上界） ----
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256), max_iter=1000, random_state=_SEED
    ).fit(Ztr, ytr)
    mae_mlp = float(np.mean(np.abs(grid(mlp.predict(Zte)) - yte)))

    # ---- PPO ----
    model_path = paths.get("model_output", "models/ppo_kick_scorer.zip")
    ppo_model = PPO.load(model_path)
    ppo_stats = _full_eval(ppo_model, val_env)
    mae_ppo = ppo_stats["mae"]

    variance_explained = 1.0 - mae_ppo / mae_const_mean

    log("-" * 78)
    log("Baseline ladder（val 集 MAE，全部 round 到 5 分档位网格）：")
    log(f"  constant (train mean={ytr.mean():.2f})   : MAE = {mae_const_mean:.2f}")
    log(f"  constant (train median={np.median(ytr):.2f}) : MAE = {mae_const_median:.2f}")
    log(f"  kNN(k=5, z-norm)                        : MAE = {mae_knn:.2f}")
    log(f"  MLP(256,256, z-norm)                    : MAE = {mae_mlp:.2f}")
    log(f"  PPO ({model_path})                      : MAE = {mae_ppo:.2f}")
    log("-" * 78)
    log(f"PPO per_bin_mae: {ppo_stats['per_bin_mae']}")
    log(f"PPO n_unique_predictions = {ppo_stats['n_unique_predictions']}")
    log(f"PPO pred_std = {ppo_stats['pred_std']:.2f}")
    log(f"PPO max_abs_error = {ppo_stats['max_abs_error']:.2f}")
    log("-" * 78)
    log(
        f"variance_explained = 1 - PPO_MAE/constant_MAE(mean) "
        f"= 1 - {mae_ppo:.2f}/{mae_const_mean:.2f} = {variance_explained:.4f}"
    )
    log("=" * 78)

    _LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline ladder on Kinetics real data.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cli_args = _build_arg_parser().parse_args()
    main(cli_args.config)
