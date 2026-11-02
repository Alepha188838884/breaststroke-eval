"""脚本 4：汇总所有蹬腿周期，构建 RL 训练数据集（观测向量）。

流程：
    1. 扫描 data/features/*_cycles.pkl。
    2. 对每个周期计算与 BreaststrokeKickEnv 一致的 **28 维**观测摘要向量
       （原始运动学统计量；不含 DTW 派生量，故无需模板）。
    3. **序列级** 80/20 划分：按 video_id 分组洗牌后切分，防止同一视频的周期
       同时落入 train 与 test（泄露）。
    4. 保存 train/test_observations.npy 与 dataset_meta.json。

关于观测维度：任务书提到"35 维"，但项目 RL 环境的观测空间实为 **28 维**
（kick_env._OBS_DIM）：角度 mean/std/min/max×4=16 + 对称 mean/std×2=4 +
角速度 mean/std×4=8。本脚本从 kick_env 导入尺度常量与边界，严格对齐环境，
使离线构建的观测能直接喂给用同一环境训练的策略。

用法：
    python scripts/04_build_training_dataset.py --features-dir data/features \
        --out-dir data/training --train-ratio 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from common import add_repo_root_to_path, ensure_dir, setup_logging

add_repo_root_to_path()
# 从 kick_env 导入观测布局常量，保证与环境观测空间单一真源一致（避免维度/尺度漂移）。
from src.kick_env import (  # noqa: E402
    _ANGLE_COLS,
    _ANGLE_SCALE,
    _OBS_DIM,
    _SYMMETRY_COLS,
    _VELOCITY_COLS,
    _VELOCITY_SCALE,
    _build_obs_bounds,
)

logger = logging.getLogger(__name__)

_OBS_LOW, _OBS_HIGH = _build_obs_bounds()


def compute_cycle_observation(cycle: np.ndarray) -> np.ndarray:
    """把 (cycle_len, 10) 周期特征转为 28 维观测向量。

    完全复刻 BreaststrokeKickEnv._compute_observation 中**原始运动学**部分
    （不含 DTW 阶段偏差/warp——那些是参考分公式输入，不进观测）：
        angle mean/std/min/max (16) + symmetry mean/std (4) +
        velocity mean/std (8)，各按 _ANGLE_SCALE / _VELOCITY_SCALE 归一后裁剪。
    """
    cycle = np.asarray(cycle, dtype=np.float64)
    angles = cycle[:, _ANGLE_COLS]
    symmetry = cycle[:, _SYMMETRY_COLS]
    velocity = cycle[:, _VELOCITY_COLS]
    obs = np.concatenate(
        [
            angles.mean(axis=0) / _ANGLE_SCALE,
            angles.std(axis=0) / _ANGLE_SCALE,
            angles.min(axis=0) / _ANGLE_SCALE,
            angles.max(axis=0) / _ANGLE_SCALE,
            symmetry.mean(axis=0) / _ANGLE_SCALE,
            symmetry.std(axis=0) / _ANGLE_SCALE,
            velocity.mean(axis=0) / _VELOCITY_SCALE,
            velocity.std(axis=0) / _VELOCITY_SCALE,
        ],
        axis=0,
    ).astype(np.float32)
    return np.clip(obs, _OBS_LOW, _OBS_HIGH)


def _load_cycles(pkl_path: Path) -> list[np.ndarray]:
    """加载一个 *_cycles.pkl，返回形状合法 (n,10) 的周期列表。"""
    with open(pkl_path, "rb") as f:
        cycles = pickle.load(f)
    valid: list[np.ndarray] = []
    for c in cycles:
        arr = np.asarray(c)
        if arr.ndim == 2 and arr.shape[1] == 10 and arr.shape[0] >= 2:
            valid.append(arr)
    return valid


def main(args: argparse.Namespace) -> None:
    features_dir = Path(args.features_dir)
    out_dir = ensure_dir(args.out_dir)
    setup_logging(out_dir / "build_dataset.log")

    if not features_dir.is_dir():
        logger.error("特征目录不存在: %s（先运行脚本 3）", features_dir)
        return

    cycle_files = sorted(features_dir.glob("*_cycles.pkl"))
    if not cycle_files:
        logger.error("在 %s 未找到 *_cycles.pkl", features_dir)
        return

    # 逐视频构建观测，按 video_id 分组（供序列级划分）
    per_video_obs: dict[str, np.ndarray] = {}
    n_source = len(cycle_files)
    for pkl_path in cycle_files:
        video_id = pkl_path.stem.removesuffix("_cycles")
        try:
            cycles = _load_cycles(pkl_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取周期失败，跳过 %s: %s", video_id, exc)
            continue
        if not cycles:
            logger.info("视频 %s 无有效周期，跳过", video_id)
            continue
        obs = np.stack([compute_cycle_observation(c) for c in cycles], axis=0)
        per_video_obs[video_id] = obs

    if not per_video_obs:
        logger.error("所有视频均无有效周期，无法构建数据集")
        return

    video_ids = sorted(per_video_obs)
    total_cycles = sum(len(v) for v in per_video_obs.values())
    logger.info(
        "来源视频 %d | 有周期视频 %d | 总周期 %d | obs_dim=%d（对齐 kick_env 观测空间）",
        n_source, len(video_ids), total_cycles, _OBS_DIM,
    )

    # 序列级 80/20 划分：洗牌 video_id 后按视频数切分，再各自 flatten 成周期
    rng = np.random.default_rng(args.seed)
    order = np.array(video_ids)
    rng.shuffle(order)

    n_vid = len(order)
    if n_vid >= 2:
        split = int(round(args.train_ratio * n_vid))
        split = min(max(split, 1), n_vid - 1)  # 保证 train/test 各至少 1 个视频
    else:
        split = n_vid  # 只有 1 个视频：全部归 train，test 为空
        logger.warning("只有 1 个视频，无法划分 test，全部归 train")

    train_ids = list(order[:split])
    test_ids = list(order[split:])

    def _stack(ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, _OBS_DIM), dtype=np.float32)
        return np.concatenate([per_video_obs[v] for v in ids], axis=0)

    train_obs = _stack(train_ids)
    test_obs = _stack(test_ids)

    # 保存
    train_path = out_dir / "train_observations.npy"
    test_path = out_dir / "test_observations.npy"
    meta_path = out_dir / "dataset_meta.json"
    try:
        np.save(train_path, train_obs)
        np.save(test_path, test_obs)
    except Exception as exc:  # noqa: BLE001
        logger.error("保存观测数组失败: %s", exc)
        return

    meta = {
        "obs_dim": int(_OBS_DIM),
        "train_ratio": float(args.train_ratio),
        "seed": int(args.seed),
        "n_source_videos": n_source,
        "n_videos_with_cycles": len(video_ids),
        "total_cycles": int(total_cycles),
        "n_train_videos": len(train_ids),
        "n_test_videos": len(test_ids),
        "n_train_cycles": int(train_obs.shape[0]),
        "n_test_cycles": int(test_obs.shape[0]),
        "train_video_ids": sorted(train_ids),
        "test_video_ids": sorted(test_ids),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("保存 dataset_meta.json 失败: %s", exc)

    # 汇总
    logger.info("=" * 60)
    logger.info(
        "数据集构建完成 | train: %d 视频 / %d 周期 | test: %d 视频 / %d 周期",
        len(train_ids), train_obs.shape[0], len(test_ids), test_obs.shape[0],
    )
    logger.info("train_observations.npy shape=%s", tuple(train_obs.shape))
    logger.info("test_observations.npy  shape=%s", tuple(test_obs.shape))
    logger.info("元信息: %s", meta_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="构建 RL 训练数据集（28 维观测）")
    p.add_argument("--features-dir", type=Path, default=Path("data/features"),
                   help="含 *_cycles.pkl 的目录")
    p.add_argument("--out-dir", type=Path, default=Path("data/training"),
                   help="数据集输出目录")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="序列级 train 占比（按视频数）")
    p.add_argument("--seed", type=int, default=42, help="划分随机种子")
    return p


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
