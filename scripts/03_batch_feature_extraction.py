"""脚本 3：对合格视频批量提取特征并分割蹬腿周期。

流程：
    1. 读 quality_report.csv，只处理 low_quality==0 的视频。
    2. 逐视频：加载 joints pkl → FeatureExtractor 提取 (N,10) 特征矩阵
       → DTWAligner.segment_kick_cycles 分割周期。
    3. 保存 data/features/{video_id}_features.npy 与 {video_id}_cycles.pkl。
    4. 写 feature_report.csv：total_valid_frames / num_cycles /
       avg_cycle_length / angle_range_left_knee / angle_range_right_knee。
    5. 汇总：合格视频数、总有效帧、总周期数、各角度全局 min/max/mean。

说明：
    - 只做周期分割，不做模板 DTW 对齐；但 segment_kick_cycles 是 DTWAligner 的
      方法，实例化需要一个模板路径（该方法本身不使用模板内容）。
    - 某些视频可能分割不出任何周期（动作不明显 / 太短），记录并继续。

用法：
    python scripts/03_batch_feature_extraction.py --data-dir data/kinetics \
        --features-dir data/features
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np

from common import add_repo_root_to_path, ensure_dir, setup_logging

add_repo_root_to_path()
from src.dtw_aligner import DTWAligner  # noqa: E402
from src.feature_extractor import FEATURE_NAMES, FeatureExtractor  # noqa: E402

logger = logging.getLogger(__name__)

# 4 个角度列索引（FEATURE_NAMES 前 4 列）
_ANGLE_COL_NAMES: list[str] = FEATURE_NAMES[:4]
_LEFT_KNEE_COL: int = 0
_RIGHT_KNEE_COL: int = 1

_REPORT_FIELDS: list[str] = [
    "video_id",
    "total_valid_frames",
    "num_cycles",
    "avg_cycle_length",
    "angle_range_left_knee",
    "angle_range_right_knee",
]


def _load_qualified_ids(data_dir: Path) -> list[str]:
    """合并读取所有 quality_report*.csv（主报告 + 分片报告），返回 low_quality==0
    的 video_id（按 video_id 去重排序）。

    脚本 2 分片并行时各写 quality_report_shard{i}.csv，这里统一合并。
    """
    report_paths = sorted(data_dir.glob("quality_report*.csv"))
    if not report_paths:
        return []
    merged: dict[str, int] = {}  # video_id -> low_quality
    for path in report_paths:
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        merged[row["video_id"]] = int(row.get("low_quality", "0"))
                    except (ValueError, KeyError):
                        logger.warning("质量报告异常行，跳过 (%s): %s", path.name, row)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取报告失败 %s: %s", path, exc)
    logger.info("合并 %d 份质量报告，共 %d 视频", len(report_paths), len(merged))
    return sorted(vid for vid, low in merged.items() if low == 0)


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    joints_dir = data_dir / "joints"
    features_dir = ensure_dir(args.features_dir)
    feature_report_path = features_dir / "feature_report.csv"
    setup_logging(features_dir / "feature_extraction.log")

    if not list(data_dir.glob("quality_report*.csv")):
        logger.error("质量报告不存在于 %s（先运行脚本 2）", data_dir)
        return
    if not args.template.is_file():
        logger.error(
            "模板文件不存在: %s（segment_kick_cycles 需实例化 DTWAligner，"
            "可用 extract_template.py 生成或 --template 指定）",
            args.template,
        )
        return

    qualified = _load_qualified_ids(data_dir)
    logger.info("合格视频（非 low_quality）%d 个", len(qualified))
    if not qualified:
        logger.warning("无合格视频可处理")
        return

    extractor = FeatureExtractor(fps=args.fps)
    aligner = DTWAligner(str(args.template))

    report_rows: list[dict] = []
    total_valid_frames = 0
    total_cycles = 0
    n_processed = 0
    n_no_cycle = 0
    # 全局角度统计（4 列）
    angle_min = np.full(4, np.inf)
    angle_max = np.full(4, -np.inf)
    angle_sum = np.zeros(4)
    angle_count = 0

    for video_id in qualified:
        pkl_path = joints_dir / f"{video_id}_joints.pkl"
        if not pkl_path.is_file():
            logger.warning("缺 joints pkl，跳过 %s", video_id)
            continue
        try:
            with open(pkl_path, "rb") as f:
                joints_seq = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取 joints pkl 失败，跳过 %s: %s", video_id, exc)
            continue

        try:
            features = extractor.extract_sequence_features(joints_seq)
        except Exception as exc:  # noqa: BLE001
            logger.error("特征提取失败，跳过 %s: %s", video_id, exc)
            continue

        n_valid = int(features.shape[0])
        if n_valid == 0:
            logger.warning("无有效帧，跳过 %s", video_id)
            continue

        try:
            cycles = aligner.segment_kick_cycles(features)
        except Exception as exc:  # noqa: BLE001
            logger.error("周期分割失败，跳过 %s: %s", video_id, exc)
            continue

        # 保存特征与周期
        try:
            np.save(features_dir / f"{video_id}_features.npy", features)
            with open(features_dir / f"{video_id}_cycles.pkl", "wb") as f:
                pickle.dump(cycles, f)
        except Exception as exc:  # noqa: BLE001
            logger.error("保存特征/周期失败，跳过 %s: %s", video_id, exc)
            continue

        # 逐视频报告行
        cycle_lengths = [int(c.shape[0]) for c in cycles]
        avg_len = float(np.mean(cycle_lengths)) if cycle_lengths else 0.0
        # np.ptp（而非 ndarray.ptp()）以兼容 numpy 2.x（后者已移除该方法）
        lk_range = float(np.ptp(features[:, _LEFT_KNEE_COL]))
        rk_range = float(np.ptp(features[:, _RIGHT_KNEE_COL]))
        report_rows.append(
            {
                "video_id": video_id,
                "total_valid_frames": n_valid,
                "num_cycles": len(cycles),
                "avg_cycle_length": f"{avg_len:.2f}",
                "angle_range_left_knee": f"{lk_range:.2f}",
                "angle_range_right_knee": f"{rk_range:.2f}",
            }
        )

        # 全局累计
        total_valid_frames += n_valid
        total_cycles += len(cycles)
        n_processed += 1
        if not cycles:
            n_no_cycle += 1
            logger.info("  %s | %d 有效帧 | 未分割出周期", video_id, n_valid)
        else:
            logger.info(
                "  %s | %d 有效帧 | %d 周期 | 平均周期长 %.1f 帧",
                video_id, n_valid, len(cycles), avg_len,
            )
        ang = features[:, :4]
        angle_min = np.minimum(angle_min, ang.min(axis=0))
        angle_max = np.maximum(angle_max, ang.max(axis=0))
        angle_sum += ang.sum(axis=0)
        angle_count += n_valid

    # 写 feature_report.csv
    try:
        with open(feature_report_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_REPORT_FIELDS)
            writer.writeheader()
            for row in sorted(report_rows, key=lambda r: r["video_id"]):
                writer.writerow(row)
    except Exception as exc:  # noqa: BLE001
        logger.error("写 feature_report.csv 失败: %s", exc)

    # 汇总统计
    logger.info("=" * 60)
    logger.info(
        "特征提取汇总 | 处理视频 %d | 总有效帧 %d | 总周期 %d | 无周期视频 %d",
        n_processed, total_valid_frames, total_cycles, n_no_cycle,
    )
    if angle_count > 0:
        angle_mean = angle_sum / angle_count
        for i, name in enumerate(_ANGLE_COL_NAMES):
            logger.info(
                "  角度 %-16s | min=%.1f max=%.1f mean=%.1f",
                name, angle_min[i], angle_max[i], angle_mean[i],
            )
    logger.info("特征报告: %s", feature_report_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="批量特征提取与周期分割")
    p.add_argument("--data-dir", type=Path, default=Path("data/kinetics"),
                   help="含 quality_report.csv 与 joints/ 的目录")
    p.add_argument("--features-dir", type=Path, default=Path("data/features"),
                   help="特征与周期输出目录")
    p.add_argument("--template", type=Path,
                   default=Path("data/templates/champion_template.npy"),
                   help="DTWAligner 实例化所需模板（周期分割不使用其内容）")
    p.add_argument("--fps", type=int, default=30, help="角速度计算帧率")
    return p


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
