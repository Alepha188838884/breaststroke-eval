"""脚本 2：对 Kinetics 蛙泳视频批量跑 HRNet 姿态估计。

流程（逐视频，处理完立即落盘，防中途崩溃丢进度）：
    1. 扫描 data/kinetics/*.mp4。
    2. VideoLoader 抽帧（fps=30, 640x480）→ PoseEstimator.estimate_batch 批量推理。
    3. 关节序列 list[Optional[dict]] 存为 data/kinetics/joints/{video_id}_joints.pkl。
    4. 统计并写入 quality_report.csv：total_frames / success_frames /
       success_rate / avg_lower_body_confidence / low_quality。
    5. 断点续传：已有 joints pkl 且已在报告中的视频跳过。
    6. 质量过滤：success_rate<0.5 或 avg_lower_body_confidence<0.4 标记 low_quality。

运行环境：
    需在装有 mmpose/mmdet/mmcv 的环境（本项目为 venv_pose）中运行。
    Mac 上 device 默认 cpu（mmcv NMS 在 MPS 上会崩，见 pose-inference-runtime）；
    Colab 上自动用 cuda。

用法：
    python scripts/02_batch_pose_estimation.py --data-dir data/kinetics --device auto
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from common import add_repo_root_to_path, auto_detect_device, ensure_dir, setup_logging

# 必须在 import PoseEstimator 之前：把仓库根加进 path，并 import mmcfg_bridge
# （它给 mmengine 的 config 加载打 patch，重定向 HRNet/RTMDet 的相对 config 路径，
#  并修复同进程内 mmdet+mmpose 的 default_scope 串台）。
add_repo_root_to_path()
import mmcfg_bridge  # noqa: E402,F401 — 副作用 import，务必先于 PoseEstimator

from src.pose_estimator import PoseEstimator, _LOWER_BODY_JOINTS  # noqa: E402
from src.video_loader import VideoLoader  # noqa: E402

logger = logging.getLogger(__name__)

# 质量过滤阈值
_MIN_SUCCESS_RATE: float = 0.5
_MIN_LOWER_CONF: float = 0.4

_REPORT_FIELDS: list[str] = [
    "video_id",
    "total_frames",
    "success_frames",
    "success_rate",
    "avg_lower_body_confidence",
    "low_quality",
]


def _avg_lower_body_conf(joints_seq: list[Optional[dict]]) -> float:
    """成功帧上、6 个下肢关节置信度的整体平均；无成功帧返回 0。"""
    confs: list[float] = []
    for joints in joints_seq:
        if joints is None:
            continue
        for name in _LOWER_BODY_JOINTS:
            if name in joints:
                confs.append(float(joints[name][2]))
    return float(sum(confs) / len(confs)) if confs else 0.0


def _report_row(video_id: str, joints_seq: list[Optional[dict]]) -> dict:
    """从关节序列计算一行质量报告（成功率 / 下肢置信度 / 低质量标记）。"""
    total = len(joints_seq)
    success = sum(1 for j in joints_seq if j is not None)
    success_rate = success / total if total else 0.0
    avg_conf = _avg_lower_body_conf(joints_seq)
    low_quality = success_rate < _MIN_SUCCESS_RATE or avg_conf < _MIN_LOWER_CONF
    return {
        "video_id": video_id,
        "total_frames": total,
        "success_frames": success,
        "success_rate": f"{success_rate:.4f}",
        "avg_lower_body_confidence": f"{avg_conf:.4f}",
        "low_quality": int(low_quality),
    }


def _load_existing_report(path: Path) -> dict[str, dict]:
    """读取已有 quality_report.csv 为 {video_id: row}，用于断点续传与合并。"""
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["video_id"]] = row
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取已有质量报告失败，将重建: %s", exc)
    return rows


def _write_report(path: Path, rows: dict[str, dict]) -> None:
    """把全部行写回 quality_report.csv（按 video_id 排序，覆盖写）。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REPORT_FIELDS)
        writer.writeheader()
        for vid in sorted(rows):
            writer.writerow(rows[vid])


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    joints_dir = ensure_dir(data_dir / "joints")
    # 分片时各进程写独立报告，避免并发写同一 CSV 竞态；脚本 3 会合并所有报告。
    report_name = (
        "quality_report.csv"
        if args.num_shards <= 1
        else f"quality_report_shard{args.shard_id}.csv"
    )
    report_path = data_dir / report_name
    setup_logging(data_dir / "pose_estimation.log")

    if not data_dir.is_dir():
        logger.error("视频目录不存在: %s（先运行脚本 1 下载）", data_dir)
        return

    videos = sorted(data_dir.glob("*.mp4"))
    if not videos:
        logger.error("在 %s 未找到 .mp4 视频", data_dir)
        return

    # --limit：本地小批量测试用（Mac CPU 上 HRNet 较慢；大批量建议 Colab GPU）。
    # 取排序后的前 N 个（其中已完成的仍会被断点续传逻辑快速跳过）。
    if args.limit is not None:
        videos = videos[: args.limit]
        logger.info("--limit=%d，仅处理前 %d 个视频", args.limit, len(videos))

    # 分片：多进程并行时，第 shard_id 个进程只取 videos[shard_id::num_shards]
    # （不相交子集），从而多进程可同时跑而互不重复。用法见下方 __main__ 注释。
    if args.num_shards > 1:
        videos = videos[args.shard_id :: args.num_shards]
        logger.info(
            "分片 %d/%d | 本进程处理 %d 个视频 | 报告=%s",
            args.shard_id, args.num_shards, len(videos), report_name,
        )

    device = auto_detect_device(args.device)
    # 限制每进程线程数：多进程时避免各自开满线程互相抢核（M2 4P+4E，单进程 4 线程
    # 仅用 ~1.5 核，故并行更划算）。缺省不动 torch 默认。
    if args.threads is not None:
        import torch
        torch.set_num_threads(args.threads)
        logger.info("torch 线程数设为 %d", args.threads)

    logger.info("设备=%s | 待处理视频 %d 个", device, len(videos))

    report = _load_existing_report(report_path)

    # 延迟到确认有活可干后再初始化模型（加载权重较慢）
    estimator: PoseEstimator | None = None
    n_done = n_skip = n_fail = 0

    for i, video_path in enumerate(tqdm(videos, desc="批量姿态估计"), start=1):
        video_id = video_path.stem
        pkl_path = joints_dir / f"{video_id}_joints.pkl"

        # 断点续传：已有 pkl 即跳过昂贵推理。若当前（分片）报告缺该行——例如换了
        # 分片报告文件、或视频被别的进程/上一轮跑过——就从已有 pkl 重算统计补上，
        # 保证本报告对自己负责的子集是完整的。
        if pkl_path.is_file():
            n_skip += 1
            if video_id not in report:
                try:
                    with open(pkl_path, "rb") as f:
                        report[video_id] = _report_row(video_id, pickle.load(f))
                    _write_report(report_path, report)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("从已有 pkl 重算统计失败 %s: %s", video_id, exc)
            continue

        logger.info("[%d/%d] 处理 %s", i, len(videos), video_id)
        try:
            loader = VideoLoader(
                str(video_path),
                fps=args.fps,
                resize_width=args.width,
                resize_height=args.height,
            )
            frames = loader.extract_all_frames()
            loader.release()
        except Exception as exc:  # noqa: BLE001 — 单个坏视频不中断整体
            logger.error("抽帧失败，跳过 %s: %s", video_id, exc)
            n_fail += 1
            continue

        if frames.shape[0] == 0:
            logger.warning("视频无帧，跳过 %s", video_id)
            n_fail += 1
            continue

        if estimator is None:
            estimator = PoseEstimator(
                model_name=args.model, device=device, min_confidence=args.min_conf
            )

        try:
            joints_seq = estimator.estimate_batch(frames)
        except Exception as exc:  # noqa: BLE001
            logger.error("姿态估计失败，跳过 %s: %s", video_id, exc)
            n_fail += 1
            continue

        # 逐视频立即落盘：先存关节 pkl，再更新报告，防崩溃丢进度
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(joints_seq, f)
        except Exception as exc:  # noqa: BLE001
            logger.error("保存 joints pkl 失败 %s: %s", video_id, exc)
            n_fail += 1
            continue

        row = _report_row(video_id, joints_seq)
        report[video_id] = row
        _write_report(report_path, report)
        n_done += 1
        logger.info(
            "  → 成功帧 %s/%s (%.1f%%) | 下肢均置信度 %s | %s",
            row["success_frames"],
            row["total_frames"],
            100 * row["success_frames"] / max(row["total_frames"], 1),
            row["avg_lower_body_confidence"],
            "LOW_QUALITY" if row["low_quality"] else "ok",
        )

    n_low = sum(1 for r in report.values() if int(r.get("low_quality", 0)))
    logger.info(
        "批量姿态估计完成 | 新处理 %d | 跳过 %d | 失败 %d | 报告共 %d 视频（低质量 %d）",
        n_done,
        n_skip,
        n_fail,
        len(report),
        n_low,
    )
    logger.info("质量报告: %s", report_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="批量 HRNet 姿态估计")
    p.add_argument("--data-dir", type=Path, default=Path("data/kinetics"),
                   help="含 .mp4 视频的目录")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="推理设备；auto=cuda>cpu（Mac 不自动用 mps，会崩）")
    p.add_argument("--model", type=str, default="hrnet_w48", help="姿态模型名")
    p.add_argument("--fps", type=int, default=30, help="抽帧目标 fps")
    p.add_argument("--width", type=int, default=640, help="resize 宽")
    p.add_argument("--height", type=int, default=480, help="resize 高")
    p.add_argument("--min-conf", type=float, default=0.3, help="关节点最小置信度")
    p.add_argument("--limit", type=int, default=None,
                   help="仅处理前 N 个视频（本地小批量测试）")
    p.add_argument("--num-shards", type=int, default=1,
                   help="分片总数：多进程并行时设为进程数（每进程处理不相交子集）")
    p.add_argument("--shard-id", type=int, default=0,
                   help="本进程分片编号，0..num_shards-1")
    p.add_argument("--threads", type=int, default=None,
                   help="每进程 torch 线程数；多进程并行时建议设 2 避免抢核")
    return p


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
