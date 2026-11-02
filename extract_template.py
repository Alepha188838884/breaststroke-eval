"""为冠军视频生成模板特征矩阵 → data/templates/champion_template.npy。

用法：
    python extract_template.py                       # 默认 data/raw/champion.mp4
    python extract_template.py path/to/champion.mp4  # 指定路径
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

import mmcfg_bridge  # noqa: F401 — patch Config.fromfile 路径重定向

from src.video_loader import VideoLoader
from src.pose_estimator import PoseEstimator, PoseEstimationError
from src.feature_extractor import FeatureExtractor, smooth_feature_matrix


FPS = 30
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
MIN_CONFIDENCE = 0.3

DEFAULT_VIDEO = Path("data/raw/champion.mp4")
DEFAULT_OUTPUT = Path("data/templates/champion_template.npy")


def load_frames(video_path: Path) -> np.ndarray:
    loader = VideoLoader(
        str(video_path), fps=FPS, resize_width=RESIZE_WIDTH, resize_height=RESIZE_HEIGHT
    )
    try:
        return loader.extract_all_frames()
    finally:
        loader.release()


def build_estimator_with_probe(probe_frame: np.ndarray) -> PoseEstimator:
    try:
        estimator = PoseEstimator(device="mps", min_confidence=MIN_CONFIDENCE)
        try:
            estimator.estimate_single(probe_frame)
        except PoseEstimationError:
            pass
        print("使用 MPS 加速")
        return estimator
    except Exception as exc:
        print(f"MPS 不可用 ({type(exc).__name__}: {exc})，回退到 CPU")
        return PoseEstimator(device="cpu", min_confidence=MIN_CONFIDENCE)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="生成冠军模板特征矩阵")
    parser.add_argument(
        "video", nargs="?", default=str(DEFAULT_VIDEO), help="冠军视频路径"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUTPUT), help="输出 .npy 路径"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)

    if not video_path.is_file():
        print(f"视频不存在: {video_path}", file=sys.stderr)
        return 1

    print(f"=== 冠军模板提取: {video_path.name} ===")
    frames = load_frames(video_path)
    print(f"抽帧完成: {frames.shape}")
    if frames.shape[0] == 0:
        print("抽帧为空", file=sys.stderr)
        return 1

    estimator = build_estimator_with_probe(frames[0])

    joint_sequence = estimator.estimate_batch(frames)
    extractor = FeatureExtractor(fps=FPS, min_confidence=MIN_CONFIDENCE)
    feature_matrix = extractor.extract_sequence_features(joint_sequence)
    print(f"特征 shape: {feature_matrix.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 模板必须是“单个蹬腿周期”：下游 DTWAligner.align / compute_phase_deviations /
    # kick_env 的 timing 因素都假设模板≈一个周期。整段特征（多周期）会让 timing 因素
    # 失效、并使参考分数偏移。先用整段特征引导一个 aligner 切周期，再保存中位长度的
    # 代表性单周期作为模板。
    from src.dtw_aligner import DTWAligner

    np.save(out_path, feature_matrix)  # 临时落盘以引导 aligner 分割
    boot = DTWAligner(str(out_path))
    cycles = boot.segment_kick_cycles(feature_matrix)
    if not cycles:
        print(
            "警告：未能从冠军视频分割出蹬腿周期，已保存整段特征（多周期）；"
            "请检查视频是否包含完整、规律的蹬腿动作。",
            file=sys.stderr,
        )
        print(f"已保存模板(整段，多周期): {out_path}")
        return 0

    lengths = [c.shape[0] for c in cycles]
    rep = cycles[int(np.argsort(lengths)[len(lengths) // 2])]
    # HRNet 逐帧抖动会让模板角速度虚高，存盘前平滑一次角度（见 feature_extractor.
    # smooth_feature_matrix docstring），避免把估计噪声当成技术基准。
    rep = smooth_feature_matrix(rep, window=5, fps=FPS)
    np.save(out_path, rep)
    print(
        f"从 {len(cycles)} 个蹬腿周期中选中位长度单周期 ({rep.shape[0]} 帧) 作为模板"
    )
    print(f"已保存模板: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
