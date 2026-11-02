"""批量提取 data/raw/*.mp4 的姿态特征 → data/processed/{name}.npy。

走 VideoLoader → PoseEstimator → FeatureExtractor 的完整流水线，
优先使用 MPS，失败时自动回退到 CPU。已存在的 .npy 会被跳过。
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
from src.feature_extractor import FeatureExtractor


FPS = 30
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
MIN_CONFIDENCE = 0.3

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_frames(video_path: Path, fps: int = FPS) -> np.ndarray:
    loader = VideoLoader(
        str(video_path), fps=fps, resize_width=RESIZE_WIDTH, resize_height=RESIZE_HEIGHT
    )
    try:
        return loader.extract_all_frames()
    finally:
        loader.release()


def build_estimator_with_probe(probe_frame: np.ndarray) -> PoseEstimator:
    """优先 MPS：用 probe_frame 试推理，任何非 PoseEstimationError 视为 MPS 失败。"""
    try:
        estimator = PoseEstimator(device="mps", min_confidence=MIN_CONFIDENCE)
        try:
            estimator.estimate_single(probe_frame)
        except PoseEstimationError:
            # 探针帧没人体不算 MPS 失败
            pass
        print("使用 MPS 加速")
        return estimator
    except Exception as exc:
        print(f"MPS 不可用 ({type(exc).__name__}: {exc})，回退到 CPU")
        return PoseEstimator(device="cpu", min_confidence=MIN_CONFIDENCE)


def process_video(
    video_path: Path,
    out_path: Path,
    estimator: PoseEstimator,
    fps: int = FPS,
    frames: np.ndarray | None = None,
) -> None:
    """处理单个视频并保存 .npy。"""
    print(f"\n=== 处理: {video_path.name} ===")

    if frames is None:
        frames = load_frames(video_path, fps=fps)
    print(f"  抽帧完成: {frames.shape}")

    joint_sequence = estimator.estimate_batch(frames)

    extractor = FeatureExtractor(fps=fps, min_confidence=MIN_CONFIDENCE)
    feature_matrix = extractor.extract_sequence_features(joint_sequence)
    print(f"  特征 shape: {feature_matrix.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feature_matrix)
    print(f"  已保存: {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量把一个目录下的 .mp4 提取成姿态特征 .npy")
    p.add_argument("--input_dir", type=Path, default=RAW_DIR,
                   help="输入视频目录（默认 data/raw）。真实片子建议单独放 data/real")
    p.add_argument("--output_dir", type=Path, default=PROCESSED_DIR,
                   help="输出特征目录（默认 data/processed）")
    p.add_argument("--fps", type=int, default=FPS,
                   help="目标抽帧率（默认 30；为与 SwimXYZ 训练一致做域差对比，建议保持 30）")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"],
                   help="推理设备。本机 Mac 用 cpu（MPS 不支持 mmcv nms）；Colab GPU 用 cuda")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    in_dir, out_dir, fps = args.input_dir, args.output_dir, args.fps

    if not in_dir.exists():
        print(f"目录不存在: {in_dir}", file=sys.stderr)
        return 1

    videos = sorted(in_dir.glob("*.mp4"))
    if not videos:
        print(f"{in_dir}/ 下没有找到 .mp4 文件")
        return 0

    print(f"共发现 {len(videos)} 个视频，fps={fps} device={args.device}，将输出到 {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[Path, Path]] = []
    for v in videos:
        out_path = out_dir / f"{v.stem}.npy"
        if out_path.exists():
            print(f"跳过（已存在）: {out_path}")
        else:
            pending.append((v, out_path))

    if not pending:
        print("所有视频均已处理。")
        for out_path in sorted(out_dir.glob("*.npy")):
            arr = np.load(out_path)
            print(f"  {out_path}  shape={arr.shape}")
        return 0

    first_video, first_out = pending[0]
    first_frames = load_frames(first_video, fps=fps)
    if first_frames.shape[0] == 0:
        print(f"  首个视频抽帧为空: {first_video}", file=sys.stderr)
        return 1

    if args.device == "auto":
        # 用首个视频第 0 帧做 MPS 探针（本机上已知会回退到 CPU）
        estimator = build_estimator_with_probe(first_frames[0])
    else:
        print(f"使用 device={args.device}")
        estimator = PoseEstimator(device=args.device, min_confidence=MIN_CONFIDENCE)

    # 首个视频复用已加载的 frames，避免再读一次
    process_video(first_video, first_out, estimator, fps=fps, frames=first_frames)
    for video_path, out_path in pending[1:]:
        process_video(video_path, out_path, estimator, fps=fps)

    print("\n=== 全部完成 ===")
    for out_path in sorted(out_dir.glob("*.npy")):
        arr = np.load(out_path)
        print(f"  {out_path}  shape={arr.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
