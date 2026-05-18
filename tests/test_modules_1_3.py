"""模块 1（VideoLoader）与模块 3（FeatureExtractor）的轻量级测试。

不依赖 GPU，不使用 unittest / pytest，直接用 assert + print。
运行方式：python tests/test_modules_1_3.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.video_loader import VideoLoader
from src.feature_extractor import FeatureExtractor


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #

_TEST_VIDEO_PATH = "/tmp/test_video.mp4"
_VIDEO_WIDTH = 640
_VIDEO_HEIGHT = 480
_VIDEO_FPS = 30
_VIDEO_SECONDS = 10
_VIDEO_TOTAL_FRAMES = _VIDEO_FPS * _VIDEO_SECONDS  # 300


def _make_synthetic_video(path: str) -> None:
    """用 OpenCV 写一段 10 秒、640x480、30fps 的纯色合成视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, _VIDEO_FPS, (_VIDEO_WIDTH, _VIDEO_HEIGHT))
    assert writer.isOpened(), f"无法创建测试视频写入器: {path}"
    # 纯色帧（BGR）：稳定的蓝绿色
    frame = np.full((_VIDEO_HEIGHT, _VIDEO_WIDTH, 3), (200, 120, 40), dtype=np.uint8)
    for _ in range(_VIDEO_TOTAL_FRAMES):
        writer.write(frame)
    writer.release()
    assert os.path.isfile(path), f"测试视频未生成: {path}"


def _make_standing_joints(confidence: float = 0.9) -> dict[str, tuple[float, float, float]]:
    """构造一组模拟直立站姿的 fake joints 字典。

    图像坐标系 y 轴向下；躯干、双腿近似竖直，因此膝/髋角接近 180 度。
    包含 FeatureExtractor 所需的全部关节：
        mid_shoulder / left_hip / right_hip / left_knee / right_knee /
        left_ankle / right_ankle
    """
    return {
        "mid_shoulder": (100.0, 100.0, confidence),
        "left_hip": (90.0, 200.0, confidence),
        "right_hip": (110.0, 200.0, confidence),
        "left_knee": (90.0, 300.0, confidence),
        "right_knee": (110.0, 300.0, confidence),
        "left_ankle": (90.0, 400.0, confidence),
        "right_ankle": (110.0, 400.0, confidence),
    }


# --------------------------------------------------------------------------- #
# 模块 1：VideoLoader
# --------------------------------------------------------------------------- #

def test_video_loader() -> None:
    _make_synthetic_video(_TEST_VIDEO_PATH)
    try:
        loader = VideoLoader(
            _TEST_VIDEO_PATH,
            fps=_VIDEO_FPS,
            resize_width=_VIDEO_WIDTH,
            resize_height=_VIDEO_HEIGHT,
        )

        # extract_all_frames(): shape == (N, 480, 640, 3) 且 N > 0
        frames = loader.extract_all_frames()
        assert isinstance(frames, np.ndarray), "extract_all_frames 应返回 ndarray"
        assert frames.ndim == 4, f"期望 4 维数组，得到 {frames.ndim} 维"
        n = frames.shape[0]
        assert n > 0, f"期望 N > 0，得到 N={n}"
        assert frames.shape[1:] == (_VIDEO_HEIGHT, _VIDEO_WIDTH, 3), (
            f"期望帧形状 (480, 640, 3)，得到 {frames.shape[1:]}"
        )

        # extract_frames(): 生成器能正常 yield (frame_index, frame)
        gen = loader.extract_frames()
        first = next(gen)
        assert isinstance(first, tuple) and len(first) == 2, "生成器应 yield (idx, frame) 元组"
        idx, frame = first
        assert isinstance(idx, int), "frame_index 应为 int"
        assert isinstance(frame, np.ndarray) and frame.shape == (
            _VIDEO_HEIGHT,
            _VIDEO_WIDTH,
            3,
        ), f"yield 的帧形状应为 (480, 640, 3)，得到 {frame.shape}"
        # 生成器至少还能继续产出若干帧
        yielded = 1 + sum(1 for _ in gen)
        assert yielded > 0

        loader.release()

        # 文件不存在时抛出 FileNotFoundError
        raised = False
        try:
            VideoLoader("/tmp/__definitely_not_exist__.mp4")
        except FileNotFoundError:
            raised = True
        assert raised, "文件不存在时应抛出 FileNotFoundError"
    finally:
        # 测试完成后删除临时视频
        if os.path.isfile(_TEST_VIDEO_PATH):
            os.remove(_TEST_VIDEO_PATH)

    print("test_video_loader PASSED")


# --------------------------------------------------------------------------- #
# 模块 3：FeatureExtractor
# --------------------------------------------------------------------------- #

def test_feature_extractor() -> None:
    extractor = FeatureExtractor(fps=30, min_confidence=0.3)

    # compute_angle: 已知直角三点应返回 90 度
    angle = FeatureExtractor.compute_angle((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))
    assert abs(angle - 90.0) < 1e-6, f"期望 90 度，得到 {angle}"
    # 顺带验证一个 180 度的退化情形（共线反向）
    straight = FeatureExtractor.compute_angle((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert abs(straight - 180.0) < 1e-6, f"期望 180 度，得到 {straight}"

    # extract_frame_features: 合法关节输入返回含 6 个 key 的字典
    joints = _make_standing_joints(confidence=0.9)
    feats = extractor.extract_frame_features(joints)
    assert feats is not None, "合法关节输入不应返回 None"
    assert len(feats) == 6, f"期望 6 个特征 key，得到 {len(feats)}"
    expected_keys = {
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "knee_symmetry",
        "hip_symmetry",
    }
    assert set(feats.keys()) == expected_keys, f"特征 key 不匹配: {set(feats.keys())}"

    # extract_sequence_features: 30 帧相同关节 -> shape (30, 10)
    sequence: list[Optional[dict[str, tuple[float, float, float]]]] = [
        _make_standing_joints(confidence=0.9) for _ in range(30)
    ]
    matrix = extractor.extract_sequence_features(sequence)
    assert matrix.shape == (30, 10), f"期望 (30, 10)，得到 {matrix.shape}"

    # confidence 低于阈值时返回 None
    low_conf_joints = _make_standing_joints(confidence=0.1)  # 0.1 < 0.3
    assert extractor.extract_frame_features(low_conf_joints) is None, (
        "置信度低于阈值时应返回 None"
    )
    # 仅单个关节置信度不足也应返回 None
    partial = _make_standing_joints(confidence=0.9)
    partial["left_knee"] = (90.0, 300.0, 0.05)
    assert extractor.extract_frame_features(partial) is None, (
        "任一关节置信度不足时应返回 None"
    )

    print("test_feature_extractor PASSED")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_video_loader()
    test_feature_extractor()
    print("ALL TESTS PASSED")
