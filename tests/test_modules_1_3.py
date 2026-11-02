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
from src.feature_extractor import FeatureExtractor, smooth_feature_matrix


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
# 模块 3：R-3 退化角度返回 NaN，并使该帧整体被丢弃
# --------------------------------------------------------------------------- #

def test_compute_angle_degenerate_returns_nan() -> None:
    extractor = FeatureExtractor(fps=30, min_confidence=0.3)

    # p1 与顶点 p2 重合 -> v1 范数退化为 0，应返回 NaN 而非 0.0
    degenerate = FeatureExtractor.compute_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert isinstance(degenerate, float) and degenerate != degenerate, (
        f"重合点应返回 NaN，得到 {degenerate}"
    )

    # 构造 left_knee 与 left_hip 重合的关节帧（角度退化），置信度均达标
    joints = _make_standing_joints(confidence=0.9)
    joints["left_hip"] = joints["left_knee"]  # left_knee_angle 的 p1 == p2
    assert extractor.extract_frame_features(joints) is None, (
        "任一角度退化为 NaN 时，整帧应按无效帧丢弃（返回 None）"
    )

    print("test_compute_angle_degenerate_returns_nan PASSED")


# --------------------------------------------------------------------------- #
# 模块 3：R-1 角速度使用真实帧间隔 dt
# --------------------------------------------------------------------------- #

def test_angular_velocity_uses_real_frame_gap() -> None:
    fps = 30
    extractor = FeatureExtractor(fps=fps, min_confidence=0.3)

    # 构造一组关节，left_knee_angle 随 x 偏移线性变化，便于手算角速度
    def _joints_at(knee_x: float) -> dict[str, tuple[float, float, float]]:
        j = _make_standing_joints(confidence=0.9)
        j["left_knee"] = (knee_x, 300.0, 0.9)
        return j

    # 连续无丢帧序列：帧 0..3，dt 恒为 1/fps
    continuous = [_joints_at(90.0 + i) for i in range(4)]
    matrix_continuous = extractor.extract_sequence_features(continuous)
    assert matrix_continuous.shape == (4, 10)
    left_knee_col = 0
    velocity_col = 6  # left_knee_angular_velocity
    # 中间帧（索引1）中心差分：(angle[2]-angle[0]) / (2*dt)
    expected_mid = (
        matrix_continuous[2, left_knee_col] - matrix_continuous[0, left_knee_col]
    ) / (2.0 * extractor.dt)
    assert abs(matrix_continuous[1, velocity_col] - expected_mid) < 1e-9, (
        "无丢帧序列的角速度应与旧实现（dt=1/fps）语义一致"
    )

    # 带缺口序列：原始帧索引 0,1,2,5,6,7（3,4 为 None，模拟丢帧）
    gapped_sequence: list[Optional[dict[str, tuple[float, float, float]]]] = [
        _joints_at(90.0 + 0),
        _joints_at(90.0 + 1),
        _joints_at(90.0 + 2),
        None,
        None,
        _joints_at(90.0 + 5),
        _joints_at(90.0 + 6),
        _joints_at(90.0 + 7),
    ]
    matrix_gapped = extractor.extract_sequence_features(gapped_sequence)
    assert matrix_gapped.shape == (6, 10), f"应保留 6 个有效帧，得到 {matrix_gapped.shape}"

    # 跨缺口处（有效帧索引 1 与 3，对应原始序列索引 1 与 5，真实间隔 4 帧）
    # 中心差分分母应为 (5-1)*dt = 4*dt，是连续情形 2*dt 的 2 倍 -> 角速度约为
    # 连续情形跨缺口处"错误分母"结果的 1/2（本例线性变化，用比值直接验证分母倍数）
    cross_gap_velocity = matrix_gapped[2, velocity_col]  # 有效帧索引2=原始索引2，其两侧为1和5
    # 有效帧索引2的中心差分：angles取自有效帧索引1,3 -> 原始索引1,5，真实间隔=5-1=4
    expected_cross_gap = (
        matrix_gapped[3, left_knee_col] - matrix_gapped[1, left_knee_col]
    ) / (4.0 * extractor.dt)
    assert abs(cross_gap_velocity - expected_cross_gap) < 1e-9, (
        "跨缺口处角速度分母应使用真实索引差 (4) 而非恒定的 2"
    )
    # 若误用恒定分母 2*dt（旧 bug），值会是正确值的 2 倍；显式断言二者不同
    wrong_denominator_velocity = (
        matrix_gapped[3, left_knee_col] - matrix_gapped[1, left_knee_col]
    ) / (2.0 * extractor.dt)
    assert abs(cross_gap_velocity - wrong_denominator_velocity / 2.0) < 1e-9, (
        "跨缺口角速度应约为错误分母(旧bug)结果的 1/2"
    )

    print("test_angular_velocity_uses_real_frame_gap PASSED")


# --------------------------------------------------------------------------- #
# 模块 1：M-5 letterbox 等比缩放 + 补黑边
# --------------------------------------------------------------------------- #

def test_letterbox_preserves_aspect_ratio() -> None:
    _make_synthetic_video(_TEST_VIDEO_PATH)
    try:
        # 目标尺寸 640x480（4:3），letterbox 一个 640x360（16:9）纯色帧
        loader = VideoLoader(
            _TEST_VIDEO_PATH,
            fps=_VIDEO_FPS,
            resize_width=640,
            resize_height=480,
        )
        src = np.full((360, 640, 3), 255, dtype=np.uint8)
        out = loader._letterbox(src)
        assert out.shape == (480, 640, 3), f"期望 (480, 640, 3)，得到 {out.shape}"

        # 等比缩放系数 = min(640/640, 480/360) = 1.0 -> 缩放后仍是 640x360，
        # 上下各补 (480-360)/2 = 60 像素黑边
        assert np.all(out[:60] == 0), "上边应为黑边"
        assert np.all(out[-60:] == 0), "下边应为黑边"
        assert np.all(out[60:420] == 255), "中间区域应保留原色，未被拉伸"

        loader.release()
    finally:
        if os.path.isfile(_TEST_VIDEO_PATH):
            os.remove(_TEST_VIDEO_PATH)

    print("test_letterbox_preserves_aspect_ratio PASSED")


# --------------------------------------------------------------------------- #
# 模块 3：姿态崩坏帧修复
# --------------------------------------------------------------------------- #

def test_pose_failure_repair() -> None:
    # 平滑大幅摆动（模拟真实蹬腿膝角 50–170° 摆动）
    t = np.linspace(0.0, 4.0 * np.pi, 60)
    clean = np.stack([110.0 - 60.0 * np.cos(t + k * 0.3) for k in range(4)], axis=1)

    # 注入两个单帧尖刺，取值刻意远离各自局部中值以构成明确崩坏：
    # 帧30 左膝 clean≈50°→顶到 175°；帧45 左髋 clean≈160°→崩到 30°
    corrupted = clean.copy()
    corrupted[30, 0] = 175.0
    corrupted[45, 2] = 30.0

    repaired, n_bad = FeatureExtractor._repair_pose_failures(corrupted)
    assert n_bad >= 2, f"应检出 ≥2 个崩坏帧，得到 {n_bad}"
    assert abs(repaired[30, 0] - clean[30, 0]) < 15.0, (
        f"帧30左膝应被插值修复回≈{clean[30,0]:.0f}，得到 {repaired[30,0]:.0f}"
    )
    assert abs(repaired[45, 2] - clean[45, 2]) < 15.0, (
        f"帧45左髋应被插值修复回≈{clean[45,2]:.0f}，得到 {repaired[45,2]:.0f}"
    )

    # 干净序列：零修复且原值不变（平滑大幅摆动不被误伤）
    r2, n2 = FeatureExtractor._repair_pose_failures(clean)
    assert n2 == 0, f"干净序列应零修复，得到 {n2}"
    assert np.allclose(r2, clean), "干净序列不应被改动"

    print("test_pose_failure_repair PASSED")


# --------------------------------------------------------------------------- #
# 模块 3：R-4 模板平滑 smooth_feature_matrix
# --------------------------------------------------------------------------- #

def test_smooth_feature_matrix() -> None:
    fps = 30
    n = 200
    t = np.arange(n) / fps
    rng = np.random.default_rng(0)

    # 平滑正弦角度 + 高频噪声，模拟 HRNet 逐帧抖动
    base = 90.0 + 40.0 * np.sin(2 * np.pi * 0.5 * t)
    angles = np.tile(base[:, None], (1, 4)) + rng.normal(scale=8.0, size=(n, 4))

    features = np.zeros((n, 10), dtype=np.float64)
    features[:, :4] = angles
    features[:, 4] = np.abs(angles[:, 0] - angles[:, 1])
    features[:, 5] = np.abs(angles[:, 2] - angles[:, 3])
    # 6-9 列（角速度）留作占位值，smooth_feature_matrix 应据平滑后角度重算，
    # 不依赖这里的占位内容
    features[:, 6:10] = 0.0

    smoothed = smooth_feature_matrix(features, window=5, fps=fps)

    # shape / dtype 不变
    assert smoothed.shape == features.shape, f"shape 应不变，得到 {smoothed.shape}"
    assert smoothed.dtype == features.dtype, f"dtype 应不变，得到 {smoothed.dtype}"

    # 对称性列 = 平滑后角度差的绝对值
    assert np.allclose(smoothed[:, 4], np.abs(smoothed[:, 0] - smoothed[:, 1])), (
        "knee_symmetry 应等于平滑后左右膝角之差的绝对值"
    )
    assert np.allclose(smoothed[:, 5], np.abs(smoothed[:, 2] - smoothed[:, 3])), (
        "hip_symmetry 应等于平滑后左右髋角之差的绝对值"
    )

    # 平滑后角速度 pooled std 应显著低于未平滑角度算出的角速度 pooled std
    dt = 1.0 / fps
    raw_velocity = np.zeros_like(angles)
    raw_velocity[1:-1] = (angles[2:] - angles[:-2]) / (2.0 * dt)
    raw_velocity[0] = (angles[1] - angles[0]) / dt
    raw_velocity[-1] = (angles[-1] - angles[-2]) / dt
    raw_pooled_std = raw_velocity.std()
    smoothed_pooled_std = smoothed[:, 6:10].std()
    assert smoothed_pooled_std < 0.7 * raw_pooled_std, (
        f"平滑后角速度 pooled std ({smoothed_pooled_std:.1f}) 应显著低于未平滑"
        f"({raw_pooled_std:.1f})"
    )

    # window <= 1：原样返回副本，逐元素相等
    unsmoothed = smooth_feature_matrix(features, window=1, fps=fps)
    assert np.array_equal(unsmoothed, features), "window=1 应逐元素等于输入"
    assert unsmoothed is not features, "应返回副本而非原数组引用"

    # 不修改输入
    features_copy = features.copy()
    smooth_feature_matrix(features, window=5, fps=fps)
    assert np.array_equal(features, features_copy), "不应就地修改输入"

    print("test_smooth_feature_matrix PASSED")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_video_loader()
    test_feature_extractor()
    test_compute_angle_degenerate_returns_nan()
    test_angular_velocity_uses_real_frame_gap()
    test_letterbox_preserves_aspect_ratio()
    test_pose_failure_repair()
    test_smooth_feature_matrix()
    print("ALL TESTS PASSED")
