"""集成测试：FeatureExtractor → DTWAligner 完整数据流。

模块 2 PoseEstimator 需 GPU，此处用合成关节序列代替；
模块 1 VideoLoader 在 test_modules_1_3.py 中已覆盖。

运行方式：python tests/test_integration_1_3_4.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.feature_extractor import FeatureExtractor
from src.dtw_aligner import DTWAligner


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

_TEMPLATE_PATH = "/tmp/test_integration_template.npy"

_N_FRAMES = 150           # 总帧数
_PERIOD = 30              # 每个蹬腿周期帧数 → 5 个完整周期
_FPS = 30

# 简化人体几何：左右髋共点于躯干中线，mid_shoulder 正上方，
# 避免肩-髋水平偏移污染髋角；左右腿分别向 ±x 弯曲。
_HIP_POS = (100.0, 200.0)
_MID_SHOULDER_POS = (100.0, 100.0)
_L_THIGH = 100.0
_L_SHIN = 100.0
_NOISE_DEG = 3.0          # 角度噪声 ±3°
_CONF = 0.9               # 关节置信度（> FeatureExtractor 默认阈值 0.3）

_REQUIRED_JOINTS = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "mid_shoulder",
}


# --------------------------------------------------------------------------- #
# 步骤 1：模拟 PoseEstimator 输出
# --------------------------------------------------------------------------- #

def _make_joint_sequence(
    n_frames: int = _N_FRAMES,
    period: int = _PERIOD,
    seed: int = 0,
) -> list[Optional[dict[str, tuple[float, float, float]]]]:
    """根据目标角度反算关节坐标，生成合成蛙泳蹬腿关节序列。

    几何关系（图像坐标系 y 朝下）：
      - 设 α = 180° - θ_hip，则 v(hip→knee) 与 v(hip→shoulder) 之夹角即为 θ_hip
      - 设 β = 180° - θ_knee，则 v(knee→ankle) 与 v(knee→hip) 之夹角即为 θ_knee
      - 左右腿用 ±sign 镜像，保证左右对称
    """
    rng = np.random.default_rng(seed)

    # 预生成平滑噪声：每个角度通道一条 ±3° 范围内的低频抖动序列。
    # 直接的逐帧白噪声会在余弦波谷附近（信号梯度≈0）形成大量伪局部极小，
    # 干扰 segment_kick_cycles 的波谷检测；用长度为 7 的卷积核做平滑后，
    # 高频成分被抑制，振幅整体仍 ≲ ±3°，足以模拟真实左右非对称偏差。
    raw_noise = rng.uniform(-_NOISE_DEG, _NOISE_DEG, size=(n_frames, 4))
    kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth_noise = np.stack(
        [np.convolve(raw_noise[:, c], kernel, mode="same") for c in range(4)],
        axis=1,
    )  # shape (n_frames, 4): [L_knee, R_knee, L_hip, R_hip]

    sequence: list[Optional[dict[str, tuple[float, float, float]]]] = []
    for t in range(n_frames):
        phase = 2.0 * np.pi * (t % period) / period
        # 膝盖角：cos=1 → 170° (蹬开)， cos=-1 → 70° (收腿)，valley 在 phase=π
        theta_knee = 120.0 + 50.0 * np.cos(phase)
        # 髋角：cos=1 → 175°， cos=-1 → 140°
        theta_hip = 157.5 + 17.5 * np.cos(phase)

        joints: dict[str, tuple[float, float, float]] = {}
        for idx, (side, sign) in enumerate([("left", 1.0), ("right", -1.0)]):
            tk = theta_knee + float(smooth_noise[t, idx])
            th = theta_hip + float(smooth_noise[t, 2 + idx])
            alpha = np.deg2rad(180.0 - th)
            beta = np.deg2rad(180.0 - tk)

            kx = _HIP_POS[0] + sign * _L_THIGH * np.sin(alpha)
            ky = _HIP_POS[1] + _L_THIGH * np.cos(alpha)
            ax = kx + sign * _L_SHIN * np.sin(alpha - beta)
            ay = ky + _L_SHIN * np.cos(alpha - beta)

            joints[f"{side}_hip"] = (_HIP_POS[0], _HIP_POS[1], _CONF)
            joints[f"{side}_knee"] = (float(kx), float(ky), _CONF)
            joints[f"{side}_ankle"] = (float(ax), float(ay), _CONF)

        joints["mid_shoulder"] = (
            _MID_SHOULDER_POS[0],
            _MID_SHOULDER_POS[1],
            _CONF,
        )
        sequence.append(joints)
    return sequence


def step1_fake_joints() -> list[Optional[dict[str, tuple[float, float, float]]]]:
    seq = _make_joint_sequence()
    assert len(seq) == _N_FRAMES, f"期望 {_N_FRAMES} 帧，得到 {len(seq)}"
    first = seq[0]
    assert first is not None, "首帧不应为 None"
    # 每帧都应包含所需的全部关节
    assert _REQUIRED_JOINTS.issubset(first.keys()), (
        f"缺关节: {_REQUIRED_JOINTS - set(first.keys())}"
    )
    # 置信度
    for name, (_, _, c) in first.items():
        assert c == _CONF, f"{name} 置信度异常: {c}"
    print("step1_fake_joints PASSED")
    return seq


# --------------------------------------------------------------------------- #
# 步骤 2：FeatureExtractor
# --------------------------------------------------------------------------- #

def step2_extract_features(
    seq: list[Optional[dict[str, tuple[float, float, float]]]],
) -> np.ndarray:
    extractor = FeatureExtractor(fps=_FPS, min_confidence=0.3)
    features = extractor.extract_sequence_features(seq)
    assert features.shape == (_N_FRAMES, 10), (
        f"期望 ({_N_FRAMES}, 10)，得到 {features.shape}"
    )

    angles = features[:, :4]  # 4 个角度列
    col_names = ["left_knee", "right_knee", "left_hip", "right_hip"]
    means = angles.mean(axis=0)
    mins = angles.min(axis=0)
    maxs = angles.max(axis=0)
    for i, name in enumerate(col_names):
        print(
            f"  {name}_angle: mean={means[i]:.2f}, range=[{mins[i]:.2f}, {maxs[i]:.2f}]"
        )

    # 合理性：膝角应主要落在 [70-3, 170+3]，髋角在 [140-3, 175+3]
    # 给少量边界冗余以吸收浮点误差
    assert mins[0] >= 65.0 and maxs[0] <= 175.0, "left_knee_angle 越界"
    assert mins[1] >= 65.0 and maxs[1] <= 175.0, "right_knee_angle 越界"
    assert mins[2] >= 135.0 and maxs[2] <= 180.0, "left_hip_angle 越界"
    assert mins[3] >= 135.0 and maxs[3] <= 180.0, "right_hip_angle 越界"

    print("step2_extract_features PASSED")
    return features


# --------------------------------------------------------------------------- #
# 步骤 3：用前 30 帧保存为模板
# --------------------------------------------------------------------------- #

def step3_save_template(features: np.ndarray) -> None:
    template = features[:_PERIOD]
    assert template.shape == (_PERIOD, 10)
    DTWAligner.save_template(template, _TEMPLATE_PATH)
    assert os.path.isfile(_TEMPLATE_PATH), "模板文件未保存"
    print(f"  saved template to {_TEMPLATE_PATH} | shape={template.shape}")
    print("step3_save_template PASSED")


# --------------------------------------------------------------------------- #
# 步骤 4：DTWAligner 全流程
# --------------------------------------------------------------------------- #

def step4_dtw(features: np.ndarray) -> None:
    aligner = DTWAligner(_TEMPLATE_PATH)

    cycles = aligner.segment_kick_cycles(features, angle_column=0)
    cycle_lens = [int(c.shape[0]) for c in cycles]
    print(f"  segmented {len(cycles)} cycles, lens={cycle_lens}")
    assert 4 <= len(cycles) <= 6, f"期望 4-6 个周期，得到 {len(cycles)}"

    # align
    aq, at, dist = aligner.align(cycles[0])
    assert isinstance(aq, np.ndarray) and isinstance(at, np.ndarray)
    assert aq.shape == at.shape, (
        f"aligned shape 不一致: {aq.shape} vs {at.shape}"
    )
    assert aq.shape[1] == 10, f"列数应为 10，得到 {aq.shape[1]}"
    assert isinstance(dist, float), f"dtw_distance 应为 float，得到 {type(dist)}"
    print(f"  align: aligned_shape={aq.shape}, dtw_distance={dist:.4f}")

    # 阶段偏差
    devs = aligner.compute_phase_deviations(aq, at)
    assert set(devs.keys()) == {"recovery", "catch", "power", "glide"}, (
        f"phase key 不匹配: {set(devs.keys())}"
    )
    for name, v in devs.items():
        assert isinstance(v, float) and v >= 0.0, f"{name} 偏差异常: {v}"
    print(
        "  phase deviations: "
        + ", ".join(f"{k}={v:.4f}" for k, v in devs.items())
    )

    # 总偏差
    overall = aligner.compute_overall_deviation(devs)
    assert isinstance(overall, float) and overall >= 0.0, (
        f"overall_deviation 异常: {overall}"
    )
    print(f"  overall_deviation = {overall:.4f}")
    print("step4_dtw PASSED")


# --------------------------------------------------------------------------- #

def main() -> None:
    try:
        seq = step1_fake_joints()
        features = step2_extract_features(seq)
        step3_save_template(features)
        step4_dtw(features)
        print("ALL INTEGRATION TESTS PASSED")
    finally:
        if os.path.isfile(_TEMPLATE_PATH):
            os.remove(_TEMPLATE_PATH)


if __name__ == "__main__":
    main()
