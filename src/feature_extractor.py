"""特征提取模块：从关节点序列计算关节角度、对称性与角速度。"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

logger = logging.getLogger(__name__)


# 姿态崩坏帧修复参数（见 FeatureExtractor._repair_pose_failures）。
# HRNet 偶发把关节点放到共线/错位处，使某角度单帧跳到病理值（如膝角被顶到 ~0°/180°
# 再跳回）。真实蹬腿膝角天然在 ~35–180° 大幅摆动，故不能按"角度值离群"判定（会误伤
# 合法深屈膝）；崩坏的真正特征是"单帧相对局部中值的巨幅跳变"，用中值滤波偏差检测。
_POSE_SMOOTH_KERNEL: int = 5      # 中值滤波核宽（奇数），定位尖刺；对孤立尖刺稳健
_POSE_SPIKE_DEG: float = 40.0     # 单帧角度偏离局部中值 > 该值视为崩坏（保守，远高于正常帧间变化）


# 特征矩阵的列名（顺序固定，与 extract_sequence_features 输出一致）
FEATURE_NAMES: list[str] = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "knee_symmetry",
    "hip_symmetry",
    "left_knee_angular_velocity",
    "right_knee_angular_velocity",
    "left_hip_angular_velocity",
    "right_hip_angular_velocity",
]

# 4 个角度的关节依赖（顶点为 p2）
_ANGLE_GROUPS: dict[str, tuple[str, str, str]] = {
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle": ("mid_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("mid_shoulder", "right_hip", "right_knee"),
}

# 基础列（前 6 列：4 角度 + 2 对称性）
_BASE_COLS: list[str] = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "knee_symmetry",
    "hip_symmetry",
]


class FeatureExtractor:
    """从关节点字典序列中提取角度、对称性与角速度特征。"""

    def __init__(self, fps: int = 30, min_confidence: float = 0.3) -> None:
        """初始化。

        Args:
            fps: 视频帧率，用于角速度的时间步长 dt = 1/fps
            min_confidence: 关节点最小置信度，低于此值视为无效
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.min_confidence = min_confidence

    @staticmethod
    def compute_angle(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> float:
        """三点夹角（degrees），p2 为顶点。

        使用向量点积公式：cos(θ) = (v1·v2) / (|v1||v2|)，
        结果 clip 到 [0, 180]。

        Args:
            p1: 第一点 (x, y)
            p2: 顶点 (x, y)
            p3: 第三点 (x, y)

        Returns:
            夹角（角度制），范围 [0, 180]；若 p1/p3 与顶点 p2 重合导致向量范数
            退化（< 1e-8），返回 ``float("nan")``（0.0 是合法角度，不能用来
            冒充退化情形，否则会伪装成真实的深屈膝）。
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:
            # 重合点的数值退化情形，返回 NaN 以便上游识别为无效帧
            return float("nan")

        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = float(np.degrees(np.arccos(cos_theta)))
        return float(np.clip(angle, 0.0, 180.0))

    @staticmethod
    def _repair_pose_failures(angles: np.ndarray) -> tuple[np.ndarray, int]:
        """检测并线性插值修复姿态崩坏导致的角度尖刺帧（逐列独立处理）。

        判据：某角度列单帧偏离其局部中值（中值滤波，对孤立尖刺稳健）超过
        ``_POSE_SPIKE_DEG``。仅修复该列命中的帧（用同列相邻好帧线性插值），不波及
        同帧其它正常列。真实平滑的大幅摆动不会被误判。

        Args:
            angles: shape ``(N, 4)`` 的角度矩阵（不就地修改）。

        Returns:
            ``(repaired_angles, n_repaired_frames)``；``n_repaired_frames`` 为至少
            有一列被修复的帧数。某列好帧不足 2 个时跳过该列（无法可靠插值）。
        """
        A = np.array(angles, dtype=np.float64)
        n = A.shape[0]
        if n < _POSE_SMOOTH_KERNEL:
            return A, 0
        bad_any = np.zeros(n, dtype=bool)
        for j in range(A.shape[1]):
            col = A[:, j]
            smoothed = median_filter(col, size=_POSE_SMOOTH_KERNEL, mode="nearest")
            badj = np.abs(col - smoothed) > _POSE_SPIKE_DEG
            good = np.where(~badj)[0]
            if badj.any() and good.size >= 2:
                bad_idx = np.where(badj)[0]
                A[bad_idx, j] = np.interp(bad_idx, good, col[good])
                bad_any |= badj
        return A, int(bad_any.sum())

    def _all_confident(
        self,
        joints: dict[str, tuple[float, float, float]],
        names: list[str],
    ) -> bool:
        """检查指定关节名列表的置信度是否全部达标。"""
        for name in names:
            if name not in joints or joints[name][2] < self.min_confidence:
                return False
        return True

    def extract_frame_features(
        self,
        joints: dict[str, tuple[float, float, float]],
    ) -> Optional[dict[str, float]]:
        """从单帧关节字典中计算 4 个角度 + 2 个对称性指标。

        Args:
            joints: {name: (x, y, confidence)} 字典

        Returns:
            6 项特征字典（4 个角度范围 [0, 180]，2 个对称性为非负度数差）；
            若任一参与计算的关节置信度不足，或任一角度因关节点重合退化为
            NaN（见 :meth:`compute_angle`），则返回 None（该帧按无效帧丢弃）。
        """
        # 收集所有参与计算的关节名
        required: set[str] = set()
        for triple in _ANGLE_GROUPS.values():
            required.update(triple)
        if not self._all_confident(joints, list(required)):
            return None

        features: dict[str, float] = {}
        for angle_name, (a, b, c) in _ANGLE_GROUPS.items():
            p1 = (joints[a][0], joints[a][1])
            p2 = (joints[b][0], joints[b][1])
            p3 = (joints[c][0], joints[c][1])
            features[angle_name] = self.compute_angle(p1, p2, p3)

        if any(math.isnan(features[name]) for name in _ANGLE_GROUPS):
            return None

        features["knee_symmetry"] = abs(
            features["left_knee_angle"] - features["right_knee_angle"]
        )
        features["hip_symmetry"] = abs(
            features["left_hip_angle"] - features["right_hip_angle"]
        )
        return features

    def extract_sequence_features(
        self,
        joint_sequence: list[Optional[dict[str, tuple[float, float, float]]]],
    ) -> np.ndarray:
        """从整段关节序列中提取特征矩阵。

        步骤：
            1. 跳过 None 帧及 extract_frame_features 返回 None 的帧，同时记录每个
               有效帧在 joint_sequence 中的原始索引 valid_idx
            2. 对 4 列角度用中心差分法计算角速度（首尾用前/后向差分）；分母用
               相邻**有效**帧在原序列中的真实索引差 × self.dt，而非恒定的
               1/fps —— 若中间有帧被丢弃（None 或无效），相邻有效帧的真实时间
               间隔是 k/fps（k 为索引差），按 1/fps 计算会把角速度高估 k 倍。
               无丢帧时索引差恒为 1，结果与"分母恒为 1/fps"的实现完全一致。
            3. 拼接为 shape (N_valid, 10) 的矩阵，列序见 FEATURE_NAMES

        Args:
            joint_sequence: 关节字典列表，失败帧为 None

        Returns:
            shape (N_valid_frames, 10) 的特征矩阵；若全部帧无效则为 (0, 10)。
        """
        valid_features: list[dict[str, float]] = []
        valid_idx: list[int] = []
        skipped = 0
        for i, joints in enumerate(joint_sequence):
            if joints is None:
                skipped += 1
                continue
            f = self.extract_frame_features(joints)
            if f is None:
                skipped += 1
                continue
            valid_features.append(f)
            valid_idx.append(i)

        if skipped:
            logger.warning(
                "extract_sequence_features 共跳过 %d/%d 帧",
                skipped,
                len(joint_sequence),
            )

        n_valid = len(valid_features)
        if n_valid == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)

        # 组装基础角度 + 对称性矩阵 (N_valid, 6)
        base = np.array(
            [[f[c] for c in _BASE_COLS] for f in valid_features],
            dtype=np.float64,
        )

        # 姿态崩坏帧修复（在算角速度之前）：修复角度尖刺并据此重算对称性，避免尖刺
        # 污染下游 DTW 对齐 / 对称性 / 角速度 / 评分。clean 合成数据上几乎零命中。
        repaired, n_bad = self._repair_pose_failures(base[:, :4])
        if n_bad:
            base[:, :4] = repaired
            base[:, 4] = np.abs(base[:, 0] - base[:, 1])  # knee_symmetry
            base[:, 5] = np.abs(base[:, 2] - base[:, 3])  # hip_symmetry
            logger.info(
                "姿态崩坏帧修复 | %d/%d 帧的角度尖刺已插值修复", n_bad, n_valid
            )

        # 对前 4 列角度做中心差分计算角速度（用修复后的角度）；分母用有效帧的
        # 真实索引差 × dt，而非恒定 1/fps（见函数 docstring）
        angles = base[:, :4]
        velocity = np.zeros_like(angles)
        if n_valid >= 2:
            idx = np.asarray(valid_idx, dtype=np.float64)
            # 中心差分（中间帧）；首尾用前/后向差分以保留长度
            velocity[1:-1] = (angles[2:] - angles[:-2]) / (
                (idx[2:] - idx[:-2])[:, None] * self.dt
            )
            velocity[0] = (angles[1] - angles[0]) / ((idx[1] - idx[0]) * self.dt)
            velocity[-1] = (angles[-1] - angles[-2]) / ((idx[-1] - idx[-2]) * self.dt)

        feature_matrix = np.concatenate([base, velocity], axis=1)
        assert feature_matrix.shape[1] == len(FEATURE_NAMES)
        return feature_matrix


def smooth_feature_matrix(
    features: np.ndarray, window: int = 5, fps: int = 30
) -> np.ndarray:
    """平滑特征矩阵的角度列，并据此重算对称性与角速度列。

    用途：HRNet 逐帧姿态估计存在抖动，会让直接从冠军视频提取的模板角速度虚高
    （实测 pooled std≈169°/s，远超 SwimXYZ 合成数据的 ≈80°/s），把姿态估计
    噪声误当成技术基准写入模板。存模板前对角度做一次滑动均值平滑，可避免
    把噪声当作 DTW 对齐 / 参考分数 smoothness 因素的基准。

    Args:
        features: shape (N, 10) 的特征矩阵，列序见 FEATURE_NAMES；不修改输入。
        window: 滑动均值窗宽（帧）；<= 1 时视为不平滑，原样返回副本。
        fps: 帧率，用于角速度 dt = 1/fps。

    Returns:
        shape / dtype 均与输入一致的新特征矩阵：前 4 列角度经
        ``uniform_filter1d`` 平滑后，第 4-5 列对称性（|左-右|）与第 6-9 列
        角速度由平滑后的角度重新计算——与
        :meth:`FeatureExtractor.extract_sequence_features` 连续无丢帧情形
        一致：中心差分，首尾用前/后向差分，dt = 1/fps。
    """
    features = np.asarray(features)
    if window <= 1:
        return features.copy()

    out = features.copy()
    angles = uniform_filter1d(
        out[:, :4].astype(np.float64), size=window, axis=0, mode="nearest"
    )
    out[:, 0:4] = angles
    out[:, 4] = np.abs(angles[:, 0] - angles[:, 1])  # knee_symmetry
    out[:, 5] = np.abs(angles[:, 2] - angles[:, 3])  # hip_symmetry

    dt = 1.0 / fps
    n = angles.shape[0]
    velocity = np.zeros_like(angles)
    if n >= 2:
        velocity[1:-1] = (angles[2:] - angles[:-2]) / (2.0 * dt)
        velocity[0] = (angles[1] - angles[0]) / dt
        velocity[-1] = (angles[-1] - angles[-2]) / dt
    out[:, 6:10] = velocity

    return out
