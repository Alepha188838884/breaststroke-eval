"""特征提取模块：从关节点序列计算关节角度、对称性与角速度。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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
            夹角（角度制），范围 [0, 180]
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)

        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:
            # 重合点的数值退化情形，返回 0
            return 0.0

        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle = float(np.degrees(np.arccos(cos_theta)))
        return float(np.clip(angle, 0.0, 180.0))

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
            6 项特征字典；若任一参与计算的关节置信度不足，则返回 None。
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
            1. 跳过 None 帧及 extract_frame_features 返回 None 的帧
            2. 对 4 列角度用中心差分法计算角速度（首尾用前/后向差分），dt = 1/fps
            3. 拼接为 shape (N_valid, 10) 的矩阵，列序见 FEATURE_NAMES

        Args:
            joint_sequence: 关节字典列表，失败帧为 None

        Returns:
            shape (N_valid_frames, 10) 的特征矩阵；若全部帧无效则为 (0, 10)。
        """
        valid_features: list[dict[str, float]] = []
        skipped = 0
        for joints in joint_sequence:
            if joints is None:
                skipped += 1
                continue
            f = self.extract_frame_features(joints)
            if f is None:
                skipped += 1
                continue
            valid_features.append(f)

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

        # 对前 4 列角度做中心差分计算角速度
        angles = base[:, :4]
        velocity = np.zeros_like(angles)
        if n_valid >= 2:
            # 中心差分（中间帧）；首尾用前/后向差分以保留长度
            velocity[1:-1] = (angles[2:] - angles[:-2]) / (2.0 * self.dt)
            velocity[0] = (angles[1] - angles[0]) / self.dt
            velocity[-1] = (angles[-1] - angles[-2]) / self.dt

        feature_matrix = np.concatenate([base, velocity], axis=1)
        assert feature_matrix.shape[1] == len(FEATURE_NAMES)
        return feature_matrix
