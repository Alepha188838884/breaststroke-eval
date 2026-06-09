"""SwimXYZ 数据集加载器：解析 COCO17 标注并转为项目所需的关节字典序列。

返回的关节字典格式与 ``PoseEstimator.estimate_single`` 输出完全兼容，
可以直接传入 ``FeatureExtractor.extract_sequence_features``。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

# 直接复用 pose_estimator 的 COCO 索引，避免双份常量漂移
from src.pose_estimator import _COCO_INDEX


logger = logging.getLogger(__name__)


# COCO17 关键点总数（每帧的关节数量）
_NUM_COCO_KEYPOINTS: int = 17

# JSON 文件中常见的"帧列表"键
_FRAMES_KEYS: tuple[str, ...] = (
    "frames",
    "annotations",
    "keypoints_2d",
    "keypoints",
    "poses",
    "data",
)

# 单帧字典中常见的"关键点"键
_FRAME_KP_KEYS: tuple[str, ...] = (
    "keypoints_2d",
    "keypoints",
    "joints_2d",
    "joints",
    "pose_2d",
    "pose",
)


class SwimXYZLoader:
    """加载 SwimXYZ 数据集 COCO17 标注并转换为项目关节字典格式。"""

    def __init__(self, labels_dir: str, stroke: str = "breaststroke") -> None:
        """初始化加载器。

        Args:
            labels_dir: 解压后的标注目录路径
                （如 ``"data/swimxyz/Breaststroke_labels/"``）。
            stroke: 泳姿名称，仅用于日志，默认 ``"breaststroke"``。

        Raises:
            FileNotFoundError: ``labels_dir`` 不存在。
        """
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"SwimXYZ 标注目录不存在: {labels_dir}")
        self.labels_dir = labels_dir
        self.stroke = stroke
        self._sequence_paths: list[str] = self._discover_files()
        logger.info(
            "SwimXYZLoader 初始化完成 | stroke=%s | dir=%s | files=%d",
            stroke,
            labels_dir,
            len(self._sequence_paths),
        )

    def _discover_files(self) -> list[str]:
        """递归扫描 ``labels_dir`` 下所有 ``.json`` / ``.npy`` 标注文件。"""
        out: list[str] = []
        for root, _, files in os.walk(self.labels_dir):
            for name in files:
                lower = name.lower()
                if lower.endswith(".json") or lower.endswith(".npy"):
                    out.append(os.path.join(root, name))
        out.sort()
        return out

    def list_sequences(self) -> list[str]:
        """列出所有可用的标注文件路径。"""
        return list(self._sequence_paths)

    def load_sequence(
        self, label_path: str
    ) -> list[dict[str, tuple[float, float, float]]]:
        """从单个标注文件加载关节序列。

        返回值格式与 :meth:`PoseEstimator.estimate_single` 完全兼容：
        列表中每项是 ``{joint_name: (x, y, confidence)}`` 字典，
        包含 ``left_shoulder``、``right_shoulder``、``left_hip``、
        ``right_hip``、``left_knee``、``right_knee``、``left_ankle``、
        ``right_ankle`` 与 ``mid_shoulder``。

        合成数据真值的置信度全部置为 ``1.0``。

        Args:
            label_path: 标注文件路径（``.json`` 或 ``.npy``）。

        Returns:
            关节字典列表，长度等于帧数。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式无法识别为有效的 COCO17 关键点序列。
        """
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"标注文件不存在: {label_path}")

        if label_path.lower().endswith(".npy"):
            keypoints = self._load_npy_keypoints(label_path)
        else:
            keypoints = self._load_json_keypoints(label_path)

        if keypoints.ndim != 3 or keypoints.shape[1] != _NUM_COCO_KEYPOINTS:
            raise ValueError(
                f"关键点数组 shape 异常: {keypoints.shape}，"
                f"期望 (N, {_NUM_COCO_KEYPOINTS}, >=2)"
            )

        joint_sequence: list[dict[str, tuple[float, float, float]]] = []
        for frame_kpts in keypoints:
            joint_sequence.append(self._frame_to_joint_dict(frame_kpts))

        logger.info(
            "加载序列 %s | 帧数=%d", label_path, len(joint_sequence)
        )
        return joint_sequence

    def load_all_sequences(
        self,
    ) -> list[list[dict[str, tuple[float, float, float]]]]:
        """加载所有标注文件，返回关节序列列表。"""
        sequences: list[list[dict[str, tuple[float, float, float]]]] = []
        for path in self._sequence_paths:
            try:
                sequences.append(self.load_sequence(path))
            except (ValueError, FileNotFoundError) as exc:
                logger.warning("跳过无效标注 %s: %s", path, exc)
        total_frames = sum(len(s) for s in sequences)
        logger.info(
            "load_all_sequences 完成 | 序列数=%d | 总帧数=%d",
            len(sequences),
            total_frames,
        )
        return sequences

    # ------------------------------------------------------------------ #
    # 内部：格式适配
    # ------------------------------------------------------------------ #

    def _frame_to_joint_dict(
        self, frame_kpts: np.ndarray
    ) -> dict[str, tuple[float, float, float]]:
        """将 ``(17, >=2)`` 的单帧关键点矩阵转为项目所需的关节字典。

        若关键点矩阵第 3 列存在（COCO 可见度标志 v：0=未标注、1=被遮挡、
        2=可见），则按以下规则换算置信度：``v==0`` → 0.0、``v==1`` → 0.5、
        ``v>=2`` → 1.0。否则（``shape[1] == 2``）默认置信度 1.0。
        ``mid_shoulder`` 由左右肩中点计算，置信度取两者较小值（保持与
        :class:`PoseEstimator` 一致）。
        """
        has_visibility = frame_kpts.shape[1] >= 3
        joints: dict[str, tuple[float, float, float]] = {}
        for name, idx in _COCO_INDEX.items():
            x = float(frame_kpts[idx, 0])
            y = float(frame_kpts[idx, 1])
            if has_visibility:
                v = float(frame_kpts[idx, 2])
                if v <= 0.0:
                    conf = 0.0
                elif v < 2.0:
                    conf = 0.5
                else:
                    conf = 1.0
            else:
                conf = 1.0
            joints[name] = (x, y, conf)

        ls = joints["left_shoulder"]
        rs = joints["right_shoulder"]
        joints["mid_shoulder"] = (
            (ls[0] + rs[0]) / 2.0,
            (ls[1] + rs[1]) / 2.0,
            min(ls[2], rs[2]),
        )
        return joints

    @staticmethod
    def _load_npy_keypoints(path: str) -> np.ndarray:
        """加载 ``.npy`` 标注并归一为 ``(N, 17, >=2)`` 数组。"""
        arr = np.load(path, allow_pickle=False)
        return SwimXYZLoader._normalize_keypoint_array(arr, source=path)

    @staticmethod
    def _load_json_keypoints(path: str) -> np.ndarray:
        """加载 ``.json`` 标注，尝试多种常见结构后归一为 ``(N, 17, >=2)``。

        ``json.JSONDecodeError`` / ``UnicodeDecodeError`` 会被包装为
        ``ValueError`` 抛出，让上层的 ``load_all_sequences`` 可以按既有
        ``(ValueError, FileNotFoundError)`` 路径跳过该文件而非整体崩溃。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"JSON 文件 {path} 解析失败: {exc.__class__.__name__}: {exc}"
            ) from exc

        frame_arrays = SwimXYZLoader._extract_frame_arrays(data)
        if not frame_arrays:
            raise ValueError(
                "无法从 JSON 中识别出帧序列，"
                f"顶层类型={type(data).__name__}，"
                f"键={list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )

        stacked = np.stack(frame_arrays, axis=0)
        return SwimXYZLoader._normalize_keypoint_array(stacked, source=path)

    @staticmethod
    def _extract_frame_arrays(data: Any) -> list[np.ndarray]:
        """递归地从 JSON 顶层结构中抽取每帧的 ``(17, >=2)`` 关键点数组。"""
        # 情况 1: 顶层就是帧列表
        if isinstance(data, list):
            return SwimXYZLoader._list_to_frame_arrays(data)

        # 情况 2: 顶层是 dict，尝试常见键
        if isinstance(data, dict):
            for key in _FRAMES_KEYS:
                if key in data:
                    inner = data[key]
                    if isinstance(inner, list):
                        frames = SwimXYZLoader._list_to_frame_arrays(inner)
                        if frames:
                            return frames
            # 部分 SwimXYZ 变体把整段序列以矩阵形式直接放在 dict 中
            for key in ("pose_2d", "keypoints_2d", "joints_2d", "kp2d"):
                if key in data:
                    try:
                        arr = np.asarray(data[key], dtype=np.float64)
                        if (
                            arr.ndim == 3
                            and arr.shape[1] == _NUM_COCO_KEYPOINTS
                            and arr.shape[2] >= 2
                        ):
                            return [arr[i] for i in range(arr.shape[0])]
                    except (TypeError, ValueError):
                        continue
        return []

    @staticmethod
    def _list_to_frame_arrays(items: list[Any]) -> list[np.ndarray]:
        """将"帧列表"中的每项解析为 ``(17, >=2)`` 矩阵。

        偶发畸形帧（个别 coerce 失败）会被跳过而非丢弃整段序列，避免一坏帧
        损失整段数据。若 **全部** 帧都无法解析则返回 ``[]``，让上层
        ``_extract_frame_arrays`` 按"非帧序列"继续尝试其它结构 / 键。
        """
        frames: list[np.ndarray] = []
        skipped = 0
        for item in items:
            frame = SwimXYZLoader._coerce_frame_keypoints(item)
            if frame is None:
                skipped += 1
                continue
            frames.append(frame)
        if not frames:
            return []
        if skipped:
            logger.warning(
                "_list_to_frame_arrays 跳过 %d/%d 个无法解析的畸形帧",
                skipped,
                len(items),
            )
        return frames

    @staticmethod
    def _coerce_frame_keypoints(item: Any) -> np.ndarray | None:
        """把单帧条目（dict 或 list）转换为 ``(17, >=2)`` 关键点矩阵。"""
        # dict 形式：在已知关键点字段中寻找
        if isinstance(item, dict):
            for key in _FRAME_KP_KEYS:
                if key in item:
                    arr = SwimXYZLoader._array_from_keypoints(item[key])
                    if arr is not None:
                        return arr
            return None

        # list / ndarray 形式：可能是 (17, 2)、(17, 3) 或 flat (34,)/(51,)
        return SwimXYZLoader._array_from_keypoints(item)

    @staticmethod
    def _array_from_keypoints(raw: Any) -> np.ndarray | None:
        """把任意关键点表达形式转为 ``(17, >=2)`` ``float64`` 数组。"""
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return None

        if arr.ndim == 2 and arr.shape[0] == _NUM_COCO_KEYPOINTS and arr.shape[1] >= 2:
            return arr
        if arr.ndim == 1:
            # flat 形式：(17*k,)
            if arr.size == _NUM_COCO_KEYPOINTS * 2:
                return arr.reshape(_NUM_COCO_KEYPOINTS, 2)
            if arr.size == _NUM_COCO_KEYPOINTS * 3:
                return arr.reshape(_NUM_COCO_KEYPOINTS, 3)
        return None

    @staticmethod
    def _normalize_keypoint_array(arr: np.ndarray, source: str) -> np.ndarray:
        """将各种合法形状归一为 ``(N, 17, >=2)`` ``float64`` 数组。"""
        arr = np.asarray(arr, dtype=np.float64)

        # 已经是目标形状
        if arr.ndim == 3 and arr.shape[1] == _NUM_COCO_KEYPOINTS and arr.shape[2] >= 2:
            return arr

        # (N, 17*2) 或 (N, 17*3) 的扁平形式
        if arr.ndim == 2 and arr.shape[1] in (
            _NUM_COCO_KEYPOINTS * 2,
            _NUM_COCO_KEYPOINTS * 3,
        ):
            per_kp = arr.shape[1] // _NUM_COCO_KEYPOINTS
            return arr.reshape(arr.shape[0], _NUM_COCO_KEYPOINTS, per_kp)

        raise ValueError(
            f"无法将标注 {source} 归一为 (N, 17, >=2) 形状，"
            f"实际 shape={arr.shape}, dtype={arr.dtype}"
        )
