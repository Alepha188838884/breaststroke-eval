"""姿态估计模块：基于 HRNet（MMPose + MMDet）逐帧提取人体关节点。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# COCO 17 关键点索引（仅保留本项目所需关节）
_COCO_INDEX: dict[str, int] = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# 下肢关节列表（用于检查有效关节数）
_LOWER_BODY_JOINTS: list[str] = [
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# HRNet-W48 默认 config / checkpoint（mmpose 官方仓库内置路径与权重）
_HRNET_W48_CONFIG: str = (
    "configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
)
_HRNET_W48_CKPT: str = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
)

# 默认人体检测器（RTMDet-M，速度与精度均衡）
_DET_CONFIG: str = "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
_DET_CKPT: str = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_m_8xb32-300e_coco/"
    "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
)

# COCO 中 person 类别的 label id
_PERSON_LABEL: int = 0
# 人体检测的最低置信度
_DET_SCORE_THRESHOLD: float = 0.3


class PoseEstimationError(RuntimeError):
    """姿态估计失败时抛出（无人体或关节数不足）。"""


def _to_numpy(arr) -> np.ndarray:
    """统一把 torch.Tensor / ndarray 转为 numpy 数组。"""
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


class PoseEstimator:
    """对单帧图像运行 HRNet 模型，输出关节点坐标与置信度。"""

    def __init__(
        self,
        model_name: str = "hrnet_w48",
        device: str = "cuda",
        min_confidence: float = 0.3,
    ) -> None:
        """初始化人体检测器与 HRNet 姿态模型。

        Args:
            model_name: 姿态模型名称，目前仅支持 "hrnet_w48"
            device: 推理设备，"cuda" 或 "cpu"
            min_confidence: 关节点置信度阈值，低于此值视为无效

        Raises:
            ImportError: 未安装 mmpose / mmdet / mmengine
            NotImplementedError: 不支持的 model_name
        """
        try:
            from mmpose.apis import init_model as init_pose_model
            from mmpose.apis import inference_topdown
            from mmdet.apis import init_detector, inference_detector
        except ImportError as exc:
            raise ImportError(
                "未安装 mmpose / mmdet / mmengine，请按以下命令安装：\n"
                "  pip install -U openmim\n"
                "  mim install mmengine 'mmcv>=2.0.0' 'mmdet>=3.0.0' 'mmpose>=1.0.0'"
            ) from exc

        if model_name != "hrnet_w48":
            raise NotImplementedError(
                f"暂未支持的姿态模型: {model_name}（目前仅实现 hrnet_w48）"
            )

        self.model_name = model_name
        self.device = device
        self.min_confidence = min_confidence

        # 保存推理函数引用，供 estimate_single 调用
        self._inference_topdown = inference_topdown
        self._inference_detector = inference_detector

        logger.info("初始化人体检测器（mmdet, RTMDet-M）...")
        self.detector = init_detector(_DET_CONFIG, _DET_CKPT, device=device)

        logger.info("初始化 HRNet 姿态模型（mmpose, %s）...", model_name)
        self.pose_model = init_pose_model(
            _HRNET_W48_CONFIG, _HRNET_W48_CKPT, device=device
        )

    def estimate_single(
        self, frame: np.ndarray
    ) -> dict[str, tuple[float, float, float]]:
        """对单帧 BGR 图像估计关节点。

        流程：mmdet 检测人体 → 取面积最大的 bbox → mmpose 推理关键点
        → 组装关节字典并附加 mid_shoulder。

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            {joint_name: (x, y, confidence)} 字典，包含：
            left_hip / right_hip / left_knee / right_knee /
            left_ankle / right_ankle / left_shoulder / right_shoulder /
            mid_shoulder（左右肩中点，用于髋角计算）

        Raises:
            PoseEstimationError: 未检测到人体，或下肢有效关节数 < 4
        """
        # 1. 人体检测
        det_result = self._inference_detector(self.detector, frame)
        pred = det_result.pred_instances
        bboxes = _to_numpy(pred.bboxes)
        scores = _to_numpy(pred.scores)
        labels = _to_numpy(pred.labels)

        person_mask = (labels == _PERSON_LABEL) & (scores >= _DET_SCORE_THRESHOLD)
        if not bool(person_mask.any()):
            raise PoseEstimationError("未检测到人体")

        person_bboxes = bboxes[person_mask]
        # 选择面积最大的 bbox
        areas = (person_bboxes[:, 2] - person_bboxes[:, 0]) * (
            person_bboxes[:, 3] - person_bboxes[:, 1]
        )
        biggest_bbox = person_bboxes[int(np.argmax(areas))][None, :]  # shape (1, 4)

        # 2. HRNet 推理
        pose_results = self._inference_topdown(self.pose_model, frame, biggest_bbox)
        if not pose_results:
            raise PoseEstimationError("HRNet 未返回关节点结果")

        kpts = _to_numpy(pose_results[0].pred_instances.keypoints)[0]  # (17, 2)
        confs = _to_numpy(pose_results[0].pred_instances.keypoint_scores)[0]  # (17,)

        # 3. 组装关节字典
        joints: dict[str, tuple[float, float, float]] = {}
        for name, idx in _COCO_INDEX.items():
            joints[name] = (
                float(kpts[idx, 0]),
                float(kpts[idx, 1]),
                float(confs[idx]),
            )

        # 4. 检查下肢有效关节数
        valid_lower = sum(
            1
            for name in _LOWER_BODY_JOINTS
            if joints[name][2] >= self.min_confidence
        )
        if valid_lower < 4:
            raise PoseEstimationError(
                f"下肢有效关节数不足 ({valid_lower} < 4)"
            )

        # 5. 计算 mid_shoulder（左右肩中点；置信度取较小值作为合成置信度）
        ls = joints["left_shoulder"]
        rs = joints["right_shoulder"]
        joints["mid_shoulder"] = (
            (ls[0] + rs[0]) / 2.0,
            (ls[1] + rs[1]) / 2.0,
            min(ls[2], rs[2]),
        )

        return joints

    def estimate_batch(
        self, frames: np.ndarray
    ) -> list[Optional[dict]]:
        """对一组帧批量估计关节点。

        Args:
            frames: shape (N, H, W, 3) 的 BGR 图像数组

        Returns:
            长度 N 的列表，每项为关节字典；若该帧检测/估计失败则为 None。
        """
        results: list[Optional[dict]] = []
        skipped = 0
        for i in tqdm(range(len(frames)), desc="HRNet 姿态估计"):
            try:
                results.append(self.estimate_single(frames[i]))
            except PoseEstimationError as e:
                logger.warning("帧 %d 姿态估计失败: %s", i, e)
                results.append(None)
                skipped += 1
        if skipped:
            logger.warning("共跳过 %d/%d 帧", skipped, len(frames))
        return results
