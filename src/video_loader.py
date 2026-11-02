"""视频加载与抽帧模块：按目标 FPS 抽取并 resize 视频帧。"""

from __future__ import annotations

import logging
import os
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoLoader:
    """读取视频文件，按目标 FPS 抽帧并 resize 为统一尺寸。

    resize 采用等比缩放 + letterbox 补黑边（见 :meth:`_letterbox`），保持原始
    宽高比，从而不改变关节角度的几何关系。旧版本用 ``cv2.resize`` 直接非等比
    拉伸到目标尺寸，16:9 片源被压成 4:3 时会系统性扭曲关节角度。**注意**：
    历史上已用旧版本提取的 joints/features 是非等比拉伸的产物，与本版本混用
    时需注意口径不一致。
    """

    def __init__(
        self,
        video_path: str,
        fps: int = 30,
        resize_width: int = 640,
        resize_height: int = 480,
    ) -> None:
        """初始化视频加载器。

        Args:
            video_path: 视频文件路径
            fps: 目标抽帧 FPS
            resize_width: resize 后的宽度
            resize_height: resize 后的高度

        Raises:
            FileNotFoundError: 视频文件不存在
            RuntimeError: 视频文件无法打开
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.video_path = video_path
        self.target_fps = fps
        self.resize_width = resize_width
        self.resize_height = resize_height

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        self.original_fps: float = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.original_fps <= 0:
            logger.warning("视频元数据 FPS 无效，frame_interval 回退为 1")
            self.frame_interval = 1
        else:
            interval = round(self.original_fps / self.target_fps)
            self.frame_interval = interval if interval >= 1 else 1

        logger.info(
            "已打开视频 %s | 原始FPS=%.2f, 总帧数=%d, 目标FPS=%d, 抽帧间隔=%d",
            video_path,
            self.original_fps,
            self.total_frames,
            self.target_fps,
            self.frame_interval,
        )

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """等比缩放 frame 并补黑边到 (resize_height, resize_width)。

        按 ``min(resize_width / W, resize_height / H)`` 缩放比例等比缩放原帧，
        再用 ``cv2.copyMakeBorder`` 在上下/左右对称补黑边，保持原始宽高比不变
        （从而不扭曲关节角度）。缩放后与目标尺寸的差值为奇数时，多出的 1
        像素补在下边/右边。

        Args:
            frame: 原始 BGR 帧，shape (H, W, 3)

        Returns:
            shape (resize_height, resize_width, 3) 的 letterbox 后 BGR 帧
        """
        h, w = frame.shape[:2]
        scale = min(self.resize_width / w, self.resize_height / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        scaled = cv2.resize(frame, (new_w, new_h))

        pad_w = self.resize_width - new_w
        pad_h = self.resize_height - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return cv2.copyMakeBorder(
            scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    def extract_frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """按 frame_interval 抽帧，每帧 letterbox resize 后逐个 yield。

        Yields:
            (frame_index, BGR_frame) 元组，frame_index 为原视频中的帧索引。
        """
        # 重置到起点，便于多次调用
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.frame_interval == 0:
                resized = self._letterbox(frame)
                yield frame_idx, resized
            frame_idx += 1

    def extract_all_frames(self) -> np.ndarray:
        """一次性抽取所有目标帧并堆叠为数组。

        Returns:
            shape (N, resize_height, resize_width, 3) 的 BGR uint8 数组；
            若视频为空则返回 shape (0, H, W, 3) 的空数组。
        """
        frames: list[np.ndarray] = []
        for _, frame in self.extract_frames():
            frames.append(frame)
        if not frames:
            return np.empty(
                (0, self.resize_height, self.resize_width, 3), dtype=np.uint8
            )
        return np.stack(frames, axis=0)

    def release(self) -> None:
        """释放 cv2.VideoCapture 资源。"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
