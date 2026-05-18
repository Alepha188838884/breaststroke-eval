"""视频加载与抽帧模块：按目标 FPS 抽取并 resize 视频帧。"""

from __future__ import annotations

import logging
import os
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoLoader:
    """读取视频文件，按目标 FPS 抽帧并 resize 为统一尺寸。"""

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

    def extract_frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """按 frame_interval 抽帧，每帧 resize 后逐个 yield。

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
                resized = cv2.resize(
                    frame, (self.resize_width, self.resize_height)
                )
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
