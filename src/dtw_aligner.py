"""DTW 对齐模块：周期分割、与冠军模板的 DTW 时间对齐与阶段偏差计算。"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# 蹬腿周期长度过滤阈值（帧数）
_MIN_CYCLE_FRAMES: int = 10
_MAX_CYCLE_FRAMES: int = 120

# 4 个阶段及其加权
_PHASE_NAMES: tuple[str, str, str, str] = ("recovery", "catch", "power", "glide")
_PHASE_WEIGHTS: dict[str, float] = {
    "recovery": 0.15,
    "catch": 0.20,
    "power": 0.40,
    "glide": 0.25,
}

# DTW / 偏差计算只考虑前 4 列角度列
_ANGLE_COLS: int = 4


class DTWAligner:
    """加载冠军模板，对蹬腿周期做 DTW 时间对齐并计算阶段偏差。"""

    def __init__(self, template_path: str) -> None:
        """初始化并加载 .npy 格式的模板特征矩阵。

        Args:
            template_path: 模板 .npy 文件路径，内容 shape 应为 (T, 10)

        Raises:
            FileNotFoundError: 模板文件不存在
        """
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"模板文件不存在: {template_path}")

        self.template_path = template_path
        self.template: np.ndarray = np.load(template_path)
        logger.info(
            "已加载模板 %s | shape=%s",
            template_path,
            tuple(self.template.shape),
        )

        # 延迟导入 dtw-python，便于在不需要对齐的场景下避免硬依赖
        try:
            from dtw import dtw as _dtw_fn
        except ImportError as exc:
            raise ImportError(
                "未安装 dtw-python，请运行: pip install dtw-python"
            ) from exc
        self._dtw_fn = _dtw_fn

    def load_template(self) -> np.ndarray:
        """返回已加载的模板特征矩阵。

        Returns:
            shape (T, 10) 的模板特征矩阵
        """
        return self.template

    @staticmethod
    def save_template(features: np.ndarray, save_path: str) -> None:
        """将特征矩阵保存为 .npy 文件，用于从冠军视频生成模板。

        Args:
            features: shape (T, 10) 的特征矩阵
            save_path: 输出 .npy 文件路径
        """
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(save_path, features)
        logger.info("已保存模板 %s | shape=%s", save_path, tuple(features.shape))

    def segment_kick_cycles(
        self, features: np.ndarray, angle_column: int = 0
    ) -> list[np.ndarray]:
        """通过检测指定角度列的周期性波谷分割出单个蹬腿周期。

        实现要点：
            - 对 features[:, angle_column] 取负，用 scipy.signal.find_peaks
              查找极小值（即原信号的波谷）
            - 相邻波谷之间为一个完整蹬腿周期
            - 过滤帧数 < _MIN_CYCLE_FRAMES 或 > _MAX_CYCLE_FRAMES 的异常周期

        Args:
            features: shape (N, 10) 的连续特征矩阵
            angle_column: 用于波谷检测的列索引，默认 0（left_knee_angle）

        Returns:
            每个有效蹬腿周期的特征矩阵列表，每项 shape (cycle_len, 10)
        """
        if features.ndim != 2:
            raise ValueError(
                f"features 必须为 2 维，得到 {features.ndim} 维"
            )
        n = features.shape[0]
        if n < _MIN_CYCLE_FRAMES * 2:
            logger.warning(
                "特征序列过短 (N=%d)，无法分割出有效周期", n
            )
            return []

        signal = features[:, angle_column]
        # 设置波谷间最小距离，避免在抖动信号上产生密集峰
        valleys, _ = find_peaks(-signal, distance=_MIN_CYCLE_FRAMES)

        cycles: list[np.ndarray] = []
        filtered = 0
        for start, end in zip(valleys[:-1], valleys[1:]):
            cycle_len = int(end - start)
            if cycle_len < _MIN_CYCLE_FRAMES or cycle_len > _MAX_CYCLE_FRAMES:
                filtered += 1
                continue
            cycles.append(features[start:end])

        logger.info(
            "周期分割完成 | 检测到波谷 %d 个，得到周期 %d 个，过滤异常 %d 个",
            len(valleys),
            len(cycles),
            filtered,
        )
        return cycles

    def align(
        self, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """将单个蹬腿周期 query 与 self.template 做 DTW 时间对齐。

        DTW 距离只在前 4 列（角度列）上计算；warp path 用于映射全部 10 列。

        Args:
            query: shape (Q, 10) 的单周期特征矩阵

        Returns:
            (aligned_query, aligned_template, dtw_distance) 元组：
                - aligned_query: shape (L, 10)
                - aligned_template: shape (L, 10)
                - dtw_distance: 归一化距离（总距离 / warp path 长度）
              L 由 DTW warp path 决定，aligned_query 与 aligned_template 行数一致。
        """
        if query.ndim != 2 or query.shape[1] < _ANGLE_COLS:
            raise ValueError(
                f"query 形状非法: {query.shape}，需要 (Q, >=4)"
            )

        q_angles = query[:, :_ANGLE_COLS]
        t_angles = self.template[:, :_ANGLE_COLS]

        alignment = self._dtw_fn(q_angles, t_angles, keep_internals=False)
        idx_q = np.asarray(alignment.index1, dtype=np.int64)
        idx_t = np.asarray(alignment.index2, dtype=np.int64)

        aligned_query = query[idx_q]
        aligned_template = self.template[idx_t]

        path_len = int(len(idx_q))
        dtw_distance = float(alignment.distance) / max(path_len, 1)

        logger.debug(
            "DTW 对齐完成 | Q=%d, T=%d, warp_len=%d, norm_dist=%.4f",
            query.shape[0],
            self.template.shape[0],
            path_len,
            dtw_distance,
        )
        return aligned_query, aligned_template, dtw_distance

    def compute_phase_deviations(
        self,
        aligned_query: np.ndarray,
        aligned_template: np.ndarray,
    ) -> dict[str, float]:
        """将对齐后的序列等分为 4 个阶段并计算前 4 列的平均绝对偏差。

        阶段划分（按行数等分）：
            - recovery: 前 25%
            - catch:    25% – 50%
            - power:    50% – 75%
            - glide:    后 25%

        Args:
            aligned_query: shape (L, 10) 的对齐后 query 序列
            aligned_template: shape (L, 10) 的对齐后模板序列

        Returns:
            形如 {"recovery": float, "catch": float, "power": float,
            "glide": float} 的偏差字典。
        """
        if aligned_query.shape != aligned_template.shape:
            raise ValueError(
                "aligned_query 与 aligned_template 形状不一致: "
                f"{aligned_query.shape} vs {aligned_template.shape}"
            )

        diff = np.abs(
            aligned_query[:, :_ANGLE_COLS] - aligned_template[:, :_ANGLE_COLS]
        )
        l = diff.shape[0]
        # 等分边界（确保覆盖全部行，最后一段直到末尾）
        bounds = [
            (0, l // 4),
            (l // 4, l // 2),
            (l // 2, (3 * l) // 4),
            ((3 * l) // 4, l),
        ]

        deviations: dict[str, float] = {}
        for name, (s, e) in zip(_PHASE_NAMES, bounds):
            if e > s:
                deviations[name] = float(np.mean(diff[s:e]))
            else:
                # 序列过短导致段为空时，回退为 0
                deviations[name] = 0.0
        return deviations

    def compute_overall_deviation(
        self, phase_deviations: dict[str, float]
    ) -> float:
        """对 4 个阶段偏差按预设权重做加权求和。

        权重：recovery=0.15, catch=0.20, power=0.40, glide=0.25

        Args:
            phase_deviations: compute_phase_deviations 的输出

        Returns:
            加权偏差分数（越低越好）
        """
        total = 0.0
        for name, w in _PHASE_WEIGHTS.items():
            total += w * float(phase_deviations.get(name, 0.0))
        return total
