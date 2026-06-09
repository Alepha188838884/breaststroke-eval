"""模块 6（FeedbackGenerator）：将有符号阶段×关节偏差转为方向性改进建议。

输入为 :meth:`DTWAligner.compute_signed_phase_feature_deviations` 的输出
（``{phase: ndarray(shape=(4,))}``，分量对应 left_knee / right_knee /
left_hip / right_hip 角度相对模板**对应阶段**的有符号平均偏差）：

    - 偏差 > 0：该关节角度大于模板 → 相对模板更伸展（more extended）；
    - 偏差 < 0：该关节角度小于模板 → 相对模板更屈曲（more flexed）。

措辞刻意采用"相对模板（more extended / more flexed than reference）"这一阶段中
立表述，而非"伸展不足/过度"——因为各阶段的生物力学目标不同（recovery 本应屈膝、
power/glide 本应伸展），绝对的"伸展不足"会在屈膝阶段产生误导；阶段的合理性由模板
本身承载。每条建议按 **严重度 = |偏差| × 阶段权重** 排序（power 阶段权重最高 0.40，
与 ``dtw_aligner._PHASE_WEIGHTS`` 一致），仅保留超过阈值的项并截断到 ``max_items`` 条。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# 有符号偏差向量 4 个分量的标签，顺序与 FEATURE_NAMES[:4] 一致
_ANGLE_LABELS: tuple[str, str, str, str] = (
    "left knee",
    "right knee",
    "left hip",
    "right hip",
)

# 阶段权重（与 dtw_aligner._PHASE_WEIGHTS 一致；power 阶段最重）
_PHASE_WEIGHTS: dict[str, float] = {
    "recovery": 0.15,
    "catch": 0.20,
    "power": 0.40,
    "glide": 0.25,
}

# 分数高于此值且无超阈偏差时给出正面反馈
_PRAISE_SCORE: float = 85.0


class FeedbackGenerator:
    """根据有符号阶段×关节偏差生成按优先级排序的改进建议。"""

    def __init__(
        self,
        deviation_threshold_deg: float = 10.0,
        max_items: int = 5,
    ) -> None:
        """初始化。

        Args:
            deviation_threshold_deg: 单阶段单关节有符号偏差的绝对值超过此度数
                才触发一条建议（过滤微小偏差）。
            max_items: 最多返回的建议条数（按严重度排序后截断）。

        Raises:
            ValueError: 阈值为负，或 ``max_items`` < 1。
        """
        if deviation_threshold_deg < 0:
            raise ValueError(
                f"deviation_threshold_deg 必须 ≥ 0，得到 {deviation_threshold_deg}"
            )
        if max_items < 1:
            raise ValueError(f"max_items 必须 ≥ 1，得到 {max_items}")
        self.threshold = float(deviation_threshold_deg)
        self.max_items = int(max_items)

    def generate(
        self,
        score: float,
        signed_phase_deviations: dict[str, np.ndarray],
    ) -> list[str]:
        """生成方向性改进建议列表。

        Args:
            score: 该周期的评分（0–100），仅用于无显著偏差时决定正面/中性措辞。
            signed_phase_deviations: ``DTWAligner.compute_signed_phase_feature_deviations``
                的输出；每个阶段对应长度 4 的有符号偏差向量。

        Returns:
            按严重度降序排列的建议字符串列表（最多 ``max_items`` 条）。无超阈偏差
            时返回单条正面（高分）或中性（其余）反馈。

        Raises:
            ValueError: 某阶段的偏差向量长度不为 4。
        """
        findings: list[tuple[float, str]] = []  # (severity, message)
        for phase, vec in signed_phase_deviations.items():
            arr = np.asarray(vec, dtype=np.float64).ravel()
            if arr.shape[0] != len(_ANGLE_LABELS):
                raise ValueError(
                    f"阶段 {phase} 的偏差向量长度应为 {len(_ANGLE_LABELS)}，"
                    f"得到 {arr.shape[0]}"
                )
            weight = _PHASE_WEIGHTS.get(phase, 0.0)
            for label, d in zip(_ANGLE_LABELS, arr):
                if abs(d) < self.threshold:
                    continue
                severity = abs(float(d)) * weight
                interp = self._interpret(float(d))
                msg = (
                    f"{phase.capitalize()} phase — {label}: {interp} "
                    f"than reference ({d:+.0f}°)"
                )
                findings.append((severity, msg))

        if not findings:
            if score >= _PRAISE_SCORE:
                return [
                    "Technique closely matches the reference across all phases; "
                    "no major corrections needed."
                ]
            return [
                "No single joint deviates beyond the feedback threshold; "
                "focus on overall consistency and tempo."
            ]

        findings.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in findings[: self.max_items]]

    @staticmethod
    def _interpret(signed_dev: float) -> str:
        """把有符号偏差翻译成阶段中立的方向性描述。

        正偏差=角度大于模板=相对模板更伸展；负偏差=相对模板更屈曲。
        刻意不用"伸展不足/过度"，避免在屈膝阶段（recovery）产生误导。
        """
        return "more extended" if signed_dev > 0 else "more flexed"
