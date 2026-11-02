"""蛙泳蹬腿评分 RL 环境：Gymnasium 自定义 Env，单步 MDP。

观测空间为 28 维的**原始运动学**摘要向量（10 列特征的统计量；**不含** DTW
阶段偏差 / warp）。设计要点：参考分数由模板对齐的 DTW 偏差公式算出，若把这些
偏差也放进观测，智能体只需把答案读出来即可（实测可零误差复现 → RL 退化为计算器）。
仅暴露原始运动学后，智能体必须**从动作本身学习**预测技术分，残差量化了"原始运动学
在多大程度上决定技术正确性分数"（即研究问题的 to-what-extent）。

动作空间为 ``Discrete(21)``，对应 0–100 分（步长 5）。每次 ``step`` 后
episode 立即结束（``terminated=True``），奖励为智能体评分与"参考分数"的
归一化绝对误差的相反数。

单步 episode（step 恒 ``terminated=True``）使本环境在 RL 意义上退化为
contextual bandit：不存在跨步状态转移，SB3 的 GAE / gamma 折扣在此设置下无
操作（每步 advantage = r − V(s)，没有未来回报可折算），PPO 实际执行的是带
价值函数基线的策略梯度回归。这是刻意设计——研究问题的对象是 reward function
（reference_score 能否被原始运动学预测），PPO 只是测量装置；论文表述应按
contextual bandit 定位，而非序列决策 MDP。

参考分数（reference_score）按 ``architecture.md §5`` 的 4 因素加权：
``angle 0.40 + symmetry 0.20 + timing 0.20 + smoothness 0.20``。

效度边界：reference_score 是规则公式（角度/对称/时序/平滑加权），**不含任何
外部人工真值**；本系统度量的是"原始运动学能在多大程度上重建该规则分"，
而非临床意义上的技术正确性。外部效度需教练盲评对照（见
``scripts/export_blind_eval.py``）。
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.dtw_aligner import (
    DTWAligner,
    _MIN_CYCLE_FRAMES as _ALIGNER_MIN_CYCLE_FRAMES,
    _PHASE_NAMES,
)
from src.feature_extractor import FEATURE_NAMES


logger = logging.getLogger(__name__)


# 与 FEATURE_NAMES 顺序绑定的列布局；若 FEATURE_NAMES 漂移则下方断言失败
_EXPECTED_FEATURE_NAMES: tuple[str, ...] = (
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
)
assert tuple(FEATURE_NAMES) == _EXPECTED_FEATURE_NAMES, (
    "FEATURE_NAMES 顺序已变更，kick_env 的列索引 / 归一化 / 参考分数公式"
    "需同步更新；当前 FEATURE_NAMES=" + str(FEATURE_NAMES)
)

# 特征矩阵列索引
_ANGLE_COLS: slice = slice(0, 4)        # 4 个角度
_SYMMETRY_COLS: slice = slice(4, 6)     # 2 个对称性
_VELOCITY_COLS: slice = slice(6, 10)    # 4 个角速度

# 蹬腿周期帧数下界（与 dtw_aligner 保持一致，避免 4 帧周期破坏阶段划分）
_MIN_CYCLE_FRAMES: int = _ALIGNER_MIN_CYCLE_FRAMES   # 10

# 观测向量总维度（仅原始运动学：角度 4×4 + 对称 2×2 + 角速度 4×2 = 28）。
# 不含 DTW 阶段偏差/总偏差/warp——那些是参考分公式的输入，放进观测会让评分变成
# 平凡的"读答案"。DTW 仍在 _compute_observation 内部计算，仅用于算参考分（reward）。
_OBS_DIM: int = 28

# 动作空间档数：21 档 × 5 分 = 0..100（覆盖 architecture.md 规定的 0–100 满分）
_N_ACTIONS: int = 21
_SCORE_PER_ACTION: int = 5
_MAX_SCORE: float = float((_N_ACTIONS - 1) * _SCORE_PER_ACTION)  # 100.0

# 观测归一化常量（避免 PPO 输入尺度差异过大）
_ANGLE_SCALE: float = 180.0       # 角度 / 对称性 / 偏差均按角度上界归一
_VELOCITY_SCALE: float = 1.0e3    # 角速度归一（人膝伸展典型 ~500°/s）

# 4 因素参考评分（与 architecture.md §5 一致）
_REFERENCE_WEIGHTS: dict[str, float] = {
    "angle": 0.40,
    "symmetry": 0.20,
    "timing": 0.20,
    "smoothness": 0.20,
}
# 各因素映射到 [0, 1]（0=完美，1=最差）的归一化上界。
# 数值标定自 SwimXYZ 蛙泳 base 骨架 / Side_underwater 视角 834 个周期的分布（≈各因素
# P99），使参考分数铺开到 0–100 而非压在高位窄带。换数据集 / 视角时用 train_agent 打印
# 的 factor_distribution() 重新标定（旧默认 90/60/500 远大于实际偏差，分数曾压在 [70,90]）。
_ANGLE_DEV_MAX: float = 20.0      # 加权阶段角度偏差 ≥ 20° 视为最差
_SYMMETRY_DEV_MAX: float = 18.0   # 左右关节差 ≥ 18° 视为最差
_SMOOTHNESS_MAX: float = 120.0    # 角速度 std ≥ 120°/s 视为最差
# timing：warp_amount 已自归一化到 [0,1]（0=匀速对齐，1=阶段节奏最大错位），故上限取
# 1.0，即 timing 因素 = warp_amount。若真实数据（SwimXYZ）分布集中在某区间，可据
# train_env.factor_distribution() 的 timing_warp 分位下调本值以铺开分数。整体匀速
# 快/慢会被 DTW 对齐吸收，不计入此项。
# （旧版分母用 max(Q,T) 致 DTW 天然偏大、实测 champion 75% 周期顶到上限而退化为常数。）
# 标定自 SwimXYZ 834 周期分布（warp P99≈0.87）。
_TIMING_WARP_MAX: float = 0.9

# score_scales 关键字 -> 模块默认常量的映射（用于 __init__ 缺省填充与合法性校验）
_SCALE_DEFAULTS: dict[str, float] = {
    "angle": _ANGLE_DEV_MAX,
    "symmetry": _SYMMETRY_DEV_MAX,
    "timing": _TIMING_WARP_MAX,
    "smoothness": _SMOOTHNESS_MAX,
}


def _build_obs_bounds() -> tuple[np.ndarray, np.ndarray]:
    """构造 28 维观测空间的下界 / 上界。所有维度归一化到 [0, 1] 或 [-1, 1]。

    与 :func:`_compute_observation` 的布局一致：归一化保证 PPO MlpPolicy
    输入尺度统一，避免大尺度维度（角速度）淹没小尺度维度（偏差）。
    """
    low = np.zeros(_OBS_DIM, dtype=np.float32)
    high = np.ones(_OBS_DIM, dtype=np.float32)
    # 仅角速度均值（20-23）可为负
    low[20:24] = -1.0
    return low, high


class BreaststrokeKickEnv(gym.Env):
    """蛙泳蹬腿技术评分 RL 环境（单步 MDP）。"""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        kick_cycles: list[np.ndarray],
        dtw_aligner: DTWAligner,
        seed: int | None = None,
        score_scales: dict[str, float] | None = None,
    ) -> None:
        """初始化环境。

        Args:
            kick_cycles: 预分割好的蹬腿周期列表，每项 shape ``(cycle_len, 10)``。
                内部会做浅拷贝（list），调用方后续 append/pop 不影响本环境。
            dtw_aligner: 已加载模板的 :class:`DTWAligner` 实例。
            seed: 可选随机种子。
            score_scales: 可选的参考分数尺度覆盖，键为 ``angle`` /
                ``symmetry`` / ``timing`` / ``smoothness``，对应覆盖模块常量
                ``_ANGLE_DEV_MAX`` / ``_SYMMETRY_DEV_MAX`` / ``_TIMING_WARP_MAX`` /
                ``_SMOOTHNESS_MAX``。缺省的键使用模块默认值；``None``（默认）
                等价于四键全部使用默认值，行为与旧版本逐 bit 一致。
                这些尺度标定自 SwimXYZ 合成数据分布，换数据域（如真实
                HRNet/Kinetics 姿态，角速度噪声显著更大）时评分因素会饱和，
                应从 config 的 ``reference_score.scale_max_real`` 传入按该域
                重标定的值（见 :meth:`factor_distribution` 的标定诊断）。

        Raises:
            ValueError: ``kick_cycles`` 为空，或包含 shape / 帧数不合法的元素；
                或 ``score_scales`` 含未知键 / 非正值。
        """
        super().__init__()
        if not kick_cycles:
            raise ValueError("kick_cycles 不能为空")

        # 尺度覆盖：先解析校验，供下方预计算循环（target_score）使用。
        unknown_keys = set((score_scales or {})) - set(_SCALE_DEFAULTS)
        if unknown_keys:
            raise ValueError(
                f"score_scales 含未知键 {sorted(unknown_keys)}，"
                f"合法键为 {sorted(_SCALE_DEFAULTS)}"
            )
        resolved_scales: dict[str, float] = dict(_SCALE_DEFAULTS)
        if score_scales:
            for key, value in score_scales.items():
                if not value > 0:
                    raise ValueError(
                        f"score_scales[{key!r}] 必须 > 0，得到 {value}"
                    )
                resolved_scales[key] = float(value)
        self._scale_angle: float = resolved_scales["angle"]
        self._scale_symmetry: float = resolved_scales["symmetry"]
        self._scale_timing: float = resolved_scales["timing"]
        self._scale_smoothness: float = resolved_scales["smoothness"]
        for i, c in enumerate(kick_cycles):
            if c.ndim != 2 or c.shape[1] != 10:
                raise ValueError(
                    f"kick_cycles[{i}] shape 非法: {c.shape}，期望 (cycle_len, 10)"
                )
            if c.shape[0] < _MIN_CYCLE_FRAMES:
                raise ValueError(
                    f"kick_cycles[{i}] 帧数 {c.shape[0]} 过短，"
                    f"需 ≥ {_MIN_CYCLE_FRAMES}"
                )

        # 防御性浅拷贝：避免调用方后续 append/pop 让 cache 与列表错位。
        # 预计算阶段逐周期计算观测，仅成功的周期才会进入 self.kick_cycles。
        src_cycles: list[np.ndarray] = list(kick_cycles)
        self.aligner: DTWAligner = dtw_aligner

        self._obs_low, self._obs_high = _build_obs_bounds()
        self.observation_space = spaces.Box(
            low=self._obs_low,
            high=self._obs_high,
            shape=(_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(_N_ACTIONS)

        # 预计算所有 cycle 的 obs / 偏差 / target_score，避免 reset 重复 DTW。
        # 仅成功算出观测的周期进入 self.kick_cycles；单个坏周期（如 DTW 抛异常）
        # 被跳过并告警，不致拖垮整个环境构建 / 训练。
        self.kick_cycles: list[np.ndarray] = []
        self._cached_obs: list[np.ndarray] = []
        self._cached_phase_deviations: list[dict[str, float]] = []
        self._cached_overall_deviation: list[float] = []
        self._cached_dtw_distance: list[float] = []
        self._cached_warp_amount: list[float] = []
        self._cached_target_score: list[float] = []
        for i, cycle in enumerate(src_cycles):
            try:
                obs, phase_devs, overall_dev, dtw_dist, warp_amount = (
                    self._compute_observation(cycle)
                )
                target_score = self._compute_reference_score(
                    cycle, phase_devs, overall_dev, warp_amount
                )
            except Exception as exc:  # noqa: BLE001 — 单个坏周期不应使全量崩溃
                logger.warning(
                    "跳过周期 %d（预计算失败: %s: %s）",
                    i,
                    type(exc).__name__,
                    exc,
                )
                continue
            self.kick_cycles.append(cycle)
            self._cached_obs.append(obs)
            # 复制阶段偏差字典，避免 aligner 复用同一 dict 引用造成 cache 互相污染
            self._cached_phase_deviations.append(dict(phase_devs))
            self._cached_overall_deviation.append(float(overall_dev))
            self._cached_dtw_distance.append(float(dtw_dist))
            self._cached_warp_amount.append(float(warp_amount))
            self._cached_target_score.append(target_score)
            if (i + 1) % 100 == 0:
                logger.info(
                    "BreaststrokeKickEnv 预计算进度 | %d / %d",
                    i + 1,
                    len(src_cycles),
                )
        if not self.kick_cycles:
            raise ValueError(
                "所有周期的观测预计算均失败，无可用周期；请检查模板与周期数据。"
            )
        logger.info(
            "BreaststrokeKickEnv 预计算完成 | cycles=%d / 输入 %d",
            len(self.kick_cycles),
            len(src_cycles),
        )

        # 当前 episode 状态（reset 后会被覆盖）
        self._current_obs: np.ndarray = np.zeros(_OBS_DIM, dtype=np.float32)
        self._current_phase_deviations: dict[str, float] = {
            name: 0.0 for name in _PHASE_NAMES
        }
        self._current_overall_deviation: float = 0.0
        self._current_target_score: float = 0.0

        # gym.Env.reset 会用 seed 初始化 self.np_random
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # 标准 Gym 接口
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """随机抽取一个周期、读取预计算的 28 维观测并返回。"""
        del options  # 当前环境不消费 options
        super().reset(seed=seed)

        idx = int(self.np_random.integers(0, len(self.kick_cycles)))
        cycle = self.kick_cycles[idx]

        # 拷贝返回，避免外部 in-place 修改污染 cache
        obs = self._cached_obs[idx].copy()
        phase_devs = dict(self._cached_phase_deviations[idx])
        overall_dev = self._cached_overall_deviation[idx]
        dtw_dist = self._cached_dtw_distance[idx]
        target_score = self._cached_target_score[idx]

        self._current_obs = obs
        self._current_phase_deviations = phase_devs
        self._current_overall_deviation = overall_dev
        self._current_target_score = target_score

        info: dict[str, Any] = {
            "cycle_index": idx,
            "cycle_length": int(cycle.shape[0]),
            "deviation": float(overall_dev),
            "dtw_distance": float(dtw_dist),
            "phase_deviations": dict(phase_devs),
            "target_score": float(target_score),
        }
        return obs, info

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """根据动作（评分档位）计算奖励并结束 episode。"""
        action_int = int(action)
        if not self.action_space.contains(action_int):
            raise ValueError(
                f"非法动作 {action_int}，应在 [0, {_N_ACTIONS - 1}] 内"
            )

        agent_score = float(action_int * _SCORE_PER_ACTION)
        target_score = float(self._current_target_score)
        reward = -abs(agent_score - target_score) / _MAX_SCORE

        info: dict[str, Any] = {
            "target_score": target_score,
            "agent_score": agent_score,
            "deviation": float(self._current_overall_deviation),
            "phase_deviations": dict(self._current_phase_deviations),
        }
        return self._current_obs.copy(), float(reward), True, False, info

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    def _compute_observation(
        self, cycle: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], float, float, float]:
        """将 ``(cycle_len, 10)`` 的特征矩阵转为 28 维**原始运动学**摘要向量。

        注意：DTW 对齐、阶段偏差、warp 仍在本方法内计算并**返回**（供
        :meth:`_compute_reference_score` 算参考分），但**不写入观测向量**——
        观测仅含角度/对称/角速度的统计量，迫使智能体从原始动作学习评分。

        Returns:
            ``(obs, phase_deviations, overall_deviation, dtw_distance, warp_amount)``：
            观测向量（float32，已裁剪到观测空间内）、阶段偏差字典、
            加权总偏差、归一化 DTW 距离、warp path 相对超出量（时序扭曲）。
        """
        cycle = np.asarray(cycle, dtype=np.float64)
        cycle_len = cycle.shape[0]

        angles = cycle[:, _ANGLE_COLS]          # (cycle_len, 4)
        symmetry = cycle[:, _SYMMETRY_COLS]     # (cycle_len, 2)
        velocity = cycle[:, _VELOCITY_COLS]     # (cycle_len, 4)

        ang_mean = angles.mean(axis=0)
        ang_std = angles.std(axis=0)
        ang_min = angles.min(axis=0)
        ang_max = angles.max(axis=0)
        sym_mean = symmetry.mean(axis=0)
        sym_std = symmetry.std(axis=0)
        vel_mean = velocity.mean(axis=0)
        vel_std = velocity.std(axis=0)

        # DTW 对齐 → 阶段偏差 / 总偏差 / 归一化距离
        aligned_q, aligned_t, dtw_dist = self.aligner.align(cycle)
        phase_devs = self.aligner.compute_phase_deviations(aligned_q, aligned_t)
        overall_dev = self.aligner.compute_overall_deviation(phase_devs)

        # timing：DTW warp path 超出"匀速对齐最小长度 max(Q,T)"的量，按可能的最大超出
        # 量 (Q+T-1)-max(Q,T) = min(Q,T)-1 归一化到 [0,1]。整体匀速快/慢时 DTW 仍能用
        # 近对角线 staircase 对齐（path_len≈max(Q,T) → warp≈0）；仅当周期内阶段节奏错位、
        # 需额外拉伸时 warp 才增大，1.0 对应全 Manhattan 路径（极端错位）。
        path_len = int(aligned_q.shape[0])
        template_len = int(self.aligner.template.shape[0])
        max_len = max(cycle_len, template_len)
        min_len = min(cycle_len, template_len)
        warp_amount = float(
            np.clip((path_len - max_len) / max(min_len - 1, 1), 0.0, 1.0)
        )

        # 观测仅含原始运动学统计量（28 维）。DTW 派生量（overall_dev/dtw_dist/
        # warp_amount/phase_devs）虽在上方算出并返回，但**刻意不进观测**——否则
        # 智能体可直接读出参考分。warp_amount 仍需算出以供参考分使用。
        obs = np.concatenate(
            [
                ang_mean / _ANGLE_SCALE,
                ang_std / _ANGLE_SCALE,
                ang_min / _ANGLE_SCALE,
                ang_max / _ANGLE_SCALE,
                sym_mean / _ANGLE_SCALE,
                sym_std / _ANGLE_SCALE,
                vel_mean / _VELOCITY_SCALE,
                vel_std / _VELOCITY_SCALE,
            ],
            axis=0,
        ).astype(np.float32)

        assert obs.shape == (_OBS_DIM,), f"观测维度异常: {obs.shape}"
        # 裁剪到观测空间内（防止 check_env 因极端值越界报错）。若发生裁剪则告警：
        # 说明某维归一化尺度常量偏小、把真实差异压平了信息，提示需按数据分布重标定。
        clipped = np.clip(obs, self._obs_low, self._obs_high)
        n_clipped = int(np.count_nonzero(np.abs(clipped - obs) > 1e-6))
        if n_clipped:
            logger.warning(
                "观测有 %d/%d 维超出归一化范围被裁剪（尺度常量可能偏小，"
                "考虑按数据分布重标定 _VELOCITY_SCALE 等）",
                n_clipped,
                _OBS_DIM,
            )
        obs = clipped
        return obs, phase_devs, float(overall_dev), float(dtw_dist), warp_amount

    def _compute_reference_score(
        self,
        cycle: np.ndarray,
        phase_devs: dict[str, float],
        overall_deviation: float,
        warp_amount: float,
    ) -> float:
        """按 architecture.md §5 的 4 因素加权计算参考评分。

        每个因素先映射到 ``[0, 1]``（0=完美，1=最差），再按
        ``angle 0.40 + symmetry 0.20 + timing 0.20 + smoothness 0.20`` 加权。
        最终评分 = ``(1 - 加权值) * 100``，clip 至 ``[0, 100]`` 并 round
        到 5 的倍数（与动作空间档位对齐）。

        Args:
            cycle: ``(cycle_len, 10)`` 原始特征矩阵（未归一化）。
            phase_devs: 已含 recovery/catch/power/glide 4 个阶段角度偏差。
            overall_deviation: 阶段加权后的总角度偏差。
            warp_amount: DTW warp path 相对超出量，作为 timing 因素输入。
        """
        del phase_devs  # angle 因素直接用 overall_deviation；保留参数便于扩展
        cycle = np.asarray(cycle, dtype=np.float64)

        # 1) angle：DTW 阶段加权角度偏差
        angle = float(np.clip(overall_deviation / self._scale_angle, 0.0, 1.0))

        # 2) symmetry：knee_symmetry + hip_symmetry 在整个周期的均值
        symmetry = float(
            np.clip(cycle[:, _SYMMETRY_COLS].mean() / self._scale_symmetry, 0.0, 1.0)
        )

        # 3) timing：DTW warp path 的非均匀时间扭曲量（整体速度差异已被 DTW 吸收）
        timing = float(np.clip(warp_amount / self._scale_timing, 0.0, 1.0))

        # 4) smoothness：角速度 std（越大越抖）
        smoothness = float(
            np.clip(cycle[:, _VELOCITY_COLS].std() / self._scale_smoothness, 0.0, 1.0)
        )

        weighted = (
            angle * _REFERENCE_WEIGHTS["angle"]
            + symmetry * _REFERENCE_WEIGHTS["symmetry"]
            + timing * _REFERENCE_WEIGHTS["timing"]
            + smoothness * _REFERENCE_WEIGHTS["smoothness"]
        )
        score_raw = (1.0 - weighted) * _MAX_SCORE
        score_clipped = max(0.0, min(_MAX_SCORE, score_raw))
        return float(round(score_clipped / _SCORE_PER_ACTION) * _SCORE_PER_ACTION)

    # ------------------------------------------------------------------ #
    # 标定诊断（只读，不改变评分）
    # ------------------------------------------------------------------ #

    def factor_distribution(self) -> dict[str, Any]:
        """返回 4 个参考分数因素的原始值分位数 + target_score 档位使用情况。

        用于按真实数据分布标定 ``_ANGLE_DEV_MAX / _SYMMETRY_DEV_MAX /
        _TIMING_WARP_MAX / _SMOOTHNESS_MAX``：若某因素的 P90 远小于当前对应
        ``*_MAX``，说明该因素被压缩在 [0, 小值] 区间、分数难以铺开，可把该
        ``*_MAX`` 下调到约 P90。**本方法只读，不改变任何评分。**
        """

        def _pct(values: np.ndarray) -> dict[str, float]:
            return {
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "max": float(values.max()),
            }

        angle = np.asarray(self._cached_overall_deviation, dtype=np.float64)
        timing = np.asarray(self._cached_warp_amount, dtype=np.float64)
        symmetry = np.asarray(
            [float(c[:, _SYMMETRY_COLS].mean()) for c in self.kick_cycles],
            dtype=np.float64,
        )
        smoothness = np.asarray(
            [float(c[:, _VELOCITY_COLS].std()) for c in self.kick_cycles],
            dtype=np.float64,
        )
        scores = np.asarray(self._cached_target_score, dtype=np.float64)

        return {
            "angle_overall_deviation": _pct(angle),
            "symmetry_mean": _pct(symmetry),
            "timing_warp": _pct(timing),
            "smoothness_vel_std": _pct(smoothness),
            "current_scale_maxes": {
                "angle": self._scale_angle,
                "symmetry": self._scale_symmetry,
                "timing": self._scale_timing,
                "smoothness": self._scale_smoothness,
            },
            "target_score": {
                "min": float(scores.min()),
                "max": float(scores.max()),
                "n_bins_used": int(len(set(scores.tolist()))),
                "n_bins_total": _N_ACTIONS,
            },
        }


# 暴露公共别名，方便外部 / 测试代码避免触碰下划线 import
PHASE_NAMES = _PHASE_NAMES
