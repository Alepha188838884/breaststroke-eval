"""模块 5（BreaststrokeKickEnv）的轻量级测试。

不依赖 GPU / 真实数据集，用随机生成的蹬腿周期 + mock DTWAligner 进行验证。
运行方式：``python tests/test_kick_env.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.kick_env import BreaststrokeKickEnv


# --------------------------------------------------------------------------- #
# 常量与工具
# --------------------------------------------------------------------------- #

_OBS_DIM: int = 35
_N_ACTIONS: int = 21
_MAX_SCORE: float = 100.0
_CYCLE_LEN: int = 30
_FEATURE_DIM: int = 10
_N_CYCLES: int = 5


# 测试与生产共用同一份 phase 权重 / 阶段名，避免双份常量漂移
from src.dtw_aligner import _PHASE_NAMES, _PHASE_WEIGHTS  # noqa: E402


class _MockDTWAligner:
    """伪 DTWAligner：免去 dtw-python 依赖，按确定性规则生成偏差。

    - ``align``：返回原 query 与"另一个版本"作为 aligned_template，
      并给出一个可计算的归一化距离。
    - ``compute_phase_deviations``：返回 4 个固定阶段名的偏差。
    - ``compute_overall_deviation``：直接复用 ``dtw_aligner._PHASE_WEIGHTS``。
    - ``template``：``BreaststrokeKickEnv._compute_reference_score`` 计算
      timing 因素时需要 ``template.shape[0]``；这里给出与 ``_CYCLE_LEN``
      相同的零矩阵以让 timing=0。
    """

    template: np.ndarray = np.zeros((_CYCLE_LEN, _FEATURE_DIM), dtype=np.float64)

    def align(
        self, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # 让 template = query - 1.0（在前 4 列），dist 取角度列的均值
        template = query.copy()
        template[:, :4] -= 1.0
        dist = float(np.mean(np.abs(query[:, :4] - template[:, :4])))
        return query.copy(), template, dist

    def compute_phase_deviations(
        self, aligned_query: np.ndarray, aligned_template: np.ndarray
    ) -> dict[str, float]:
        diff = np.abs(aligned_query[:, :4] - aligned_template[:, :4])
        l = diff.shape[0]
        bounds = [
            (0, l // 4),
            (l // 4, l // 2),
            (l // 2, (3 * l) // 4),
            ((3 * l) // 4, l),
        ]
        out: dict[str, float] = {}
        for name, (s, e) in zip(_PHASE_NAMES, bounds):
            out[name] = float(np.mean(diff[s:e])) if e > s else 0.0
        return out

    @staticmethod
    def compute_overall_deviation(
        phase_deviations: dict[str, float],
    ) -> float:
        total = 0.0
        for name, w in _PHASE_WEIGHTS.items():
            total += w * float(phase_deviations.get(name, 0.0))
        return total


def _make_random_cycles(
    n_cycles: int = _N_CYCLES,
    cycle_len: int = _CYCLE_LEN,
    seed: int = 0,
) -> list[np.ndarray]:
    """生成 ``n_cycles`` 个 shape ``(cycle_len, 10)`` 的随机周期。

    所有特征值都在观测空间合理范围内，避免裁剪掩盖问题：
    角度 ∈ [60, 170]、对称性 ∈ [0, 30]、角速度 ∈ [-200, 200]。
    """
    rng = np.random.default_rng(seed)
    cycles: list[np.ndarray] = []
    for _ in range(n_cycles):
        feat = np.empty((cycle_len, _FEATURE_DIM), dtype=np.float64)
        feat[:, 0:4] = rng.uniform(60.0, 170.0, size=(cycle_len, 4))
        feat[:, 4:6] = rng.uniform(0.0, 30.0, size=(cycle_len, 2))
        feat[:, 6:10] = rng.uniform(-200.0, 200.0, size=(cycle_len, 4))
        cycles.append(feat)
    return cycles


def _make_env(seed: int = 0) -> BreaststrokeKickEnv:
    cycles = _make_random_cycles(seed=seed)
    aligner = _MockDTWAligner()
    # mock 与真实 DTWAligner 鸭子类型兼容；BreaststrokeKickEnv 只调用
    # align / compute_phase_deviations / compute_overall_deviation 三个方法
    return BreaststrokeKickEnv(cycles, aligner, seed=seed)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. spaces & 基本结构
# --------------------------------------------------------------------------- #

def test_spaces() -> None:
    env = _make_env()
    assert env.observation_space.shape == (_OBS_DIM,), (
        f"observation_space.shape 期望 (35,)，得到 {env.observation_space.shape}"
    )
    assert env.observation_space.dtype == np.float32, (
        f"observation_space.dtype 期望 float32，得到 {env.observation_space.dtype}"
    )
    assert env.action_space.n == _N_ACTIONS, (
        f"action_space.n 期望 {_N_ACTIONS}，得到 {env.action_space.n}"
    )
    print("test_spaces PASSED")


# --------------------------------------------------------------------------- #
# 2. reset 返回 shape (35,) float32 观测
# --------------------------------------------------------------------------- #

def test_reset_shape() -> None:
    env = _make_env()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), f"obs 应为 ndarray，得到 {type(obs)}"
    assert obs.shape == (_OBS_DIM,), f"obs.shape 期望 (35,)，得到 {obs.shape}"
    assert obs.dtype == np.float32, f"obs.dtype 期望 float32，得到 {obs.dtype}"
    assert env.observation_space.contains(obs), "obs 必须落在观测空间内"
    assert isinstance(info, dict), "info 应为 dict"
    for key in ("target_score", "deviation", "phase_deviations", "cycle_length"):
        assert key in info, f"reset() info 缺少键 {key}"
    print("test_reset_shape PASSED")


# --------------------------------------------------------------------------- #
# 3. step 返回 terminated=True 与合理 reward
# --------------------------------------------------------------------------- #

def test_step_terminates_and_reward_range() -> None:
    env = _make_env()
    env.reset()
    # action=20 → 100 分（满分），验证动作空间覆盖 architecture.md 规定的 0–100
    for action in [0, 5, 10, 19, 20]:
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        del info
        assert terminated is True, (
            f"单步 MDP，terminated 应为 True，得到 {terminated}"
        )
        assert truncated is False, (
            f"truncated 应为 False，得到 {truncated}"
        )
        assert obs.shape == (_OBS_DIM,)
        assert obs.dtype == np.float32
        assert -1.0 - 1e-6 <= reward <= 0.0 + 1e-6, (
            f"reward 应在 [-1, 0]，得到 {reward}"
        )
    print("test_step_terminates_and_reward_range PASSED")


# --------------------------------------------------------------------------- #
# 4. info 字典包含 target_score / agent_score / deviation / phase_deviations
# --------------------------------------------------------------------------- #

def test_step_info_keys() -> None:
    env = _make_env()
    env.reset()
    _, _, _, _, info = env.step(10)
    for key in ("target_score", "agent_score", "deviation", "phase_deviations"):
        assert key in info, f"step() info 缺少键 {key}"

    assert info["agent_score"] == 50.0, (
        f"action=10 应映射到 score=50，得到 {info['agent_score']}"
    )

    # target_score 必须是 5 的倍数且落在 [0, 100]
    ts = info["target_score"]
    assert 0.0 <= ts <= _MAX_SCORE, f"target_score 越界: {ts}"
    assert abs(ts / 5.0 - round(ts / 5.0)) < 1e-6, (
        f"target_score 必须是 5 的倍数，得到 {ts}"
    )

    pd = info["phase_deviations"]
    assert isinstance(pd, dict)
    assert set(pd.keys()) == set(_PHASE_NAMES), (
        f"phase_deviations 键不匹配: {set(pd.keys())}"
    )
    print("test_step_info_keys PASSED")


# --------------------------------------------------------------------------- #
# 5. 奖励等于 target/agent 误差的归一化负值
# --------------------------------------------------------------------------- #

def test_reward_matches_score_error() -> None:
    env = _make_env()
    env.reset()
    _, reward, _, _, info = env.step(0)
    expected = -abs(info["target_score"] - info["agent_score"]) / _MAX_SCORE
    assert abs(reward - expected) < 1e-6, (
        f"reward 与公式不符: reward={reward}, expected={expected}"
    )
    print("test_reward_matches_score_error PASSED")


# --------------------------------------------------------------------------- #
# 6. Gymnasium 官方 env_checker 合规性
# --------------------------------------------------------------------------- #

def test_passes_check_env() -> None:
    """确保 BreaststrokeKickEnv 通过 gymnasium.utils.env_checker.check_env。"""
    from gymnasium.utils.env_checker import check_env

    env = _make_env()
    check_env(env, warn=True, skip_render_check=True)
    print("test_passes_check_env PASSED")


# --------------------------------------------------------------------------- #
# 7. obs / info 拷贝：外部修改不应污染内部 cache
# --------------------------------------------------------------------------- #

def test_obs_and_info_are_isolated_from_cache() -> None:
    """reset/step 返回的 obs 与 info dict 必须是独立副本，外部修改不影响后续 episode。"""
    env = _make_env(seed=123)
    obs_a, info_a = env.reset(seed=0)
    obs_baseline = obs_a.copy()
    devs_baseline = dict(info_a["phase_deviations"])

    # 用户随手污染
    obs_a[:] = -9999.0
    info_a["phase_deviations"]["recovery"] = -1.0

    # 重新触发同一周期：seed=0 应让 np_random 复现同一 idx
    obs_b, info_b = env.reset(seed=0)
    assert np.allclose(obs_b, obs_baseline), "cache 中的 obs 被外部修改污染了"
    assert info_b["phase_deviations"]["recovery"] == devs_baseline["recovery"], (
        "cache 中的 phase_deviations 被外部修改污染了"
    )

    # step 同理
    obs_step, _, _, _, info_step = env.step(0)
    obs_step[:] = -1.0
    info_step["phase_deviations"]["catch"] = -1.0
    env.reset(seed=0)
    obs_again, info_again = env.reset(seed=0)
    assert np.allclose(obs_again, obs_baseline), "step 返回值被修改后污染了 cache"
    assert info_again["phase_deviations"]["catch"] == devs_baseline["catch"], (
        "step 的 info 被修改后污染了 cache"
    )
    print("test_obs_and_info_are_isolated_from_cache PASSED")


# --------------------------------------------------------------------------- #
# 8. 调用方在构造后 append/pop kick_cycles 不应影响内部 cache
# --------------------------------------------------------------------------- #

def test_kick_cycles_defensive_copy() -> None:
    """__init__ 必须对 kick_cycles 做浅拷贝；外部 append/pop 不应让 idx 越界。"""
    cycles = _make_random_cycles()
    aligner = _MockDTWAligner()
    user_list = list(cycles)
    env = BreaststrokeKickEnv(user_list, aligner, seed=0)  # type: ignore[arg-type]

    # 用户在构造后修改源列表
    user_list.append(np.zeros((_CYCLE_LEN, _FEATURE_DIM), dtype=np.float64))
    user_list.pop(0)

    # env 内部应该仍以构造时的 _N_CYCLES 为准
    assert len(env.kick_cycles) == _N_CYCLES, (
        f"env.kick_cycles 长度应保留构造时的 {_N_CYCLES}，"
        f"得到 {len(env.kick_cycles)}"
    )
    for _ in range(5):
        env.reset()
        env.step(0)  # 不应 IndexError
    print("test_kick_cycles_defensive_copy PASSED")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_spaces()
    test_reset_shape()
    test_step_terminates_and_reward_range()
    test_step_info_keys()
    test_reward_matches_score_error()
    test_passes_check_env()
    test_obs_and_info_are_isolated_from_cache()
    test_kick_cycles_defensive_copy()
    print("ALL TESTS PASSED")
