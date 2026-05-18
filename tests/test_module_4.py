"""模块 4（DTWAligner）的轻量级测试。

不依赖 GPU，不使用 unittest / pytest，直接用 assert + print。
运行方式：python tests/test_module_4.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dtw_aligner import DTWAligner


# --------------------------------------------------------------------------- #
# 常量与工具
# --------------------------------------------------------------------------- #

_TEMPLATE_PATH = "/tmp/test_template.npy"
_TEMPLATE_FRAMES = 60
_FEATURE_DIM = 10

# 角度范围：valley≈60°, peak≈170°（中心 115°, 振幅 55°）
_ANGLE_CENTER = 115.0
_ANGLE_AMPL = 55.0


def _make_template(n_frames: int = _TEMPLATE_FRAMES, seed: int = 0) -> np.ndarray:
    """生成 (n_frames, 10) 的合成模板：前 4 列为不同相位的正弦角度，后 6 列随机。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    # 4 列不同相位（各错开 π/8），保证 angle ∈ [60, 170]
    phases = [0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8]
    feat = np.zeros((n_frames, _FEATURE_DIM), dtype=np.float64)
    for i, phi in enumerate(phases):
        # 让第 0 列波谷出现在 t=0 附近，便于周期分割检测一致
        feat[:, i] = _ANGLE_CENTER - _ANGLE_AMPL * np.cos(t + phi)
    feat[:, 4:] = rng.standard_normal((n_frames, _FEATURE_DIM - 4))
    return feat


def _make_continuous_features(
    n_cycles: int = 5,
    cycle_len: int = 60,
    seed: int = 1,
) -> np.ndarray:
    """生成 n_cycles * cycle_len 帧的连续特征矩阵；第 0 列为多个完整正弦周期。"""
    rng = np.random.default_rng(seed)
    total = n_cycles * cycle_len
    # 用整数倍 2π 保证恰好 n_cycles 个完整周期
    t = np.linspace(0.0, n_cycles * 2.0 * np.pi, total, endpoint=False)
    feat = np.zeros((total, _FEATURE_DIM), dtype=np.float64)
    # 用 +cos(t)：峰在 t=0（≈170°），谷在 t=π（≈60°），保证波谷都落在序列内部，
    # 避免 find_peaks 在边界处漏检
    feat[:, 0] = _ANGLE_CENTER + _ANGLE_AMPL * np.cos(t)
    # 其它列填随机即可，不影响分割
    feat[:, 1:4] = _ANGLE_CENTER + 10.0 * rng.standard_normal((total, 3))
    feat[:, 4:] = rng.standard_normal((total, _FEATURE_DIM - 4))
    return feat


# --------------------------------------------------------------------------- #
# 1. save_template / load_template / __init__
# --------------------------------------------------------------------------- #

def test_save_load_template() -> None:
    template = _make_template()
    assert template.shape == (_TEMPLATE_FRAMES, _FEATURE_DIM)

    DTWAligner.save_template(template, _TEMPLATE_PATH)
    assert os.path.isfile(_TEMPLATE_PATH), "save_template 后文件应存在"

    aligner = DTWAligner(_TEMPLATE_PATH)
    loaded = aligner.load_template()
    assert isinstance(loaded, np.ndarray), "load_template 应返回 ndarray"
    assert loaded.shape == (_TEMPLATE_FRAMES, _FEATURE_DIM), (
        f"期望 shape (60, 10)，得到 {loaded.shape}"
    )
    # 内容也应一致
    assert np.allclose(loaded, template), "load_template 内容应与保存一致"

    # 文件不存在时抛 FileNotFoundError
    raised = False
    try:
        DTWAligner("/tmp/__definitely_not_a_template__.npy")
    except FileNotFoundError:
        raised = True
    assert raised, "模板路径不存在时应抛出 FileNotFoundError"

    print("test_save_load_template PASSED")


# --------------------------------------------------------------------------- #
# 2. segment_kick_cycles
# --------------------------------------------------------------------------- #

def test_segment_kick_cycles() -> None:
    aligner = DTWAligner(_TEMPLATE_PATH)
    features = _make_continuous_features(n_cycles=5, cycle_len=60)
    assert features.shape == (300, _FEATURE_DIM)

    cycles = aligner.segment_kick_cycles(features, angle_column=0)
    assert isinstance(cycles, list), "应返回 list"
    assert 4 <= len(cycles) <= 6, f"期望 4-6 个周期，得到 {len(cycles)}"
    for i, c in enumerate(cycles):
        assert isinstance(c, np.ndarray), f"第 {i} 个周期不是 ndarray"
        assert c.ndim == 2 and c.shape[1] == _FEATURE_DIM, (
            f"第 {i} 个周期 shape 异常: {c.shape}"
        )
        assert 10 <= c.shape[0] <= 120, (
            f"第 {i} 个周期帧数 {c.shape[0]} 不在 [10, 120] 之内"
        )

    print("test_segment_kick_cycles PASSED")


# --------------------------------------------------------------------------- #
# 3. align
# --------------------------------------------------------------------------- #

def test_align() -> None:
    aligner = DTWAligner(_TEMPLATE_PATH)
    template = aligner.load_template()

    # 自对齐：距离应接近 0
    aq, at, dist = aligner.align(template.copy())
    assert aq.shape == at.shape, (
        f"aligned_query 与 aligned_template shape 不一致: {aq.shape} vs {at.shape}"
    )
    assert aq.shape[1] == _FEATURE_DIM
    assert dist < 1.0, f"模板自对齐距离应接近 0，得到 {dist}"

    # 偏移版本：所有角度 +15 度，距离应明显大于 0
    shifted = template.copy()
    shifted[:, :4] += 15.0
    aq2, at2, dist2 = aligner.align(shifted)
    assert aq2.shape == at2.shape
    assert dist2 > 0.0, f"偏移版本距离应 > 0，得到 {dist2}"
    # 顺带验证偏移版本距离远大于自对齐
    assert dist2 > dist, "偏移版本距离应大于自对齐距离"

    print("test_align PASSED")


# --------------------------------------------------------------------------- #
# 4. compute_phase_deviations
# --------------------------------------------------------------------------- #

def test_compute_phase_deviations() -> None:
    aligner = DTWAligner(_TEMPLATE_PATH)
    template = aligner.load_template()
    shifted = template.copy()
    shifted[:, :4] += 15.0

    aq, at, _ = aligner.align(shifted)
    devs = aligner.compute_phase_deviations(aq, at)

    assert isinstance(devs, dict), "应返回 dict"
    assert set(devs.keys()) == {"recovery", "catch", "power", "glide"}, (
        f"key 不匹配: {set(devs.keys())}"
    )
    for name, v in devs.items():
        assert isinstance(v, float), f"{name} 应为 float，得到 {type(v)}"
        assert v >= 0.0, f"{name} 偏差应 >= 0，得到 {v}"

    print("test_compute_phase_deviations PASSED")


# --------------------------------------------------------------------------- #
# 5. compute_overall_deviation
# --------------------------------------------------------------------------- #

def test_compute_overall_deviation() -> None:
    aligner = DTWAligner(_TEMPLATE_PATH)

    # 全零偏差 -> 0
    zero = {"recovery": 0.0, "catch": 0.0, "power": 0.0, "glide": 0.0}
    assert aligner.compute_overall_deviation(zero) == 0.0, "全零偏差应返回 0"

    # 权重求和：0.15 + 0.20 + 0.40 + 0.25 = 1.0，故 10 全填应返回 10.0
    ten = {"recovery": 10.0, "catch": 10.0, "power": 10.0, "glide": 10.0}
    overall = aligner.compute_overall_deviation(ten)
    assert abs(overall - 10.0) < 1e-9, f"期望 10.0，得到 {overall}"

    print("test_compute_overall_deviation PASSED")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        test_save_load_template()
        test_segment_kick_cycles()
        test_align()
        test_compute_phase_deviations()
        test_compute_overall_deviation()
        print("ALL TESTS PASSED")
    finally:
        if os.path.isfile(_TEMPLATE_PATH):
            os.remove(_TEMPLATE_PATH)
