"""模块 6（FeedbackGenerator）+ DTWAligner 有符号偏差 + pipeline 打分编排的测试。

不依赖 GPU / mmpose / 真实数据集，用合成数据 + stub 模型验证。
运行方式：python tests/test_module_6.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dtw_aligner import DTWAligner
from src.feedback import FeedbackGenerator

_FEATURE_DIM = 10
_ANGLE_CENTER = 115.0
_ANGLE_AMPL = 55.0


def _make_template(n_frames: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    feat = np.zeros((n_frames, _FEATURE_DIM), dtype=np.float64)
    for i in range(4):
        feat[:, i] = _ANGLE_CENTER - _ANGLE_AMPL * np.cos(t + i * np.pi / 8)
    feat[:, 4:] = rng.standard_normal((n_frames, _FEATURE_DIM - 4))
    return feat


def _make_continuous_features(n_cycles=5, cycle_len=60, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = n_cycles * cycle_len
    t = np.linspace(0.0, n_cycles * 2.0 * np.pi, total, endpoint=False)
    feat = np.zeros((total, _FEATURE_DIM), dtype=np.float64)
    feat[:, 0] = _ANGLE_CENTER + _ANGLE_AMPL * np.cos(t)
    feat[:, 1:4] = _ANGLE_CENTER + 10.0 * rng.standard_normal((total, 3))
    feat[:, 4:] = rng.standard_normal((total, _FEATURE_DIM - 4))
    return feat


# --------------------------------------------------------------------------- #
# 1. DTWAligner.compute_signed_phase_feature_deviations
# --------------------------------------------------------------------------- #

def test_signed_phase_feature_deviations() -> None:
    tpl_path = os.path.join(tempfile.mkdtemp(), "tpl.npy")
    template = _make_template()
    DTWAligner.save_template(template, tpl_path)
    aligner = DTWAligner(tpl_path)

    # 该方法消费**已对齐**序列：直接喂等长受控输入，精确验证有符号均值。
    # （不经 align()——DTW 会把恒定偏移 warp 吸收掉，那是 DTW 行为而非本方法契约。）
    at = template.copy()
    aq_up = template.copy()
    aq_up[:, :4] += 15.0
    devs = aligner.compute_signed_phase_feature_deviations(aq_up, at)
    assert set(devs.keys()) == {"recovery", "catch", "power", "glide"}
    for name, vec in devs.items():
        assert vec.shape == (4,), f"{name} 偏差向量 shape 应为 (4,)，得到 {vec.shape}"
        assert np.allclose(vec, 15.0, atol=1e-6), f"{name} 偏差应=+15，得到 {vec}"

    aq_dn = template.copy()
    aq_dn[:, :4] -= 20.0
    devs2 = aligner.compute_signed_phase_feature_deviations(aq_dn, at)
    for name, vec in devs2.items():
        assert np.allclose(vec, -20.0, atol=1e-6), f"{name} 偏差应=-20，得到 {vec}"

    # 单关节定向偏差：只抬高 power 段的左膝列，其它列/阶段应≈0
    at2 = template.copy()
    aq3 = template.copy()
    l = aq3.shape[0]
    aq3[l // 2 : (3 * l) // 4, 0] += 30.0  # power 段(50%-75%) 左膝 +30
    d3 = aligner.compute_signed_phase_feature_deviations(aq3, at2)
    assert d3["power"][0] > 25.0, f"power 左膝应显著为正，得到 {d3['power'][0]}"
    assert abs(d3["recovery"][0]) < 1e-6, "recovery 左膝不应受影响"
    assert abs(d3["power"][1]) < 1e-6, "power 右膝不应受影响"

    # 经 align() 的端到端：符号应被保留（幅度可被 DTW 压缩）
    aq_e, at_e, _ = aligner.align(aq_up)
    de = aligner.compute_signed_phase_feature_deviations(aq_e, at_e)
    for name, vec in de.items():
        assert vec.mean() > 0, f"{name} 上移后经 align 均值仍应为正，得到 {vec}"

    # 形状不一致应抛 ValueError
    raised = False
    try:
        aligner.compute_signed_phase_feature_deviations(
            np.zeros((5, 10)), np.zeros((6, 10))
        )
    except ValueError:
        raised = True
    assert raised, "形状不一致应抛 ValueError"

    print("test_signed_phase_feature_deviations PASSED")


# --------------------------------------------------------------------------- #
# 2. FeedbackGenerator
# --------------------------------------------------------------------------- #

def test_feedback_direction_and_threshold() -> None:
    gen = FeedbackGenerator(deviation_threshold_deg=10.0, max_items=5)

    # 无超阈偏差 + 高分 → 正面反馈
    tiny = {p: np.zeros(4) for p in ("recovery", "catch", "power", "glide")}
    out = gen.generate(score=90.0, signed_phase_deviations=tiny)
    assert len(out) == 1 and "no major corrections" in out[0].lower(), out

    # 无超阈偏差 + 低分 → 中性反馈
    out = gen.generate(score=40.0, signed_phase_deviations=tiny)
    assert len(out) == 1 and "threshold" in out[0].lower(), out

    # power 阶段左膝 -30°（相对模板更屈）→ 阶段中立措辞 "more flexed than reference"
    devs = {p: np.zeros(4) for p in ("recovery", "catch", "power", "glide")}
    devs["power"][0] = -30.0
    out = gen.generate(score=50.0, signed_phase_deviations=devs)
    assert any("more flexed than reference" in m.lower() for m in out), out
    assert any("left knee" in m.lower() for m in out), out
    assert any("power phase" in m.lower() for m in out), out

    # 正偏差 → more extended than reference
    devs2 = {p: np.zeros(4) for p in ("recovery", "catch", "power", "glide")}
    devs2["glide"][2] = 25.0  # left hip 相对模板更伸展
    out2 = gen.generate(score=50.0, signed_phase_deviations=devs2)
    assert any("more extended than reference" in m.lower() for m in out2), out2
    assert any("left hip" in m.lower() for m in out2), out2

    print("test_feedback_direction_and_threshold PASSED")


def test_feedback_severity_ordering_and_cap() -> None:
    # power(权重0.40)的小偏差 vs recovery(权重0.15)的大偏差：用严重度=|d|×权重排序
    gen = FeedbackGenerator(deviation_threshold_deg=10.0, max_items=2)
    devs = {p: np.zeros(4) for p in ("recovery", "catch", "power", "glide")}
    devs["power"][0] = -30.0      # severity = 30*0.40 = 12.0 → 第一
    devs["recovery"][1] = 40.0    # severity = 40*0.15 = 6.0  → 第二
    devs["glide"][3] = 11.0       # severity = 11*0.25 = 2.75 → 被 max_items=2 截断
    out = gen.generate(score=50.0, signed_phase_deviations=devs)
    assert len(out) == 2, f"max_items=2 应截断到 2 条，得到 {len(out)}"
    assert "power phase" in out[0].lower(), f"power 应排第一，得到 {out}"

    # 非法参数
    for bad in (lambda: FeedbackGenerator(deviation_threshold_deg=-1.0),
                lambda: FeedbackGenerator(max_items=0)):
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        assert raised, "非法构造参数应抛 ValueError"

    # 偏差向量长度非 4 应抛
    raised = False
    try:
        gen.generate(50.0, {"power": np.zeros(3)})
    except ValueError:
        raised = True
    assert raised, "偏差向量长度非 4 应抛 ValueError"

    print("test_feedback_severity_ordering_and_cap PASSED")


# --------------------------------------------------------------------------- #
# 3. pipeline.score_features（stub 模型，免训练）
# --------------------------------------------------------------------------- #

class _StubModel:
    """固定返回 action=15（→ 75 分）的假模型，仅验证编排逻辑。"""

    def predict(self, obs, deterministic=True):
        return np.int64(15), None


def test_pipeline_score_features() -> None:
    from src.pipeline import EvaluationPipeline

    work = tempfile.mkdtemp()
    tpl_path = os.path.join(work, "champion_template.npy")
    DTWAligner.save_template(_make_template(60), tpl_path)

    cfg_path = os.path.join(work, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(
            "paths:\n"
            f"  template: {tpl_path}\n"
            f"  model_output: {os.path.join(work, 'model.zip')}\n"
            "video:\n  fps: 30\n"
            "pose:\n  min_confidence: 0.3\n"
            "feedback:\n  deviation_threshold_deg: 10.0\n  max_items: 5\n"
        )

    pipe = EvaluationPipeline(config_path=cfg_path)
    pipe._model = _StubModel()  # 注入 stub，跳过真实模型加载

    feats = _make_continuous_features(n_cycles=5, cycle_len=60)
    result = pipe.score_features(feats)

    assert result["score"] == 75, f"stub action=15 → 总分应为 75，得到 {result['score']}"
    assert result["n_cycles"] >= 1
    assert len(result["cycles"]) == result["n_cycles"]
    c0 = result["cycles"][0]
    for key in ("cycle_index", "agent_score", "reference_score",
                "phase_deviations", "feedback"):
        assert key in c0, f"per-cycle 缺少键 {key}"
    assert c0["agent_score"] == 75.0
    assert isinstance(c0["feedback"], list)

    # 无周期可分割应抛 ValueError
    raised = False
    try:
        pipe.score_features(np.full((40, 10), 115.0))  # 平直信号，无波谷
    except ValueError:
        raised = True
    assert raised, "无可分割周期应抛 ValueError"

    print("test_pipeline_score_features PASSED")


# --------------------------------------------------------------------------- #
# 4. timing 因素：匀速快/慢应≈0（被 DTW 吸收），不饱和
# --------------------------------------------------------------------------- #

def test_timing_factor_uniform_tempo() -> None:
    """同形状、仅帧数不同的周期(匀速)对齐模板后 timing(warp) 应接近 0。

    回归保护：旧公式 (path-max)/max 会让正常对齐也顶到上限而退化为常数。
    """
    from src.kick_env import BreaststrokeKickEnv

    work = tempfile.mkdtemp()
    tpl_path = os.path.join(work, "tpl.npy")
    DTWAligner.save_template(_make_template(45), tpl_path)
    aligner = DTWAligner(tpl_path)

    # 与模板同形状、不同帧数(30/40/50)的匀速周期
    cycles = []
    for n in (30, 40, 50):
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        c = np.zeros((n, _FEATURE_DIM), dtype=np.float64)
        for i in range(4):
            c[:, i] = _ANGLE_CENTER - _ANGLE_AMPL * np.cos(t + i * np.pi / 8)
        cycles.append(c)

    env = BreaststrokeKickEnv(cycles, aligner, seed=0)
    dist = env.factor_distribution()
    warp_p90 = dist["timing_warp"]["p90"]
    assert warp_p90 < 0.25, (
        f"匀速周期 timing(warp) 应≈0，但 p90={warp_p90:.3f}（疑似 timing 度量饱和）"
    )
    print("test_timing_factor_uniform_tempo PASSED")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_signed_phase_feature_deviations()
    test_feedback_direction_and_threshold()
    test_feedback_severity_ordering_and_cap()
    test_pipeline_score_features()
    test_timing_factor_uniform_tempo()
    print("ALL TESTS PASSED")
