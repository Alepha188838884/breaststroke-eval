"""``src.train_agent._full_eval`` / ``FullEvalCallback`` 的轻量级测试。

背景：旧评估协议用 ``env.reset()`` 有放回随机抽样 n_episodes 次估计验证集
MAE，样本量小于抽样次数时单次实现方差可达 ±0.5~0.7 分且不可复现。新协议
``_full_eval`` 对 env 缓存的每个周期确定性枚举恰好一次，本文件验证：

    1. ``_full_eval`` 的统计量与"手算值"（对 env 自身缓存的 target_score 用
       固定动作直接算 MAE）一致，且 per_bin_mae / n_unique_predictions 正确。
    2. 同一模型两次调用 ``_full_eval`` 结果逐位相同（确定性、可复现）。
    3. ``FullEvalCallback._run_eval``：仅 MAE 改善时才（重新）保存 best_model.zip。

不依赖 GPU / 真实数据集 / 真实 PPO：用随机生成的蹬腿周期 + mock DTWAligner +
鸭子类型假模型（仅需 ``predict(obs, deterministic)`` / ``save(path)``）。
运行方式：``python tests/test_full_eval.py``
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dtw_aligner import _PHASE_NAMES, _PHASE_WEIGHTS
from src.kick_env import BreaststrokeKickEnv
from src.train_agent import FullEvalCallback, _full_eval


_CYCLE_LEN: int = 30
_FEATURE_DIM: int = 10
_N_CYCLES: int = 6


# --------------------------------------------------------------------------- #
# 与 tests/test_kick_env.py 相同套路的 mock（各测试文件自包含，避免跨文件依赖）
# --------------------------------------------------------------------------- #


class _MockDTWAligner:
    """伪 DTWAligner：免去 dtw-python 依赖，按确定性规则生成偏差。"""

    template: np.ndarray = np.zeros((_CYCLE_LEN, _FEATURE_DIM), dtype=np.float64)

    def align(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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
    def compute_overall_deviation(phase_deviations: dict[str, float]) -> float:
        total = 0.0
        for name, w in _PHASE_WEIGHTS.items():
            total += w * float(phase_deviations.get(name, 0.0))
        return total


def _make_random_cycles(
    n_cycles: int = _N_CYCLES, cycle_len: int = _CYCLE_LEN, seed: int = 0
) -> list[np.ndarray]:
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
    return BreaststrokeKickEnv(cycles, aligner, seed=seed)  # type: ignore[arg-type]


class _ConstantModel:
    """假模型：predict 恒返回动作 ``k``（鸭子类型，仅需 predict / save 两个方法）。"""

    def __init__(self, k: int) -> None:
        self.k = k

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        del obs, deterministic
        return np.array(self.k), None

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(str(self.k).encode("utf-8"))


# --------------------------------------------------------------------------- #
# 1. _full_eval 与手算值一致
# --------------------------------------------------------------------------- #


def test_full_eval_matches_hand_computed_mae() -> None:
    env = _make_env(seed=1)
    k = 7
    model = _ConstantModel(k)
    stats = _full_eval(model, env)

    targets = np.asarray(env._cached_target_score, dtype=np.float64)
    expected_pred = float(k * 5)
    expected_mae = float(np.mean(np.abs(expected_pred - targets)))
    expected_max_err = float(np.max(np.abs(expected_pred - targets)))

    assert abs(stats["mae"] - expected_mae) < 1e-9, (
        f"mae 与手算值不符: got={stats['mae']}, expected={expected_mae}"
    )
    assert abs(stats["max_abs_error"] - expected_max_err) < 1e-9
    assert abs(stats["mean_reward"] - (-expected_mae / 100.0)) < 1e-9, (
        "mean_reward 应等于 -mae/100（与 env reward 定义一致）"
    )
    assert stats["n_unique_predictions"] == 1, "恒定动作应只产生 1 种预测值"
    assert stats["pred_std"] == 0.0, "恒定动作预测标准差应为 0"

    expected_bins = set(int(t) for t in targets)
    assert set(stats["per_bin_mae"].keys()) == expected_bins, (
        f"per_bin_mae 键应与 target 档位集合一致: "
        f"got={set(stats['per_bin_mae'].keys())}, expected={expected_bins}"
    )
    for bin_val in expected_bins:
        bin_targets = targets[targets == bin_val]
        expected_bin_mae = float(np.mean(np.abs(expected_pred - bin_targets)))
        assert abs(stats["per_bin_mae"][bin_val] - expected_bin_mae) < 1e-9, (
            f"档位 {bin_val} 的 per_bin_mae 不符"
        )
    print("test_full_eval_matches_hand_computed_mae PASSED")


# --------------------------------------------------------------------------- #
# 2. 确定性：同一模型两次调用结果逐位相同
# --------------------------------------------------------------------------- #


def test_full_eval_is_deterministic() -> None:
    env = _make_env(seed=2)
    model = _ConstantModel(k=3)
    stats_a = _full_eval(model, env)
    stats_b = _full_eval(model, env)
    assert stats_a["mae"] == stats_b["mae"]
    assert stats_a["max_abs_error"] == stats_b["max_abs_error"]
    assert stats_a["mean_reward"] == stats_b["mean_reward"]
    assert stats_a["n_unique_predictions"] == stats_b["n_unique_predictions"]
    assert stats_a["pred_std"] == stats_b["pred_std"]
    assert stats_a["per_bin_mae"] == stats_b["per_bin_mae"]
    print("test_full_eval_is_deterministic PASSED")


# --------------------------------------------------------------------------- #
# 3. FullEvalCallback._run_eval：仅 MAE 改善才保存 best_model.zip
# --------------------------------------------------------------------------- #


def test_full_eval_callback_saves_best_model_only_on_improvement(tmp_path) -> None:
    val_env = _make_env(seed=3)
    save_dir = str(tmp_path / "best")
    callback = FullEvalCallback(
        val_env, eval_freq=100, best_model_save_path=save_dir, verbose=0
    )

    # 用一组动作档位算出各自的全量 MAE，挑出明显更差 / 更好的一对，
    # 不依赖具体数值假设（数据驱动，避免因随机周期分布不同而失效）。
    candidates = [_ConstantModel(k) for k in (0, 4, 10, 16, 20)]
    maes = [_full_eval(m, val_env)["mae"] for m in candidates]
    worse_idx = int(np.argmax(maes))
    better_idx = int(np.argmin(maes))
    assert maes[worse_idx] > maes[better_idx], (
        "候选 MAE 应有区分度；若本断言失败，说明测试数据退化，请更换构造种子"
    )
    worse_model, better_model = candidates[worse_idx], candidates[better_idx]

    # Step 1: 较差模型先评估，建立初始 best
    callback.model = worse_model
    result_1 = callback._run_eval()
    assert abs(result_1["mae"] - maes[worse_idx]) < 1e-9
    best_path = os.path.join(save_dir, "best_model.zip")
    assert os.path.isfile(best_path), "首次评估应无条件建立 best_model"
    assert callback.best_mae == maes[worse_idx]
    mtime_1 = os.path.getmtime(best_path)

    # Step 2: 同一（较差）模型重复评估，MAE 持平，不应重新保存
    callback.model = worse_model
    callback._run_eval()
    assert callback.best_mae == maes[worse_idx]
    assert os.path.getmtime(best_path) == mtime_1, "MAE 未改善（持平）不应重新保存"

    # Step 3: 更优模型评估，MAE 改善，应覆盖保存
    time.sleep(0.01)  # 确保 mtime 有可观测差异
    callback.model = better_model
    callback._run_eval()
    assert callback.best_mae == maes[better_idx]
    mtime_3 = os.path.getmtime(best_path)
    assert mtime_3 > mtime_1, "MAE 改善后应重新保存 best_model"

    # evaluations 记录了 3 次评估
    assert len(callback.evaluations) == 3

    # _on_training_end 应把 evaluations 落盘为 evaluations.npz
    callback._on_training_end()
    npz_path = os.path.join(save_dir, "evaluations.npz")
    assert os.path.isfile(npz_path), "_on_training_end 应保存 evaluations.npz"
    data = np.load(npz_path)
    assert len(data["maes"]) == 3
    print("test_full_eval_callback_saves_best_model_only_on_improvement PASSED")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_full_eval_matches_hand_computed_mae()
    test_full_eval_is_deterministic()
    with tempfile.TemporaryDirectory() as td:
        test_full_eval_callback_saves_best_model_only_on_improvement(Path(td))
    print("ALL TESTS PASSED")
