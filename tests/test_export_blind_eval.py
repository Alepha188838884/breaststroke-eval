"""``scripts/export_blind_eval.py`` 的轻量级测试：agent_score 在模型缺失时留空、脚本不崩，
以及盲评拆分导出 / join 比对 / 旧格式兼容比对。

不依赖真实 PPO 模型（``--model`` 指向的路径可以不存在）：直接调用模块内部函数
``_resolve_model_path`` / ``_load_model`` / ``score_video`` / ``export`` / ``compare``，
用真实 :class:`DTWAligner` + 合成蹬腿周期构造最小场景。
运行方式：``python tests/test_export_blind_eval.py`` 或 ``pytest tests/test_export_blind_eval.py``
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPTS_DIR not in sys.path:
    # export_blind_eval.py 以 `from common import ...` 方式导入同目录的 common.py，
    # 需要 scripts/ 在 sys.path 上（脚本独立运行时由 sys.path[0] 自动提供）。
    sys.path.insert(0, _SCRIPTS_DIR)

from src.dtw_aligner import DTWAligner  # noqa: E402

import export_blind_eval as eb  # noqa: E402


_FEATURE_DIM: int = 10


def _make_cycle(n_frames: int, seed: int) -> np.ndarray:
    """生成一个 ``(n_frames, 10)`` 的合成蹬腿周期：前 4 列为正弦角度，其余随机。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    feat = np.empty((n_frames, _FEATURE_DIM), dtype=np.float64)
    for i in range(4):
        feat[:, i] = 115.0 - 55.0 * np.cos(t + i * np.pi / 8)
    feat[:, 4:6] = rng.uniform(0.0, 10.0, size=(n_frames, 2))
    feat[:, 6:10] = rng.uniform(-100.0, 100.0, size=(n_frames, 4))
    return feat


# --------------------------------------------------------------------------- #
# 1. --model 未给出时按 config 解析；显式给出时优先于 config
# --------------------------------------------------------------------------- #


def test_resolve_model_path_prefers_explicit_arg() -> None:
    resolved = eb._resolve_model_path("explicit.zip", "configs/config.yaml")
    assert resolved == "explicit.zip", "显式 --model 应优先于 config"
    print("test_resolve_model_path_prefers_explicit_arg PASSED")


def test_resolve_model_path_falls_back_to_config() -> None:
    resolved = eb._resolve_model_path(None, os.path.join(_PROJECT_ROOT, "configs/config.yaml"))
    assert resolved, "未给 --model 时应从 config paths.model_output 解析出非空路径"
    print("test_resolve_model_path_falls_back_to_config PASSED")


# --------------------------------------------------------------------------- #
# 2. 模型文件不存在 -> _load_model 告警并返回 None（脚本不崩）
# --------------------------------------------------------------------------- #


def test_load_model_missing_file_returns_none() -> None:
    model = eb._load_model("/nonexistent/path/model_xyz_does_not_exist.zip")
    assert model is None, "模型文件不存在时 _load_model 应返回 None 而非抛异常"
    print("test_load_model_missing_file_returns_none PASSED")


# --------------------------------------------------------------------------- #
# 3. score_video(model=None) -> agent_score 为 None，system_score 仍正常计算
# --------------------------------------------------------------------------- #


def test_score_video_agent_score_none_when_model_missing() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_")
    try:
        features_dir = os.path.join(tmp_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        cycles = [_make_cycle(20, seed=i) for i in range(3)]
        with open(os.path.join(features_dir, "vid_x_cycles.pkl"), "wb") as f:
            pickle.dump(cycles, f)

        template_path = os.path.join(tmp_dir, "template.npy")
        DTWAligner.save_template(_make_cycle(30, seed=99), template_path)
        aligner = DTWAligner(template_path)

        result = eb.score_video(
            "vid_x", Path(features_dir), aligner, score_scales=None, model=None
        )
        assert result is not None, "有效周期应能成功评分"
        assert result["agent_score"] is None, "model=None 时 agent_score 必须为 None"
        assert result["n_cycles"] == 3
        assert 0.0 <= result["system_score"] <= 100.0
        print("test_score_video_agent_score_none_when_model_missing PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 4. score_video(model=假模型) -> agent_score 为该假模型的确定性蒸馏分均值
# --------------------------------------------------------------------------- #


class _ConstantModel:
    """鸭子类型假模型：predict 恒返回动作 k。"""

    def __init__(self, k: int) -> None:
        self.k = k

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        del obs, deterministic
        return np.array(self.k), None


def test_score_video_agent_score_present_when_model_given() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_model_")
    try:
        features_dir = os.path.join(tmp_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        cycles = [_make_cycle(20, seed=i) for i in range(3)]
        with open(os.path.join(features_dir, "vid_y_cycles.pkl"), "wb") as f:
            pickle.dump(cycles, f)

        template_path = os.path.join(tmp_dir, "template.npy")
        DTWAligner.save_template(_make_cycle(30, seed=99), template_path)
        aligner = DTWAligner(template_path)

        model = _ConstantModel(k=8)
        result = eb.score_video(
            "vid_y", Path(features_dir), aligner, score_scales=None, model=model
        )
        assert result is not None
        assert result["agent_score"] == 40.0, (
            f"恒定动作 8 应映射到 agent_score=40，得到 {result['agent_score']}"
        )
        print("test_score_video_agent_score_present_when_model_given PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 5. export() 产出「盲版拆分」：coach_sheet.csv 不含系统分，answer_key.csv 含系统分，
#    coach_rating.html 含每个 video_id 与 clips/ 引用及导出按钮的 JS。
# --------------------------------------------------------------------------- #


def _make_export_scenario(tmp_dir: str, video_ids: list[str]) -> dict[str, str]:
    """在 ``tmp_dir`` 下搭建 export() 所需的 dataset_meta/features/template，返回路径字典。"""
    features_dir = os.path.join(tmp_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    for i, vid in enumerate(video_ids):
        cycles = [_make_cycle(20, seed=i * 10 + j) for j in range(2)]
        with open(os.path.join(features_dir, f"{vid}_cycles.pkl"), "wb") as f:
            pickle.dump(cycles, f)

    template_path = os.path.join(tmp_dir, "template.npy")
    DTWAligner.save_template(_make_cycle(30, seed=99), template_path)

    dataset_meta_path = os.path.join(tmp_dir, "dataset_meta.json")
    with open(dataset_meta_path, "w", encoding="utf-8") as f:
        json.dump({"test_video_ids": video_ids}, f)

    return {
        "features_dir": features_dir,
        "template": template_path,
        "dataset_meta": dataset_meta_path,
    }


def test_export_blind_split_no_system_scores_in_coach_sheet() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_split_")
    try:
        video_ids = ["vidA_0_10", "vidB_5_15", "vidC_20_30"]
        paths = _make_export_scenario(tmp_dir, video_ids)
        out_dir = os.path.join(tmp_dir, "blind_eval")

        args = argparse.Namespace(
            n=None,
            seed=42,
            config=os.path.join(tmp_dir, "nonexistent_config.yaml"),
            model="/nonexistent/model_xyz.zip",
            dataset_meta=paths["dataset_meta"],
            features_dir=paths["features_dir"],
            template=paths["template"],
            out=out_dir,
        )
        eb.export(args)

        coach_sheet_path = Path(out_dir) / "coach_sheet.csv"
        answer_key_path = Path(out_dir) / "answer_key.csv"
        html_path = Path(out_dir) / "coach_rating.html"
        assert coach_sheet_path.is_file()
        assert answer_key_path.is_file()
        assert html_path.is_file()

        with open(coach_sheet_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            coach_fields = set(reader.fieldnames or [])
            coach_rows = list(reader)
        assert coach_fields == {"video_id", "n_cycles", "coach_score", "notes"}
        assert "system_score" not in coach_fields
        assert "agent_score" not in coach_fields
        assert len(coach_rows) == len(video_ids)
        for row in coach_rows:
            assert row["coach_score"] == "", "coach_score 必须留空"

        with open(answer_key_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            answer_fields = set(reader.fieldnames or [])
            answer_rows = list(reader)
        assert {"video_id", "n_cycles", "system_score", "agent_score"} == answer_fields
        for row in answer_rows:
            assert row["system_score"] != ""

        coach_ids = {r["video_id"] for r in coach_rows}
        answer_ids = {r["video_id"] for r in answer_rows}
        assert coach_ids == answer_ids == set(video_ids)

        html = html_path.read_text(encoding="utf-8")
        for vid in video_ids:
            assert vid in html
            assert f"clips/{vid}.mp4" in html
        # answer_key 中的系统分数值不应出现在 HTML 里
        for row in answer_rows:
            assert row["system_score"] not in html
        assert "exportCsv" in html and "filled_coach_sheet.csv" in html
        print("test_export_blind_split_no_system_scores_in_coach_sheet PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 6. compare() 新格式：join answer_key.csv + filled coach_sheet.csv
# --------------------------------------------------------------------------- #


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compare_join_answer_key_and_coach_sheet(capsys) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_compare_")
    try:
        answer_key_path = os.path.join(tmp_dir, "answer_key.csv")
        _write_csv(
            answer_key_path,
            ["video_id", "n_cycles", "system_score", "agent_score"],
            [
                {"video_id": "v1", "n_cycles": 3, "system_score": "80.0", "agent_score": "75.0"},
                {"video_id": "v2", "n_cycles": 2, "system_score": "60.0", "agent_score": "55.0"},
                {"video_id": "v3", "n_cycles": 4, "system_score": "90.0", "agent_score": "85.0"},
                {"video_id": "v_no_coach", "n_cycles": 2, "system_score": "70.0", "agent_score": "65.0"},
            ],
        )
        filled_coach_sheet_path = os.path.join(tmp_dir, "filled_coach_sheet.csv")
        _write_csv(
            filled_coach_sheet_path,
            ["video_id", "n_cycles", "coach_score", "notes"],
            [
                {"video_id": "v1", "n_cycles": 3, "coach_score": "78", "notes": ""},
                {"video_id": "v2", "n_cycles": 2, "coach_score": "", "notes": "unsure"},
                {"video_id": "v3", "n_cycles": 4, "coach_score": "88", "notes": ""},
                {"video_id": "v_not_in_answer_key", "n_cycles": 1, "coach_score": "50", "notes": ""},
            ],
        )

        args = argparse.Namespace(
            compare=filled_coach_sheet_path, answer_key=answer_key_path
        )
        eb.compare(args)

        captured = capsys.readouterr()
        # 只对 v1/v3（两表都有且 coach_score 非空）计算 -> n=2
        assert "n=2" in captured.out
        assert "system_score vs coach" in captured.out
        assert "agent_score vs coach" in captured.out
        print("test_compare_join_answer_key_and_coach_sheet PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_compare_join_small_sample_triggers_power_warning(capsys) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_power_")
    try:
        n_rows = 5
        answer_rows = [
            {
                "video_id": f"v{i}",
                "n_cycles": 2,
                "system_score": f"{50.0 + i * 5}",
                "agent_score": f"{48.0 + i * 5}",
            }
            for i in range(n_rows)
        ]
        coach_rows = [
            {"video_id": f"v{i}", "n_cycles": 2, "coach_score": f"{52.0 + i * 4}", "notes": ""}
            for i in range(n_rows)
        ]
        answer_key_path = os.path.join(tmp_dir, "answer_key.csv")
        _write_csv(
            answer_key_path, ["video_id", "n_cycles", "system_score", "agent_score"], answer_rows
        )
        filled_coach_sheet_path = os.path.join(tmp_dir, "filled_coach_sheet.csv")
        _write_csv(
            filled_coach_sheet_path, ["video_id", "n_cycles", "coach_score", "notes"], coach_rows
        )

        args = argparse.Namespace(
            compare=filled_coach_sheet_path, answer_key=answer_key_path
        )
        eb.compare(args)  # 不应崩

        captured = capsys.readouterr()
        assert f"n={n_rows}" in captured.out
        assert "功效" in captured.out and "警告" in captured.out
        print("test_compare_join_small_sample_triggers_power_warning PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 7. compare() 向后兼容：旧版单文件格式（同一 CSV 含 system/agent/coach）
# --------------------------------------------------------------------------- #


def test_compare_legacy_single_csv_still_works(capsys) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="export_blind_eval_test_legacy_")
    try:
        legacy_csv_path = os.path.join(tmp_dir, "legacy_filled.csv")
        _write_csv(
            legacy_csv_path,
            ["video_id", "n_cycles", "system_score", "agent_score", "coach_score"],
            [
                {"video_id": "v1", "n_cycles": 3, "system_score": "80.0", "agent_score": "75.0", "coach_score": "78"},
                {"video_id": "v2", "n_cycles": 2, "system_score": "60.0", "agent_score": "55.0", "coach_score": "58"},
                {"video_id": "v3", "n_cycles": 4, "system_score": "90.0", "agent_score": "85.0", "coach_score": "88"},
            ],
        )

        args = argparse.Namespace(compare=legacy_csv_path, answer_key="/nonexistent/answer_key.csv")
        eb.compare(args)  # 走兼容路径，不应要求 --answer-key 存在

        captured = capsys.readouterr()
        assert "n=3" in captured.out
        assert "system_score vs coach" in captured.out
        print("test_compare_legacy_single_csv_still_works PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_resolve_model_path_prefers_explicit_arg()
    test_resolve_model_path_falls_back_to_config()
    test_load_model_missing_file_returns_none()
    test_score_video_agent_score_none_when_model_missing()
    test_score_video_agent_score_present_when_model_given()
    test_export_blind_split_no_system_scores_in_coach_sheet()
    print("ALL TESTS PASSED (run with pytest for the capsys-based tests)")
