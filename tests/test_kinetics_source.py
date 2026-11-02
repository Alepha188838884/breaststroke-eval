"""``src.train_agent.build_kick_cycles_kinetics`` 的轻量级测试。

不依赖 GPU / ``data/`` 下的真实文件：用 ``tempfile`` 在临时目录里合成
``{video_id}_cycles.pkl`` / ``dataset_meta.json`` / 单周期模板 ``.npy``。
运行方式：``python tests/test_kinetics_source.py``
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

# 将项目根目录加入 sys.path，以便导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dtw_aligner import DTWAligner
from src.train_agent import build_kick_cycles_kinetics


_FEATURE_DIM = 10
_MIN_CYCLE_FRAMES = 10  # 与 dtw_aligner._MIN_CYCLE_FRAMES 一致


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


def _write_cycles_pkl(features_dir: str, video_id: str, lengths: list[int]) -> None:
    cycles = [_make_cycle(n, seed=i) for i, n in enumerate(lengths)]
    with open(os.path.join(features_dir, f"{video_id}_cycles.pkl"), "wb") as f:
        pickle.dump(cycles, f)


def _make_dataset(tmp_dir: str) -> tuple[str, str, str]:
    """构造 2 train + 1 test 视频（另加 1 个缺文件的 train 视频）+ meta + 模板。

    Returns:
        ``(features_dir, meta_path, template_path)``。
    """
    features_dir = os.path.join(tmp_dir, "features")
    os.makedirs(features_dir, exist_ok=True)

    # train: vid_a 有效周期长度 [20, 20]，1 个过短周期 [5]（应被过滤）
    _write_cycles_pkl(features_dir, "vid_a", [20, 20, 5])
    # train: vid_b 有效周期长度 [15, 12]
    _write_cycles_pkl(features_dir, "vid_b", [15, 12])
    # train: vid_missing 无对应 pkl 文件，测试缺文件跳过不崩
    # (故意不写文件)

    # test: vid_c 有效周期长度 [18]，1 个过短周期 [3]（应被过滤）
    _write_cycles_pkl(features_dir, "vid_c", [18, 3])

    meta = {
        "train_video_ids": ["vid_a", "vid_b", "vid_missing"],
        "test_video_ids": ["vid_c"],
    }
    meta_path = os.path.join(tmp_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # 单周期模板（帧数远小于 _MAX_CYCLE_FRAMES=120，_ensure_single_cycle_template 应为 no-op）
    template_path = os.path.join(tmp_dir, "template.npy")
    template = _make_cycle(30, seed=99)
    DTWAligner.save_template(template, template_path)

    return features_dir, meta_path, template_path


def test_build_kick_cycles_kinetics_counts_and_filtering() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="kinetics_test_")
    try:
        features_dir, meta_path, template_path = _make_dataset(tmp_dir)

        train_cycles, val_cycles, aligner = build_kick_cycles_kinetics(
            features_dir=features_dir,
            meta_path=meta_path,
            template_path=template_path,
        )

        # vid_a: 20,20 有效（5 被过滤）；vid_b: 15,12 有效；vid_missing: 0（跳过不崩）
        assert len(train_cycles) == 4, (
            f"train_cycles 应为 4（2 短周期被过滤，vid_missing 跳过），得到 {len(train_cycles)}"
        )
        # vid_c: 18 有效（3 被过滤）
        assert len(val_cycles) == 1, (
            f"val_cycles 应为 1（1 个短周期被过滤），得到 {len(val_cycles)}"
        )
        for c in train_cycles + val_cycles:
            assert c.shape[1] == _FEATURE_DIM
            assert c.shape[0] >= _MIN_CYCLE_FRAMES

        assert isinstance(aligner, DTWAligner)
        assert aligner.template.shape[0] == 30, "模板应保持原单周期长度（no-op）"

        print("test_build_kick_cycles_kinetics_counts_and_filtering PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_kick_cycles_kinetics_empty_train_raises() -> None:
    """train 侧全部视频缺文件/无有效周期时应 raise RuntimeError。"""
    tmp_dir = tempfile.mkdtemp(prefix="kinetics_test_empty_")
    try:
        features_dir = os.path.join(tmp_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        # test 侧提供有效数据，train 侧留空（全部视频缺文件）
        _write_cycles_pkl(features_dir, "vid_c", [18])
        meta = {"train_video_ids": ["vid_missing_only"], "test_video_ids": ["vid_c"]}
        meta_path = os.path.join(tmp_dir, "dataset_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        template_path = os.path.join(tmp_dir, "template.npy")
        DTWAligner.save_template(_make_cycle(30, seed=99), template_path)

        try:
            build_kick_cycles_kinetics(
                features_dir=features_dir,
                meta_path=meta_path,
                template_path=template_path,
            )
            assert False, "train 侧为空时应抛出 RuntimeError"
        except RuntimeError:
            pass
        print("test_build_kick_cycles_kinetics_empty_train_raises PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_build_kick_cycles_kinetics_counts_and_filtering()
    test_build_kick_cycles_kinetics_empty_train_raises()
    print("ALL TESTS PASSED")
