"""SwimXYZLoader 测试：真实 SwimXYZ ``.txt`` 格式解析（base 骨架）。

覆盖：skeleton/view 发现过滤、分号分隔 + 逗号小数解析、按列名做关节映射、
缺失关节（空字段）→ 置信度 0、mid_shoulder 计算、缺列时抛 ValueError。
运行方式：python tests/test_swimxyz_loader.py
"""

from __future__ import annotations

import os
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.swimxyz_loader import SwimXYZLoader

# 需要的 8 个 base 骨架关节（+ Nose 作无关额外列，测试按名查找不依赖列序）
_JOINTS = [
    "Nose",
    "L UpperArm", "R UpperArm",
    "L Thigh", "R Thigh",
    "L Calf", "R Calf",
    "L Foot", "R Foot",
]

# 每个关节一组合理像素坐标 (x, y, z=depth)
_COORDS = {
    "Nose": (100.0, 50.0, 5.0),
    "L UpperArm": (90.0, 100.0, 5.0), "R UpperArm": (110.0, 100.0, 5.0),
    "L Thigh": (90.0, 200.0, 5.0), "R Thigh": (110.0, 200.0, 5.0),
    "L Calf": (90.0, 300.0, 5.0), "R Calf": (110.0, 300.0, 5.0),
    "L Foot": (90.0, 400.0, 5.0), "R Foot": (110.0, 400.0, 5.0),
}


def _fmt(v: float) -> str:
    """逗号小数（欧洲制），如 90.0 → '90,00'。"""
    return f"{v:.2f}".replace(".", ",")


def _make_txt(path: str, drop_joint: str | None = None, n_frames: int = 3) -> None:
    """写一个 SwimXYZ 风格 .txt：分号分隔、逗号小数、末尾分号；某关节可设为缺失(空字段)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ";".join(f"{j}.{c}" for j in _JOINTS for c in "xyz") + ";"
    lines = [header]
    for _ in range(n_frames):
        cells = []
        for j in _JOINTS:
            if j == drop_joint:
                cells += ["", "", ""]  # 缺失关节 → 空字段
            else:
                cells += [_fmt(c) for c in _COORDS[j]]
        lines.append(";".join(cells) + ";")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _leaf(root: str, view: str) -> str:
    return os.path.join(
        root, "Breaststroke", view, "Swimmer_0", "Water_0", "Light_0",
        "Speed_2", "position_1", "base", "2D_cam.txt",
    )


def test_discovery_and_view_filter() -> None:
    root = tempfile.mkdtemp()
    _make_txt(_leaf(root, "Side_underwater"))
    _make_txt(_leaf(root, "Aerial"))  # 应被默认 view 过滤掉

    ld = SwimXYZLoader(os.path.join(root, "Breaststroke"))  # 默认 base + Side_underwater
    seqs = ld.list_sequences()
    assert len(seqs) == 1, f"默认 view 应只发现 1 个 Side_underwater 序列，得到 {len(seqs)}"
    assert "/Side_underwater/" in seqs[0]

    # view=None → 两个都发现
    ld_all = SwimXYZLoader(os.path.join(root, "Breaststroke"), view=None)
    assert len(ld_all.list_sequences()) == 2, "view=None 应发现全部视角"
    print("test_discovery_and_view_filter PASSED")


def test_parse_and_joint_mapping() -> None:
    root = tempfile.mkdtemp()
    _make_txt(_leaf(root, "Side_underwater"), n_frames=4)
    ld = SwimXYZLoader(os.path.join(root, "Breaststroke"))
    seq = ld.load_sequence(ld.list_sequences()[0])

    assert len(seq) == 4, f"应解析 4 帧，得到 {len(seq)}"
    j0 = seq[0]
    expected = {
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "mid_shoulder",
    }
    assert set(j0.keys()) == expected, f"关节键不匹配: {set(j0.keys())}"

    # 映射正确性：L Calf → left_knee，坐标应为 (90, 300)，逗号小数被正确解析
    assert j0["left_knee"][:2] == (90.0, 300.0), f"left_knee 坐标={j0['left_knee']}"
    assert j0["left_knee"][2] == 1.0, "可见关节置信度应为 1.0"
    # mid_shoulder = 左右肩(UpperArm)中点 = ((90+110)/2, 100)
    assert j0["mid_shoulder"][:2] == (100.0, 100.0), f"mid_shoulder={j0['mid_shoulder']}"
    print("test_parse_and_joint_mapping PASSED")


def test_missing_joint_zero_confidence() -> None:
    root = tempfile.mkdtemp()
    _make_txt(_leaf(root, "Side_underwater"), drop_joint="L Foot", n_frames=2)
    ld = SwimXYZLoader(os.path.join(root, "Breaststroke"))
    seq = ld.load_sequence(ld.list_sequences()[0])
    # L Foot 缺失 → left_ankle 置信度 0（下游 FeatureExtractor 会据此丢帧）
    assert seq[0]["left_ankle"][2] == 0.0, "缺失关节置信度应为 0.0"
    assert seq[0]["left_knee"][2] == 1.0, "其它在场关节置信度应为 1.0"
    print("test_missing_joint_zero_confidence PASSED")


def test_missing_column_raises() -> None:
    root = tempfile.mkdtemp()
    path = _leaf(root, "Side_underwater")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 写一个缺少 "L Calf" 列的表头 → 应在 load 时抛 ValueError
    bad_joints = [j for j in _JOINTS if j != "L Calf"]
    header = ";".join(f"{j}.{c}" for j in bad_joints for c in "xyz") + ";"
    row = ";".join(_fmt(_COORDS[j][k]) for j in bad_joints for k in range(3)) + ";"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + row)

    ld = SwimXYZLoader(os.path.join(root, "Breaststroke"))
    raised = False
    try:
        ld.load_sequence(ld.list_sequences()[0])
    except ValueError:
        raised = True
    assert raised, "缺少所需关节列时应抛 ValueError"
    print("test_missing_column_raises PASSED")


if __name__ == "__main__":
    test_discovery_and_view_filter()
    test_parse_and_joint_mapping()
    test_missing_joint_zero_confidence()
    test_missing_column_raises()
    print("ALL TESTS PASSED")
