"""探查 SwimXYZ Side_underwater 序列：解析完整骨架，找蹬腿/腿张开明显的帧。

输出每帧的脚分离度/膝角，并把若干候选帧的骨架散点预览存到 outputs/_scratch/。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SX = ROOT / "data" / "swimxyz" / "Breaststroke" / "Side_underwater"
SCRATCH = ROOT / "outputs" / "_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)


def parse_full(path: Path):
    """解析 2D_cam.txt → 帧列表，每帧 {joint_name:(x,y,visible)}（含全部关节）。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(";")
    # 收集所有有 .x/.y 的关节名
    names = []
    cols = {}
    for i, h in enumerate(header):
        if h.endswith(".x"):
            base = h[:-2]
            yi = header.index(f"{base}.y") if f"{base}.y" in header else None
            if yi is not None:
                names.append(base)
                cols[base] = (i, yi)
    frames = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(";")
        d = {}
        for nm, (xi, yi) in cols.items():
            def val(k):
                if k >= len(f) or f[k].strip() == "":
                    return None
                try:
                    return float(f[k].replace(",", "."))
                except ValueError:
                    return None
            x, y = val(xi), val(yi)
            d[nm] = (x, y, x is not None and y is not None)
        frames.append(d)
    return names, frames


def knee_angle(d, hip, knee, ankle):
    try:
        h = np.array(d[hip][:2]); k = np.array(d[knee][:2]); a = np.array(d[ankle][:2])
    except Exception:
        return 180.0
    v1, v2 = h - k, a - k
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def main():
    seqs = sorted(SX.rglob("base/2D_cam.txt"))
    print(f"Side_underwater base 序列数: {len(seqs)}")
    if not seqs:
        return
    # 取前几个序列，找腿张开 / 膝弯曲最明显的帧
    best = []  # (score, seqidx, frameidx, metrics, frame, names)
    for si, sp in enumerate(seqs[:8]):
        names, frames = parse_full(sp)
        for fi, d in enumerate(frames):
            need = ["L Foot", "R Foot", "L Calf", "R Calf", "L Thigh", "R Thigh"]
            if not all(d.get(n, (None, None, False))[2] for n in need):
                continue
            lf, rf = np.array(d["L Foot"][:2]), np.array(d["R Foot"][:2])
            lk, rk = np.array(d["L Calf"][:2]), np.array(d["R Calf"][:2])
            lh, rh = np.array(d["L Thigh"][:2]), np.array(d["R Thigh"][:2])
            hipw = np.linalg.norm(lh - rh) + 1e-6
            foot_sep = np.linalg.norm(lf - rf) / hipw          # 脚分离（V 张开）
            knee_sep = np.linalg.norm(lk - rk) / hipw
            ka = (knee_angle(d, "L Thigh", "L Calf", "L Foot")
                  + knee_angle(d, "R Thigh", "R Calf", "R Foot")) / 2
            score = foot_sep + knee_sep                         # 越大越"张开"
            best.append((score, si, fi, (foot_sep, knee_sep, ka), d, names, str(sp)))
    best.sort(key=lambda r: -r[0])
    print("\n=== 脚/膝张开最大的 8 帧 ===")
    for score, si, fi, (fs, ks, ka), d, names, sp in best[:8]:
        print(f"seq{si} f{fi:3d} foot_sep={fs:.2f} knee_sep={ks:.2f} knee_ang={ka:5.1f}  {Path(sp).parent.parent.parent.name}")

    # 预览：取张开最大的 4 帧 + 膝最弯的 2 帧，画骨架散点
    def preview(rows, tag):
        for n, (score, si, fi, m, d, names, sp) in enumerate(rows):
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            xs = [d[k][0] for k in names if d[k][2]]
            ys = [d[k][1] for k in names if d[k][2]]
            ax.scatter(xs, ys, s=10, c="orange")
            for k in names:
                if d[k][2]:
                    ax.annotate(k, (d[k][0], d[k][1]), fontsize=4, color="white")
            ax.set_title(f"seq{si} f{fi} foot_sep={m[0]:.2f} knee_ang={m[2]:.0f}", fontsize=8)
            ax.invert_yaxis()  # 图像坐标 y 向下
            ax.set_aspect("equal"); ax.set_facecolor("#0a2a3a")
            fig.savefig(SCRATCH / f"sx_{tag}_{n}_s{si}f{fi}.png", bbox_inches="tight")
            plt.close(fig)
    preview(best[:4], "spread")
    most_bent = sorted(best, key=lambda r: r[3][2])[:2]
    preview(most_bent, "bent")
    print("预览已存 outputs/_scratch/sx_*.png")


if __name__ == "__main__":
    main()
