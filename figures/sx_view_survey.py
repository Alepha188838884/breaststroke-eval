"""survey: 各视角下挑一个屈膝帧，画骨架，看哪个视角的泳者侧影最像人。"""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SX = ROOT / "data" / "swimxyz" / "Breaststroke"
SCRATCH = ROOT / "outputs" / "_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

SX_MAP = {
    "head": "Head", "neck": "Neck",
    "l_shoulder": "L UpperArm", "r_shoulder": "R UpperArm",
    "l_elbow": "L Forearm", "r_elbow": "R Forearm",
    "l_wrist": "L Hand", "r_wrist": "R Hand",
    "l_hip": "L Thigh", "r_hip": "R Thigh",
    "l_knee": "L Calf", "r_knee": "R Calf",
    "l_ankle": "L Foot", "r_ankle": "R Foot",
}
BONES = [("head", "neck"), ("neck", "l_shoulder"), ("neck", "r_shoulder"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("neck", "l_hip"), ("neck", "r_hip"), ("l_hip", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("l_shoulder", "r_shoulder")]
VIEWS = ["Side_underwater", "Side_water_level", "Side_above_water", "Aerial", "Front"]


def parse_full(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(";")
    cols = {}
    for i, h in enumerate(header):
        if h.endswith(".x") and f"{h[:-2]}.y" in header:
            cols[h[:-2]] = (i, header.index(f"{h[:-2]}.y"))
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(";")

        def v(k):
            if k >= len(f) or f[k].strip() == "":
                return None
            try:
                return float(f[k].replace(",", "."))
            except ValueError:
                return None
        out.append({nm: (v(xi), v(yi)) for nm, (xi, yi) in cols.items()})
    return out


def ang(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 180.0 if n1 < 1e-6 or n2 < 1e-6 else math.degrees(
        math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


fig, axes = plt.subplots(1, len(VIEWS), figsize=(4 * len(VIEWS), 4), dpi=110)
for ax, view in zip(axes, VIEWS):
    seqs = sorted((SX / view).rglob("base/2D_cam.txt"))
    chosen = None
    for sp in seqs[:40]:
        for d in parse_full(sp):
            P = {lk: d.get(rk) for lk, rk in SX_MAP.items()}
            if any(P[k] is None or P[k][0] is None for k in SX_MAP):
                continue
            kl = ang(P["l_hip"], P["l_knee"], P["l_ankle"])
            kr = ang(P["r_hip"], P["r_knee"], P["r_ankle"])
            if 65 <= (kl + kr) / 2 <= 105:   # 明显屈膝
                chosen = P
                break
        if chosen:
            break
    ax.set_title(f"{view}\n{'bent frame' if chosen else 'NONE'}", fontsize=9)
    ax.set_facecolor("#0a2a3a")
    if chosen:
        for a, b in BONES:
            ax.plot([chosen[a][0], chosen[b][0]], [chosen[a][1], chosen[b][1]],
                    "-", color="#7fd0ec", lw=2)
        xs = [chosen[k][0] for k in SX_MAP]; ys = [chosen[k][1] for k in SX_MAP]
        ax.scatter(xs, ys, s=18, c="orange", zorder=5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(SCRATCH / "sx_view_survey.png", facecolor="white")
print("saved outputs/_scratch/sx_view_survey.png")
