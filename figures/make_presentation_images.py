"""一次生成两张 PPT 图（自包含，可在有 Desktop 权限的终端直接运行）：

  outputs/breaststroke_midkick.png  —— 真实水下蛙泳照片，蹬腿/双腿张开最明显的一帧
                                       （从 data/raw/champion.mp4 提取，纯画面、无标注）
  outputs/swimxyz_synthetic.png     —— 用真实 SwimXYZ Side_underwater 关节数据渲染的
                                       CG 风格合成泳者（简化人体轮廓 + COCO 风格彩色关节点
                                       + 骨架线），页脚 "SwimXYZ — synthetic swimmers,
                                       exact joint labels"

运行（务必用装了 mmpose 的 venv）：
    venv_pose/bin/python figures/make_presentation_images.py

可选参数：
    --stride N    图1 姿态扫描步长（默认 12，越大越快但越粗）
    --img1-frame K  跳过扫描，直接用第 K 帧做图1
    --no-img1 / --no-img2  只生成其中一张
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "data" / "raw" / "champion.mp4"
# 图2 用 Front（正面）视角：侧/俯视在 2D 投影里严重前缩成横条、不像人；
# 正面视角左右肢分开、躯干竖直、蹬腿张开成清晰 V，最像合成泳者。
SX_VIEW = "Front"
SXDIR = ROOT / "data" / "swimxyz" / "Breaststroke" / SX_VIEW
OUT = ROOT / "outputs"
SCRATCH = OUT / "_scratch"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

S = 3  # 超采样倍率


# ============================ 通用绘图工具 ============================
def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def save_png_bgr(bgr, path, dpi=300):
    """BGR ndarray → PNG（写入 DPI 元数据）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


def load_font(size, bold=True):
    cands = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/HelveticaNeue.ttc"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"])
    for p in cands:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ============================ 图 1：蛙泳照片 ============================
LOWER = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def _knee_angle(j, hip, knee, ankle):
    h = np.array(j[hip][:2]); k = np.array(j[knee][:2]); a = np.array(j[ankle][:2])
    v1, v2 = h - k, a - k
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def find_and_save_midkick(stride: int, forced_frame: int | None):
    import mmcfg_bridge  # noqa: F401  必须在 PoseEstimator 之前
    from src.video_loader import VideoLoader
    from src.pose_estimator import PoseEstimator, PoseEstimationError

    loader = VideoLoader(str(VIDEO), fps=25, resize_width=1920, resize_height=1080)
    cap = loader.cap
    N = loader.total_frames

    def grab(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return fr if ok else None

    if forced_frame is not None:
        fr = grab(forced_frame)
        loader.release()
        if fr is None:
            raise RuntimeError(f"读取帧 {forced_frame} 失败")
        save_png_bgr(fr, OUT / "breaststroke_midkick.png")
        print(f"[图1] 已用指定帧 {forced_frame} 保存 outputs/breaststroke_midkick.png")
        return

    est = PoseEstimator(device="cpu", min_confidence=0.3)  # mmcv nms 无 MPS 内核
    rows = []
    idxs = list(range(int(N * 0.05), int(N * 0.95), stride))
    print(f"[图1] 扫描 {len(idxs)} 帧找『双腿张开』最明显的一帧 …")
    for n, idx in enumerate(idxs):
        if n % 20 == 0:
            print(f"      {n}/{len(idxs)} (frame {idx})")
        fr = grab(idx)
        if fr is None:
            continue
        try:
            j = est.estimate_single(fr)
        except PoseEstimationError:
            continue
        lh, rh = np.array(j["left_hip"][:2]), np.array(j["right_hip"][:2])
        lk, rk = np.array(j["left_knee"][:2]), np.array(j["right_knee"][:2])
        lf, rf = np.array(j["left_ankle"][:2]), np.array(j["right_ankle"][:2])
        scale = np.linalg.norm(np.array(j["left_shoulder"][:2]) - lh) + 1e-6
        foot_sep = np.linalg.norm(lf - rf) / scale       # 双脚分离（俯/斜视下的 V）
        knee_sep = np.linalg.norm(lk - rk) / scale
        bend = min(_knee_angle(j, "left_hip", "left_knee", "left_ankle"),
                   _knee_angle(j, "right_hip", "right_knee", "right_ankle"))
        conf = np.mean([j[n2][2] for n2 in LOWER])
        spread = foot_sep + knee_sep
        rows.append((idx, spread, foot_sep, knee_sep, bend, conf))

    loader.release()
    if not rows:
        raise RuntimeError("没有可用帧（姿态全部失败）")

    # 主选：在置信度达标的帧里取『张开度最大』；同时存若干候选 + 对比图
    ok = [r for r in rows if r[5] > 0.5] or rows
    ok.sort(key=lambda r: -r[1])
    cands = ok[:8]
    print("[图1] 候选（张开度降序）：")
    for idx, sp, fs, ks, bd, cf in cands:
        print(f"      frame {idx:5d} spread={sp:.2f} foot={fs:.2f} knee={ks:.2f} bend={bd:5.1f} conf={cf:.2f}")

    loader2 = VideoLoader(str(VIDEO), fps=25, resize_width=1920, resize_height=1080)
    tiles = []
    for k, (idx, sp, *_rest) in enumerate(cands):
        loader2.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok2, fr = loader2.cap.read()
        if not ok2:
            continue
        cv2.imwrite(str(SCRATCH / f"img1_cand{k}_f{idx}.png"), fr)  # 干净候选
        t = cv2.resize(fr, (480, 270))
        cv2.putText(t, f"{idx} s={sp:.2f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, f"{idx} s={sp:.2f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
        if k == 0:
            save_png_bgr(fr, OUT / "breaststroke_midkick.png")  # 主输出
    loader2.release()
    if tiles:
        while len(tiles) < 8:
            tiles.append(np.zeros((270, 480, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(2)])
        cv2.imwrite(str(SCRATCH / "img1_candidates.jpg"), grid)
    print(f"[图1] 主图= frame {cands[0][0]} → outputs/breaststroke_midkick.png")
    print("[图1] 若想换，把 outputs/_scratch/img1_candN_*.png 选一张覆盖主图，"
          "或用 --img1-frame K 重跑。对比图：outputs/_scratch/img1_candidates.jpg")


# ============================ 图 2：SwimXYZ 合成渲染 ============================
# swimxyz base 关节名 → 逻辑关节
SX_MAP = {
    "head": "Head", "neck": "Neck", "pelvis": "Pelvis",
    "l_shoulder": "L UpperArm", "r_shoulder": "R UpperArm",
    "l_elbow": "L Forearm", "r_elbow": "R Forearm",
    "l_wrist": "L Hand", "r_wrist": "R Hand",
    "l_hip": "L Thigh", "r_hip": "R Thigh",
    "l_knee": "L Calf", "r_knee": "R Calf",
    "l_ankle": "L Foot", "r_ankle": "R Foot",
    "l_toe": "L Toe0", "r_toe": "R Toe0",
}
BONES = [  # 骨架线 / 人体轮廓胶囊
    ("head", "neck"), ("neck", "pelvis"),
    ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"), ("pelvis", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
    ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
]
# COCO 风格关节配色（OpenPose 彩虹点）
JCOLOR = {
    "head": (255, 64, 64), "neck": (255, 128, 0),
    "l_shoulder": (255, 200, 0), "r_shoulder": (200, 255, 0),
    "l_elbow": (0, 220, 120), "r_elbow": (0, 220, 200),
    "l_wrist": (0, 180, 255), "r_wrist": (0, 120, 255),
    "l_hip": (255, 220, 0), "r_hip": (255, 170, 0),
    "l_knee": (255, 110, 40), "r_knee": (255, 70, 40),
    "l_ankle": (255, 40, 90), "r_ankle": (230, 40, 140),
    "pelvis": (255, 255, 255), "l_toe": (180, 80, 255), "r_toe": (140, 80, 255),
}


def parse_full(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(";")
    cols = {}
    for i, h in enumerate(header):
        if h.endswith(".x"):
            base = h[:-2]
            if f"{base}.y" in header:
                cols[base] = (i, header.index(f"{base}.y"))
    frames = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(";")

        def val(k):
            if k >= len(f) or f[k].strip() == "":
                return None
            try:
                return float(f[k].replace(",", "."))
            except ValueError:
                return None
        d = {}
        for nm, (xi, yi) in cols.items():
            x, y = val(xi), val(yi)
            d[nm] = (x, y, x is not None and y is not None)
        frames.append(d)
    return frames


def _ang(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def pick_swimxyz_frame():
    """挑 Front 视角下『蛙泳蹬腿张开成 V』最明显的一帧：屈膝 + 双腿外展 + 左右对称。

    核心关节必须可见，脚趾可选。打分 = 双脚横向分离（归一肩宽）+ 膝分离，
    限制双膝弯曲(kmean∈[70,150])，并惩罚左右不对称。
    """
    seqs = sorted(SXDIR.rglob("base/2D_cam.txt"))
    if not seqs:
        raise RuntimeError(f"未找到 SwimXYZ {SX_VIEW} base/2D_cam.txt")
    optional = {"l_toe", "r_toe"}
    core = [SX_MAP[k] for k in SX_MAP if k not in optional]
    best = None  # (score, pts, info)
    for sp in seqs[:120]:
        for d in parse_full(sp):
            if not all(d.get(n, (None, None, False))[2] for n in core):
                continue
            pts = {}
            for lk, rk in SX_MAP.items():
                v = d.get(rk, (None, None, False))
                if v[2]:
                    pts[lk] = (v[0], v[1])
            kl = _ang(pts["l_hip"], pts["l_knee"], pts["l_ankle"])
            kr = _ang(pts["r_hip"], pts["r_knee"], pts["r_ankle"])
            kmean = (kl + kr) / 2
            if not (72 <= kmean <= 118):       # 深屈膝『蛙抱』（非全直、非深折）
                continue
            mhx = (pts["l_hip"][0] + pts["r_hip"][0]) / 2
            mhy = (pts["l_hip"][1] + pts["r_hip"][1]) / 2
            hx, hy = pts["head"]
            if hy >= mhy:                       # 要求头在髋上方（竖直朝向）
                continue
            ax, ay = mhx - hx, mhy - hy          # 体轴（头→髋）
            tilt = abs(ax) / (abs(ay) + 1e-6)    # 越小越竖直
            shoulder_w = math.dist(pts["l_shoulder"], pts["r_shoulder"]) + 1e-6
            foot_sep = math.dist(pts["l_ankle"], pts["r_ankle"]) / shoulder_w
            knee_sep = math.dist(pts["l_knee"], pts["r_knee"]) / shoulder_w
            # 竖直 + 双腿张开宽（V 明显）+ 左右对称
            score = foot_sep + 0.4 * knee_sep - 0.9 * tilt - 0.01 * abs(kl - kr)
            if best is None or score > best[0]:
                best = (score, pts, (kl, kr, foot_sep, tilt))
    if best is None:
        raise RuntimeError("SwimXYZ 序列里没有合适的蹬腿张开帧")
    k = best[2]
    print(f"[图2] 选中蹬腿帧 knee=({k[0]:.0f},{k[1]:.0f})° foot_sep={k[2]:.2f}×肩宽 "
          f"tilt={k[3]:.2f} score={best[0]:.2f}（{len(best[1])} 个可见关节）")
    return best[1]


def _fit(pts, W, H, frac=0.66, cyf=0.5):
    xs = [p[0] for p in pts.values()]; ys = [p[1] for p in pts.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = (x1 - x0) or 1, (y1 - y0) or 1
    sc = min(W * frac / bw, H * frac / bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {k: (W / 2 + (p[0] - cx) * sc, H * cyf + (p[1] - cy) * sc) for k, p in pts.items()}


def _underwater_bg(W, H):
    yy = np.linspace(0, 1, H)[:, None]
    top = np.array([10, 26, 52]); bot = np.array([12, 60, 78])
    base = (top * (1 - yy) + bot * yy)
    img = np.repeat(base[:, None, :], W, axis=1)
    xx = np.linspace(0, 1, W)[None, :]
    caustic = 8 * (np.sin(xx * 38 + 0.6) * np.sin(yy * 26)) ** 2
    img += caustic[..., None]
    # 顶部光束
    beam = np.clip(1 - (yy / 0.45), 0, 1) ** 2 * 16
    img += beam[..., None] * np.array([0.4, 0.9, 1.0])
    # 暗角
    YY, XX = np.mgrid[0:H, 0:W]
    r = ((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2
    img *= (1 - 0.45 * np.clip(r, 0, 1))[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def _taper(draw, p1, r1, p2, r2, fill):
    """画一段两端半径不同的『锥形肢体』（梯形 + 两端圆头），形成有粗细变化的实心人体。"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    draw.polygon([(p1[0] + nx * r1, p1[1] + ny * r1),
                  (p2[0] + nx * r2, p2[1] + ny * r2),
                  (p2[0] - nx * r2, p2[1] - ny * r2),
                  (p1[0] - nx * r1, p1[1] - ny * r1)], fill=fill)
    draw.ellipse([p1[0] - r1, p1[1] - r1, p1[0] + r1, p1[1] + r1], fill=fill)
    draw.ellipse([p2[0] - r2, p2[1] - r2, p2[0] + r2, p2[1] + r2], fill=fill)


def render_swimxyz():
    W, H = 1920, 1080
    pts = _fit(pick_swimxyz_frame(), W, H, frac=0.76, cyf=0.45)  # Front 竖向人形，略上移留页脚空间
    bg = _underwater_bg(W, H)

    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def P(k):
        return (pts[k][0] * S, pts[k][1] * S)

    # 1) 实心 CG 假人：按关节给粗细，画锥形肢体 + 躯干核心，暗边描廓使其成一个整体人形
    RB = H * S
    rad = {
        "head": 0.060 * RB, "neck": 0.050 * RB,
        "l_shoulder": 0.052 * RB, "r_shoulder": 0.052 * RB,
        "l_elbow": 0.031 * RB, "r_elbow": 0.031 * RB,
        "l_wrist": 0.018 * RB, "r_wrist": 0.018 * RB,
        "l_hip": 0.060 * RB, "r_hip": 0.060 * RB,
        "l_knee": 0.041 * RB, "r_knee": 0.041 * RB,
        "l_ankle": 0.024 * RB, "r_ankle": 0.024 * RB,
        "l_toe": 0.013 * RB, "r_toe": 0.013 * RB,
    }

    def mid(a, b):
        return ((pts[a][0] + pts[b][0]) / 2 * S, (pts[a][1] + pts[b][1]) / 2 * S)

    msh, mhip = mid("l_shoulder", "r_shoulder"), mid("l_hip", "r_hip")
    LIMBS = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
             ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
             ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
             ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe")]
    fill_body = (150, 198, 222, 255)
    edge_body = (36, 90, 116, 255)
    core_sh, core_hip = 0.080 * RB, 0.066 * RB

    def draw_body(grow, col):
        _taper(d, msh, core_sh + grow, mhip, core_hip + grow, col)          # 躯干核心
        _taper(d, P("neck"), rad["neck"] + grow, msh, core_sh * 0.82 + grow, col)  # 颈
        for a, b in LIMBS:
            if a in pts and b in pts:
                _taper(d, P(a), rad[a] + grow, P(b), rad[b] + grow, col)    # 四肢
        hx, hy = P("head")
        hr = rad["head"] + grow
        d.ellipse([hx - hr, hy - hr * 1.12, hx + hr, hy + hr * 1.12], fill=col)  # 头

    draw_body(0.013 * RB, edge_body)   # 暗边描廓
    draw_body(0.0, fill_body)          # 实心体

    # 2) 骨架线（细，叠在模型上做标注）+ COCO 风格彩色关节点（白描边）
    for a, b in BONES:
        if a in pts and b in pts:
            d.line([P(a), P(b)], fill=(28, 66, 86, 235), width=int(0.0034 * RB))
    r = int(0.0105 * RB)
    ow = int(0.0026 * RB)
    for k, c in JCOLOR.items():
        if k not in pts:
            continue
        x, y = P(k)
        d.ellipse([x - r - ow, y - r - ow, x + r + ow, y + r + ow], fill=(255, 255, 255, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=c + (255,))

    ov_small = ov.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(bg, ov_small)

    # 4) 页脚（顶层、不超采样也够清晰，用描边）
    d2 = ImageDraw.Draw(out)
    f = load_font(int(0.026 * H), bold=True)
    txt = "SwimXYZ — synthetic swimmers, exact joint labels"
    bb = d2.textbbox((0, 0), txt, font=f, stroke_width=2)
    tw = bb[2] - bb[0]
    d2.text(((W - tw) / 2, H - int(0.075 * H)), txt, font=f, fill=(235, 248, 255, 255),
            stroke_width=2, stroke_fill=(4, 12, 24, 255))
    tag = load_font(int(0.018 * H), bold=False)
    d2.text((int(0.02 * W), int(0.03 * H)), "Front view  ·  base (SMPL) skeleton  ·  2D ground-truth joints",
            font=tag, fill=(200, 230, 245, 220), stroke_width=1, stroke_fill=(4, 12, 24, 220))

    out.convert("RGB").save(OUT / "swimxyz_synthetic.png", dpi=(300, 300))
    print(f"[图2] 已保存 outputs/swimxyz_synthetic.png ({W}x{H})")


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--img1-frame", type=int, default=640,
                    help="图1 使用的帧（默认 640：池中央干净侧视蹬腿）")
    ap.add_argument("--scan", action="store_true",
                    help="改为扫描全片自动选『双腿张开最大』帧（慢，会跑 HRNet）")
    ap.add_argument("--no-img1", action="store_true")
    ap.add_argument("--no-img2", action="store_true")
    a = ap.parse_args()
    if not a.no_img1:
        find_and_save_midkick(a.stride, None if a.scan else a.img1_frame)
    if not a.no_img2:
        render_swimxyz()


if __name__ == "__main__":
    main()
