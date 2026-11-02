"""生成 PPT 用的 HRNet 骨架叠加图（真实推理）。

输出：
  outputs/hrnet_skeleton_overlay.png  —— 带关节标签 + 角标的标注版
  outputs/hrnet_skeleton_clean.png    —— 仅骨架+关节点的干净版（PPT 二选一）

用法：
  venv_pose/bin/python figures/make_skeleton_overlay.py            # 用缓存关键点
  venv_pose/bin/python figures/make_skeleton_overlay.py --refresh  # 重新跑 HRNet
  venv_pose/bin/python figures/make_skeleton_overlay.py --frame 1500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "data" / "raw" / "champion.mp4"
OUT = ROOT / "outputs"
SCRATCH = OUT / "_scratch"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

CHOSEN_FRAME = 980

SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
JOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]

# 身体部位配色（用于骨架线渐变端点）：上肢蓝 → 下肢橙/红
PART_COLOR = {
    "left_shoulder": (46, 155, 255),   # 蓝
    "right_shoulder": (46, 155, 255),
    "left_hip": (255, 154, 31),        # 橙（下肢起点）
    "right_hip": (255, 154, 31),
    "left_knee": (255, 106, 19),       # 深橙
    "right_knee": (255, 106, 19),
    "left_ankle": (255, 45, 45),       # 红
    "right_ankle": (255, 45, 45),
}

S = 3  # 超采样倍率


# ----------------------------- 推理 / 缓存 -----------------------------
def run_pose(frame_idx: int):
    import mmcfg_bridge  # noqa: F401  必须在 PoseEstimator 前 import
    from src.video_loader import VideoLoader
    from src.pose_estimator import PoseEstimator

    loader = VideoLoader(str(VIDEO), fps=25, resize_width=1920, resize_height=1080)
    loader.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = loader.cap.read()
    loader.release()
    if not ret:
        raise RuntimeError(f"读取帧 {frame_idx} 失败")
    est = PoseEstimator(device="cpu", min_confidence=0.3)  # mmcv nms 无 MPS 内核
    joints = est.estimate_single(frame)
    return frame, joints


def get_frame_and_joints(frame_idx: int, refresh: bool):
    cache = SCRATCH / f"chosen_{frame_idx}.npz"
    if cache.is_file() and not refresh:
        d = np.load(cache, allow_pickle=True)
        return d["frame"], d["joints"].item()
    frame, joints = run_pose(frame_idx)
    np.savez(cache, frame=frame, joints=np.array(joints, dtype=object))
    return frame, joints


# ----------------------------- 绘制工具 -----------------------------
def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def conf_color(c: float):
    """置信度 → 颜色：0.3 黄 → 1.0 绿。"""
    t = max(0.0, min(1.0, (c - 0.3) / 0.7))
    return lerp((255, 205, 0), (0, 230, 118), t)


def load_font(size: int, bold: bool = False):
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/HelveticaNeue.ttc"]
        if bold else
        ["/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Helvetica.ttc",
         "/System/Library/Fonts/HelveticaNeue.ttc"]
    )
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def grad_line(draw, p1, p2, c1, c2, width, alpha=255, n=48):
    """画渐变线（带圆角，端点处补圆避免缺口）。"""
    for i in range(n):
        t1, t2 = i / n, (i + 1) / n
        x1 = p1[0] + (p2[0] - p1[0]) * t1
        y1 = p1[1] + (p2[1] - p1[1]) * t1
        x2 = p1[0] + (p2[0] - p1[0]) * t2
        y2 = p1[1] + (p2[1] - p1[1]) * t2
        c = lerp(c1, c2, (t1 + t2) / 2) + (alpha,)
        draw.line([(x1, y1), (x2, y2)], fill=c, width=width)
        r = width / 2
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=c)


# ----------------------------- 主渲染 -----------------------------
def render(frame_bgr, joints, labeled: bool):
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1) 轻微压暗（alpha=0.15 黑蒙版）让骨架更突出
    dark = (rgb.astype(np.float32) * 0.85).astype(np.uint8)
    photo = Image.fromarray(dark).convert("RGBA")

    # 2) 超采样图层画骨架/关节/标签
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)

    def sp(name):  # 关节坐标 → 超采样坐标
        x, y, _ = joints[name]
        return (x * S, y * S)

    lw = int(round(0.0034 * H)) * S          # 线宽 ~3.7px @1080
    halo_w = lw + int(round(0.004 * H)) * S   # 深色 halo 宽度

    # 2a) 先画所有 halo（深色描边，让骨架在水下背景里弹出）
    for a, b in SKELETON:
        grad_line(d, sp(a), sp(b), (8, 12, 20), (8, 12, 20), halo_w, alpha=150)
    # 2b) 再画渐变彩色骨架线
    for a, b in SKELETON:
        grad_line(d, sp(a), sp(b), PART_COLOR[a], PART_COLOR[b], lw, alpha=255)

    # 2c) 关节点：白描边 + 置信度着色
    r = int(round(0.0072 * H)) * S            # 半径 ~7.8px @1080
    ow = int(round(0.0020 * H)) * S           # 白描边宽
    for name in JOINTS:
        x, y = sp(name)
        col = conf_color(joints[name][2])
        d.ellipse([x - r - ow, y - r - ow, x + r + ow, y + r + ow],
                  fill=(255, 255, 255, 255))
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))

    # 2d) 关节标签：向外偏移 + 迭代去重叠 + 引导线（处理侧视下左右关节重合）
    if labeled:
        font = load_font(int(round(0.019 * H)) * S, bold=True)
        cx = np.mean([joints[n][0] for n in JOINTS]) * S
        cy = np.mean([joints[n][1] for n in JOINTS]) * S
        off = int(round(0.055 * H)) * S
        pad = int(round(0.006 * H)) * S
        gap = int(round(0.004 * H)) * S
        Wp, Hp = W * S, H * S

        # 初始位置：沿“离质心”方向外推；左右关节加上下偏置打破完全重合
        labels = []
        for name in JOINTS:
            x, y = sp(name)
            dx, dy = x - cx, y - cy
            nrm = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / nrm, dy / nrm
            bias = -off * 0.45 if name.startswith("left") else off * 0.45
            bb = d.textbbox((0, 0), name, font=font, stroke_width=max(1, S // 2))
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            labels.append({
                "name": name, "ax": x, "ay": y,
                "cx": x + ux * off, "cy": y + uy * off + bias,
                "w": tw + 2 * pad, "h": th + 2 * pad, "bb": bb,
            })

        # 迭代：AABB 互推开 + 夹回画面内
        for _ in range(200):
            for i in range(len(labels)):
                a = labels[i]
                for j in range(i + 1, len(labels)):
                    b = labels[j]
                    ox = (a["w"] + b["w"]) / 2 + gap - abs(a["cx"] - b["cx"])
                    oy = (a["h"] + b["h"]) / 2 + gap - abs(a["cy"] - b["cy"])
                    if ox > 0 and oy > 0:
                        if ox < oy:
                            s = ox / 2 * (1 if a["cx"] <= b["cx"] else -1)
                            a["cx"] -= s; b["cx"] += s
                        else:
                            s = oy / 2 * (1 if a["cy"] <= b["cy"] else -1)
                            a["cy"] -= s; b["cy"] += s
            for a in labels:
                a["cx"] = min(max(a["cx"], a["w"] / 2 + 4 * S), Wp - a["w"] / 2 - 4 * S)
                a["cy"] = min(max(a["cy"], a["h"] / 2 + 4 * S), Hp - a["h"] / 2 - 4 * S)

        # 先画引导线（关节 → 标签），再画底框与文字
        for a in labels:
            d.line([(a["ax"], a["ay"]), (a["cx"], a["cy"])],
                   fill=(255, 255, 255, 140), width=max(1, S))
        for a in labels:
            x0, y0 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
            d.rounded_rectangle([x0, y0, x0 + a["w"], y0 + a["h"]],
                                radius=int(0.5 * pad) + S, fill=(10, 14, 22, 185))
            d.text((x0 + pad - a["bb"][0], y0 + pad - a["bb"][1]), a["name"],
                   font=font, fill=(255, 255, 255, 255),
                   stroke_width=max(1, S // 2), stroke_fill=(0, 0, 0, 255))

    # 2e) 角标（仅标注版，左下角）
    if labeled:
        ffont = load_font(int(round(0.0165 * H)) * S, bold=False)
        ftxt = "HRNet-W48  |  COCO 17 keypoints  |  conf ≥ 0.3"
        m = int(round(0.022 * H)) * S
        bb = d.textbbox((0, 0), ftxt, font=ffont, stroke_width=max(1, S // 2))
        fh = bb[3] - bb[1]
        d.text((m, H * S - m - fh - bb[1]), ftxt, font=ffont,
               fill=(255, 255, 255, 235), stroke_width=max(1, S // 2),
               stroke_fill=(0, 0, 0, 230))

    # 3) 降采样合成
    ov_small = ov.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(photo, ov_small).convert("RGB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=CHOSEN_FRAME)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    frame, joints = get_frame_and_joints(args.frame, args.refresh)
    print(f"帧 {args.frame} 关键点置信度：")
    for n in JOINTS:
        print(f"  {n:16s} ({joints[n][0]:7.1f},{joints[n][1]:7.1f})  conf={joints[n][2]:.3f}")

    labeled = render(frame, joints, labeled=True)
    clean = render(frame, joints, labeled=False)
    p1 = OUT / "hrnet_skeleton_overlay.png"
    p2 = OUT / "hrnet_skeleton_clean.png"
    labeled.save(p1, dpi=(300, 300))
    clean.save(p2, dpi=(300, 300))
    print(f"已保存：{p1}  ({labeled.size[0]}x{labeled.size[1]})")
    print(f"已保存：{p2}")

    # 同时存一份缩略图便于快速查看
    labeled.resize((960, 540), Image.LANCZOS).save(SCRATCH / "preview_labeled.png")
    clean.resize((960, 540), Image.LANCZOS).save(SCRATCH / "preview_clean.png")


if __name__ == "__main__":
    main()
