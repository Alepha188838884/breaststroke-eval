"""扫描 champion.mp4 候选帧：跑真实 HRNet，按下肢置信度 + 膝弯曲度选最佳帧。

输出：终端打印每个候选的指标，并把带骨架的低清预览写到 outputs/_scratch/scan_*.jpg
方便人工二次确认。最终渲染由 make_skeleton_overlay.py 负责。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mmcfg_bridge  # noqa: F401  必须在构造 PoseEstimator 前 import（重定向 config 路径）
from src.video_loader import VideoLoader
from src.pose_estimator import PoseEstimator, PoseEstimationError

VIDEO = ROOT / "data" / "raw" / "champion.mp4"
SCRATCH = ROOT / "outputs" / "_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)

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
LOWER = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def knee_angle(joints, hip, knee, ankle):
    h = np.array(joints[hip][:2]); k = np.array(joints[knee][:2]); a = np.array(joints[ankle][:2])
    v1 = h - k; v2 = a - k
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cs = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return math.degrees(math.acos(cs))


def quick_draw(frame, joints):
    img = frame.copy()
    for a, b in SKELETON:
        pa = tuple(map(int, joints[a][:2])); pb = tuple(map(int, joints[b][:2]))
        cv2.line(img, pa, pb, (0, 255, 0), 3, cv2.LINE_AA)
    for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                 "left_knee", "right_knee", "left_ankle", "right_ankle"]:
        p = tuple(map(int, joints[name][:2]))
        cv2.circle(img, p, 7, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def main():
    loader = VideoLoader(str(VIDEO), fps=25, resize_width=1920, resize_height=1080)
    cap = loader.cap
    n = loader.total_frames
    est = PoseEstimator(device="cpu", min_confidence=0.3)  # mmcv 的 nms 无 MPS 内核

    # 广扫：覆盖侧视与斜视段
    candidates = list(range(700, 1600, 40))
    rows = []
    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            j = est.estimate_single(frame)
        except PoseEstimationError as e:
            print(f"{idx:5d}  SKIP  {e}")
            continue
        lower_conf = np.mean([j[name][2] for name in LOWER])
        full_conf = np.mean([j[name][2] for name in
                             ["left_shoulder", "right_shoulder"] + LOWER])
        ka_l = knee_angle(j, "left_hip", "left_knee", "left_ankle")
        ka_r = knee_angle(j, "right_hip", "right_knee", "right_ankle")
        bend = min(ka_l, ka_r)  # 越小越弯（蹬腿动作越明显）
        rows.append((idx, full_conf, lower_conf, ka_l, ka_r, bend, j, frame))
        print(f"{idx:5d}  full={full_conf:.3f} lower={lower_conf:.3f}  "
              f"kneeL={ka_l:5.1f} kneeR={ka_r:5.1f}  bend={bend:5.1f}")

    loader.release()

    # 选择：高置信度(>0.6) 且膝弯曲明显(bend 小)的若干帧出预览
    good = [r for r in rows if r[1] > 0.55]
    good.sort(key=lambda r: r[5])  # bend 升序（最弯在前）
    top = good[:6]
    print("\n=== 预览候选（最弯 + 高置信度）===")
    for idx, fc, lc, kl, kr, bend, j, frame in top:
        prev = quick_draw(frame, j)
        prev = cv2.resize(prev, (960, 540))
        out = SCRATCH / f"scan_{idx}_bend{int(bend)}_conf{int(fc*100)}.jpg"
        cv2.imwrite(str(out), prev)
        print(f"  saved {out.name}")


if __name__ == "__main__":
    main()
