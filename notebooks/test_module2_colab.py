"""
Colab 测试脚本：蛙泳蹬腿评估系统 Module 1-4 流水线验证。

使用方式：
  1. 在 Colab 中新建 notebook（运行时选择 GPU）。
  2. 把本文件内容按 `# %%` 分隔逐 cell 复制进 notebook 中依次执行。
  3. 也可以用 jupytext / VSCode 把本文件直接转成 .ipynb。

前置假设：
  - 当前工作目录为项目根（含有 src/ 目录）。Colab 中可先 git clone 或
    将整个项目上传到 /content/Extended Essay 并 `%cd` 进去。
"""

# %% [Cell 1] 环境安装
# 在 Colab 中执行（pip / mim 命令通过 ! 前缀）。本地运行时请删除 ! 前缀
# 或改成 subprocess 调用。numpy 必须 < 2，否则 mmcv 编译会失败。

!pip install -q "numpy<2"
!pip install -q -U openmim
!mim install -q mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0"
!pip install -q opencv-python tqdm pyyaml scipy dtw-python gymnasium stable-baselines3
!pip install -q yt-dlp matplotlib


# %% [Cell 2] 下载测试视频
# 占位 URL：请替换成实际的蛙泳视频 URL（建议侧拍、清晰、单人）。
# 备选方案：直接用 Colab 左侧文件面板上传视频，命名为 test_breaststroke.mp4
# 放到 data/raw/ 即可跳过下载。

import os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/templates", exist_ok=True)

# TODO: 替换为实际蛙泳视频 URL
VIDEO_URL = "https://www.youtube.com/watch?v=REPLACE_ME_WITH_BREASTSTROKE_URL"
OUTPUT_PATH = "data/raw/test_breaststroke.mp4"

if not os.path.exists(OUTPUT_PATH):
    # 用 yt-dlp 下载，限制最高 720p mp4；下载失败时提示手动上传
    ret = os.system(
        f'yt-dlp -f "best[ext=mp4][height<=720]" -o "{OUTPUT_PATH}" "{VIDEO_URL}"'
    )
    if ret != 0 or not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"yt-dlp 下载失败。请手动上传蛙泳视频到 {OUTPUT_PATH}。\n"
            "  方法：Colab 左侧文件面板 → 上传 → 把文件拖入 data/raw/ "
            "并重命名为 test_breaststroke.mp4"
        )
else:
    print(f"已存在视频文件: {OUTPUT_PATH}")


# %% [Cell 3] 单独测试 Module 2（PoseEstimator）
# 加载视频前 30 帧，跑 HRNet 姿态估计，检查关节输出与下肢置信度。

import sys
import numpy as np

# 确保能 import src.* （Colab 中可能需要把项目根加入 sys.path）
if "." not in sys.path:
    sys.path.insert(0, ".")

from src.video_loader import VideoLoader
from src.pose_estimator import PoseEstimator

VIDEO_PATH = "data/raw/test_breaststroke.mp4"

loader = VideoLoader(VIDEO_PATH, fps=30, resize_width=640, resize_height=480)
all_frames = loader.extract_all_frames()
loader.release()
print(f"视频抽帧完成 | 总抽帧数={len(all_frames)}")

frames_30 = all_frames[:30]
print(f"取前 30 帧用于 Module 2 单测 | shape={frames_30.shape}")

estimator = PoseEstimator(device="cuda")
results_30 = estimator.estimate_batch(frames_30)

succ = sum(1 for r in results_30 if r is not None)
print(f"\n[Module 2 结果] 成功帧数 / 总帧数 = {succ} / {len(results_30)}")

# 打印第一个成功帧的关节信息
first_idx = next((i for i, r in enumerate(results_30) if r is not None), None)
if first_idx is None:
    raise RuntimeError("前 30 帧全部姿态估计失败，请检查视频质量或 GPU 环境。")

first_joints = results_30[first_idx]
print(f"\n[Frame {first_idx}] 关节坐标（x, y, confidence）：")
for name, (x, y, c) in first_joints.items():
    print(f"  {name:>16s}: ({x:7.2f}, {y:7.2f}, {c:.3f})")

# 下肢关节置信度均值
LOWER = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
lower_confs = []
for r in results_30:
    if r is None:
        continue
    for n in LOWER:
        lower_confs.append(r[n][2])
print(f"\n下肢关节置信度均值（{len(lower_confs)} 个样本）: "
      f"{np.mean(lower_confs):.3f}")


# %% [Cell 4] 测试完整 pipeline Module 1-4
# 抽全部帧 → HRNet → FeatureExtractor → 保存模板 → DTW 周期分割 + 对齐

from src.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.dtw_aligner import DTWAligner

# 4.1 重新抽全部帧并跑姿态估计
loader = VideoLoader(VIDEO_PATH, fps=30, resize_width=640, resize_height=480)
all_frames = loader.extract_all_frames()
loader.release()
print(f"抽帧完成 | shape={all_frames.shape}")

joint_seq = estimator.estimate_batch(all_frames)
succ_all = sum(1 for r in joint_seq if r is not None)
print(f"姿态估计完成 | 成功 {succ_all} / {len(joint_seq)}")

# 4.2 特征提取
extractor = FeatureExtractor(fps=30, min_confidence=0.3)
features = extractor.extract_sequence_features(joint_seq)
print(f"\n[FeatureExtractor] 特征矩阵 shape = {features.shape}")
print("前 4 列角度范围（degrees）：")
for i, name in enumerate(FEATURE_NAMES[:4]):
    col = features[:, i]
    print(f"  {name:>20s}: min={col.min():.2f}, "
          f"max={col.max():.2f}, mean={col.mean():.2f}")

# 4.3 保存前 30 帧有效特征作为模板（仅作 pipeline 联通性测试，非真实冠军模板）
TEMPLATE_PATH = "data/templates/test_template.npy"
if features.shape[0] < 30:
    raise RuntimeError(
        f"有效特征帧不足 30 ({features.shape[0]})，无法生成测试模板。"
    )
DTWAligner.save_template(features[:30], TEMPLATE_PATH)

# 4.4 DTW 周期分割 + 对齐第一个周期
aligner = DTWAligner(TEMPLATE_PATH)
cycles = aligner.segment_kick_cycles(features, angle_column=0)
print(f"\n[DTW] 检测到蹬腿周期数 = {len(cycles)}")

if not cycles:
    print("未分割出有效周期；可能视频中蹬腿动作不明显或长度不足。")
else:
    aligned_q, aligned_t, dist = aligner.align(cycles[0])
    deviations = aligner.compute_phase_deviations(aligned_q, aligned_t)
    overall = aligner.compute_overall_deviation(deviations)
    print(f"第一个周期 | DTW 归一化距离 = {dist:.4f}")
    print("阶段偏差：")
    for k, v in deviations.items():
        print(f"  {k:>9s}: {v:.4f}")
    print(f"加权偏差总分 = {overall:.4f}")

print("\nFULL PIPELINE TEST PASSED")


# %% [Cell 5] 可视化验证
# 图1：在第一个成功帧上叠加关节点 + 骨架
# 图2：left_knee_angle 时间序列曲线

import cv2
import matplotlib.pyplot as plt

# COCO 骨架连接（项目所用关节子集）
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

vis_idx = next((i for i, r in enumerate(joint_seq) if r is not None), None)
if vis_idx is None:
    raise RuntimeError("无可用帧用于可视化。")

vis_frame = cv2.cvtColor(all_frames[vis_idx], cv2.COLOR_BGR2RGB).copy()
vis_joints = joint_seq[vis_idx]

# 画骨架（蓝色线）
for a, b in SKELETON:
    if a in vis_joints and b in vis_joints:
        xa, ya, ca = vis_joints[a]
        xb, yb, cb = vis_joints[b]
        if ca >= 0.3 and cb >= 0.3:
            cv2.line(vis_frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (0, 128, 255), 2)
# 画关节点（红色圆点）
for name, (x, y, c) in vis_joints.items():
    if c >= 0.3:
        cv2.circle(vis_frame, (int(x), int(y)), 5, (255, 0, 0), -1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(vis_frame)
axes[0].set_title(f"Frame {vis_idx}: detected joints + skeleton")
axes[0].axis("off")

# left_knee_angle 时间序列
left_knee_col = FEATURE_NAMES.index("left_knee_angle")
axes[1].plot(features[:, left_knee_col], color="crimson", linewidth=1.5)
axes[1].set_xlabel("Valid frame index")
axes[1].set_ylabel("left_knee_angle (deg)")
axes[1].set_title("left_knee_angle over time")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
