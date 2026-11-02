# Kinetics-700 蛙泳批处理 Pipeline

用 Kinetics-700 的**真实**蛙泳视频替代 SwimXYZ 合成数据，作为 RL 评分器的训练数据源。
四段式流水线，每段可独立运行、支持断点续传：

```
01_download → 02_pose → 03_feature → 04_dataset
  yt-dlp        HRNet       角度/周期      28维观测
```

## 环境

一次性装好依赖（本仓库用 `venv_pose`，已含 mmpose/mmdet/mmcv/torch）：

```bash
# 系统依赖（切片必需）
brew install ffmpeg          # macOS
# apt-get install -y ffmpeg  # Ubuntu / Colab

# Python 依赖
pip install -r scripts/requirements.txt
# 全新环境的 mm* 建议用 openmim（见 requirements.txt 顶部注释）
```

> **设备**：`--device auto` = 有 CUDA 用 CUDA，否则 CPU。**Mac 上不自动用 MPS**
> （mmcv 的 NMS 在 MPS 上会崩）。大批量 HRNet 请用 Colab / GPU 机器。

## 各脚本

| 脚本 | 输入 | 输出 |
|------|------|------|
| `01_download_kinetics_breaststroke.py` | Kinetics-700 annotation | `data/kinetics/*.mp4` + `failed_downloads.txt` |
| `02_batch_pose_estimation.py` | `*.mp4` | `data/kinetics/joints/*.pkl` + `quality_report.csv` |
| `03_batch_feature_extraction.py` | joints + quality_report | `data/features/*_{features.npy,cycles.pkl}` + `feature_report.csv` |
| `04_build_training_dataset.py` | `*_cycles.pkl` | `data/training/{train,test}_observations.npy` + `dataset_meta.json` |

- 观测为 **28 维**，与 `src/kick_env.py::BreaststrokeKickEnv` 观测空间逐位一致
  （角度 mean/std/min/max×4=16 + 对称 mean/std×2=4 + 角速度 mean/std×4=8）。
- 划分为**序列级** 80/20（按视频分组，防同一视频周期泄露到 train+test）。

## 本地运行（小批量验证）

```bash
# 1) 下载（--limit 控制条数；不加 --limit 下全部，共约 906 条）
python scripts/01_download_kinetics_breaststroke.py --data-dir data/kinetics --limit 150

# 2) 姿态估计（Mac 用 CPU，慢，~3 分钟/视频；--limit 做小批量）
python scripts/02_batch_pose_estimation.py --data-dir data/kinetics --device cpu --limit 20

# 3) 特征提取 + 周期分割
python scripts/03_batch_feature_extraction.py --data-dir data/kinetics --features-dir data/features

# 4) 构建训练集
python scripts/04_build_training_dataset.py --features-dir data/features --out-dir data/training
```

## 大批量 HRNet 处理（Colab T4 GPU，推荐）

Mac CPU ~3 分钟/视频，全量 900+ 视频不现实。把 HRNet 放到 Colab：

```python
# Colab notebook
!apt-get -qq install -y ffmpeg
!pip install -U openmim && mim install mmengine "mmcv==2.1.0" "mmdet==3.3.0" "mmpose==1.3.2"
!pip install -q yt-dlp dtw-python gymnasium stable-baselines3

# 上传本仓库（含 src/、scripts/、mmcfg_bridge.py、configs/、data/templates/）
%cd /content/Extended_Essay

# 方案A：本地已下好的 mp4 打包上传到 data/kinetics/ 后：
!python scripts/02_batch_pose_estimation.py --data-dir data/kinetics --device auto   # →cuda
!python scripts/03_batch_feature_extraction.py --data-dir data/kinetics --features-dir data/features
!python scripts/04_build_training_dataset.py --features-dir data/features --out-dir data/training

# 方案B：在 Colab 直接下载再处理（省去上传大文件）
!python scripts/01_download_kinetics_breaststroke.py --data-dir data/kinetics   # 全量
```

GPU 上 HRNet 约 10–20× 于 CPU，全量处理从数十小时降到 1–2 小时。

## 数据质量说明（真实观察）

Kinetics 是"野外"视频：水下、远景、水花、岸上镜头很常见，人体检测率参差不齐。
`quality_report.csv` 用 `success_rate<0.5 或 avg_lower_body_confidence<0.4` 标记
`low_quality`，脚本 3 只处理合格视频。实测部分片段膝角几乎全程接近 180°（未见屈膝），
反映合成→真实的域差——这正是本 EE 要讨论的核心问题之一。
