# Breaststroke Kick Evaluation System - Architecture Document

## 1. Project Structure

```
breaststroke-eval/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   ├── raw/                     # Original video files
│   ├── processed/               # Extracted keypoints (JSON/NPY)
│   └── templates/               # Champion reference angle sequences
├── src/
│   ├── __init__.py
│   ├── video_loader.py          # Module 1
│   ├── pose_estimator.py        # Module 2
│   ├── feature_extractor.py     # Module 3
│   ├── dtw_aligner.py           # Module 4
│   ├── rl_env.py                # Module 5 (CORE)
│   ├── train_agent.py           # Module 5b
│   ├── feedback.py              # Module 6
│   └── pipeline.py              # End-to-end orchestration
├── models/                      # Saved RL checkpoints
├── notebooks/                   # Colab notebooks
├── tests/
├── requirements.txt
└── README.md
```

---

## 2. requirements.txt

```
numpy>=1.24,<2.0
opencv-python>=4.8
scipy>=1.11
torch>=2.0
torchvision>=0.15
mmcv>=2.0
mmpose>=1.0
mmdet>=3.0
mmengine>=0.8
gymnasium>=0.29
stable-baselines3>=2.1
dtw-python>=1.3
pyyaml>=6.0
matplotlib>=3.7
tqdm>=4.65
```

---

## 3. Module Interface Definitions

### Module 1: video_loader.py
**职责**: 读取视频文件，按目标FPS抽帧，resize

```
class VideoLoader:
    __init__(video_path: str, fps: int, resize_width: int, resize_height: int)
    extract_frames() -> Generator[tuple[int, np.ndarray]]
        # yields (frame_index, BGR_frame)
    extract_all_frames() -> np.ndarray
        # returns shape (N, H, W, 3)
```

**输入**: 视频文件路径
**输出**: 逐帧BGR图像序列

---

### Module 2: pose_estimator.py
**职责**: 对每帧图像运行HRNet，输出关节点坐标

```
class PoseEstimator:
    __init__(model_name: str = "hrnet_w48", device: str = "cuda")
    estimate_single(frame: np.ndarray) -> dict[str, tuple[float, float, float]]
        # returns {joint_name: (x, y, confidence)} for one frame
    estimate_batch(frames: np.ndarray) -> list[dict]
        # returns list of per-frame joint dicts
```

**输入**: BGR图像 (H, W, 3)
**输出**: 下肢关节点坐标字典，key为关节名（left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle），value为(x, y, confidence)

---

### Module 3: feature_extractor.py
**职责**: 从关节点坐标计算角度、角速度、对称性等特征

```
class FeatureExtractor:
    __init__(fps: int)
    compute_angle(p1, p2, p3: tuple) -> float
        # 三点计算夹角（degrees），p2为顶点
    extract_frame_features(joints: dict) -> dict[str, float]
        # returns {angle_name: value, symmetry: value, ...} for one frame
    extract_sequence_features(joint_sequence: list[dict]) -> np.ndarray
        # returns shape (N_frames, N_features) feature matrix
        # features: left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
        #           knee_symmetry, hip_symmetry, left_knee_angular_velocity, ...
```

**输入**: 关节点坐标序列 (来自Module 2)
**输出**: 特征矩阵 (N_frames, N_features)，每行是一帧的特征向量

---

### Module 4: dtw_aligner.py
**职责**: 将待评估序列与标准模板做时间对齐，分割蹬腿周期

```
class DTWAligner:
    __init__(template_path: str)
    load_template() -> np.ndarray
        # loads champion reference feature sequence
    segment_kick_cycles(features: np.ndarray) -> list[np.ndarray]
        # detects individual kick cycles from continuous sequence
        # returns list of per-cycle feature matrices
    align(query: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]
        # returns (aligned_query, aligned_template, dtw_distance)
    compute_phase_deviations(aligned_query, aligned_template) -> dict[str, float]
        # returns {phase_name: deviation_score} for each kick phase
        # phases: recovery, catch, power, glide
```

**输入**: 待评估特征序列 + 模板特征序列
**输出**: 对齐后的序列对、DTW距离、各阶段偏差分数

---

### Module 5: rl_env.py (CORE)
**职责**: 自定义Gymnasium环境，RL agent在此学习评分

```
class BreaststrokeKickEnv(gymnasium.Env):
    observation_space: Box  # 对齐后的偏差特征向量
    action_space: Discrete(20)  # 20个分数档位，每档5分 (0-100)

    __init__(template_path: str, reward_weights: dict)

    reset() -> tuple[np.ndarray, dict]
        # loads a random kick cycle, computes deviation from template
        # returns (observation, info)
        # observation = flattened deviation features for entire kick cycle

    step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]
        # action = score bin (0-19, maps to 0-100)
        # reward = -|predicted_score - reference_score|
        #   where reference_score is computed from:
        #     angle_deviation * 0.40
        #     symmetry * 0.20
        #     timing * 0.20
        #     smoothness * 0.20
        # episode ends after one action (single-step episode)
        # returns (obs, reward, terminated=True, truncated=False, info)

    _compute_reference_score(deviations: dict) -> float
        # weighted combination of deviation metrics -> 0-100 score
```

**State空间**: 一个完整蹬腿周期的偏差特征向量（对齐后各帧各角度与模板的差值，加上对称性和timing指标）
**Action空间**: Discrete(20)，对应0-100分的20个档位
**Reward**: 预测分数与参考分数的负绝对误差
**Episode**: 单步——agent看完一个周期的特征，输出一个分数，episode结束

---

### Module 5b: train_agent.py
**职责**: 训练RL agent的脚本

```
def build_env(config: dict) -> BreaststrokeKickEnv
def train(config: dict) -> None
    # uses Stable-Baselines3 PPO
    # saves model to models/ directory
def evaluate_agent(model_path: str, test_videos: list[str]) -> dict
    # loads trained model, runs on test set, returns metrics
```

---

### Module 6: feedback.py
**职责**: 根据偏差分析生成自然语言改进建议

```
class FeedbackGenerator:
    __init__()
    generate(score: int, phase_deviations: dict, feature_deviations: dict) -> list[str]
        # returns list of actionable feedback strings
        # e.g. ["Power phase: knee extension insufficient, try to fully extend legs during kick"]
        # prioritized by deviation magnitude
```

**输入**: 评分 + 各阶段/各特征的偏差
**输出**: 按优先级排列的改进建议列表

---

### Pipeline: pipeline.py
**职责**: 串联所有模块的端到端流程

```
class EvaluationPipeline:
    __init__(config_path: str, model_path: str)
    build_template(video_path: str) -> None
        # Module 1 -> 2 -> 3 -> save to templates/
    score_video(video_path: str) -> dict
        # Module 1 -> 2 -> 3 -> 4 -> 5 -> 6
        # returns {"score": int, "phase_scores": dict, "feedback": list[str]}
```

---

## 4. Data Flow

```
Video (.mp4)
    │
    ▼
[Module 1: VideoLoader] ──── frames (N, H, W, 3)
    │
    ▼
[Module 2: PoseEstimator] ── joints list[dict{name: (x,y,conf)}]
    │
    ▼
[Module 3: FeatureExtractor] ── features (N_frames, N_features)
    │
    ▼
[Module 4: DTWAligner] ── aligned deviations + phase scores
    │
    ▼
[Module 5: RL Environment] ── score (0-100)
    │
    ▼
[Module 6: FeedbackGenerator] ── feedback strings
```

---

## 5. Key Design Decisions

1. **Single-step RL episode**: Agent receives full kick cycle features, outputs one score. Not frame-by-frame.
2. **Discrete action space**: 20 bins × 5 points = 0-100 range. Simpler than continuous regression.
3. **Reference score as reward signal**: Weighted combination of angle deviation (0.40), symmetry (0.20), timing (0.20), smoothness (0.20).
4. **DTW for alignment**: Handles different kick speeds between swimmers.
5. **Phase-aware scoring**: Kick split into recovery/catch/power/glide, power phase weighted highest (0.40).
6. **HRNet via MMPose**: Higher accuracy than MediaPipe, justified for academic paper.
