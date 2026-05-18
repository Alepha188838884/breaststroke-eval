# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

IB Extended Essay 项目：**蛙泳蹬腿评估系统（Breaststroke Kick Evaluation System）**。使用计算机视觉（姿态估计）+ DTW时间对齐 + 强化学习，对蛙泳蹬腿动作进行自动评分并生成改进建议。

## 技术栈

- **语言**: Python
- **姿态估计**: HRNet (hrnet_w48) via MMPose + MMDet + MMEngine
- **特征对齐**: DTW (dtw-python)
- **强化学习**: Stable-Baselines3 PPO, Gymnasium
- **计算框架**: PyTorch
- **视觉处理**: OpenCV, NumPy, SciPy

## 架构（6模块流水线）

详见 `architecture.md`。数据流：

```
视频 → VideoLoader(抽帧) → PoseEstimator(HRNet关节点) → FeatureExtractor(角度/对称性)
→ DTWAligner(模板对齐+周期分割) → RL Environment(PPO评分) → FeedbackGenerator(建议)
```

关键设计决策：
- RL采用单步episode：agent看完整个蹬腿周期特征，输出一个分数
- 离散动作空间：20档 × 5分 = 0-100分
- 参考分数权重：角度偏差0.40、对称性0.20、时序0.20、平滑度0.20
- 蹬腿分为4阶段：recovery/catch/power/glide，power阶段权重最高(0.40)

## 项目结构

代码计划放在 `breaststroke-eval/` 下，配置在 `configs/config.yaml`。`文献/` 目录存放EE相关文献和注释书目。
