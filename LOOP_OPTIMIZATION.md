# LOOP_OPTIMIZATION.md — 蛙泳打腿评估 EE 项目

> 循环优化审计 · 2026-07-12

## 项目概览与阶段

IB Extended Essay 配套的蛙泳打腿 RL 评分系统（HRNet 姿态 → 特征 → PPO 蒸馏评分），**活跃开发但接近收尾**：代码侧 43 测试全绿、baseline ladder 已可复现（PPO 4.44 ≈ MLP 上界 4.49），剩余工作以论文写作和效度验证为主。本地 pytest 齐全但无 CI。

## 循环机会映射

| 任务 | 当前手动流程 | 推荐循环类型 | 具体命令 / prompt | 停止条件 | token 成本备注 |
|---|---|---|---|---|---|
| 改代码后的回归验证 | 人工跑 pytest、baseline_ladder、pipeline 冒烟三件套 | **skill 内脚本** | 新建 `.claude/skills/verify/SKILL.md`：conda 环境用 `/opt/anaconda3/envs/breaststrokeEval/bin/python`（numpy<2 约束）；顺序跑 ① `pytest`（43 passed 基线）② `scripts/baseline_ladder.py` 数字对表（constant 10.79 / kNN 5.70 / MLP 4.49 / PPO 4.44）③ `python3 -m src.pipeline --features … --model models/ppo_kick_scorer_kinetics.zip` 冒烟 | 一次性建设 | 三步全是确定性脚本，写进 skill 后验证零推理成本；EE 数字被论文引用，回归漂移必须能被机器发现 |
| 数值实验重跑（改特征/env 后） | 人工重跑 03+04 脚本 + 重训 PPO + 全量 eval | **/goal** | `/goal 应用 <某改动> 后重建 features/training 并重训（--source kinetics，本地约 50s），完成判据：43 pytest 全绿且 val MAE ≤ 4.6（4 种子 4.45±0.10 的合理带内）` | MAE 达标或 turn cap 15 | 训练仅 ~50s（obs 预计算），循环代价低；MAE 阈值来自 STATE.md 实测的种子方差，不是拍脑袋 |
| Colab GPU 重跑 scripts/02（letterbox 修复落地） | 挂起中：需人在 Colab 操作 | 不建议循环 | 一次性人工任务 | — | 外部 GPU 会话无法从本机编排；跑完后用上面的 verify skill 验证下游 |
| 教练盲评回收 | export_blind_eval 导出 CSV，等教练填完人工跑 --compare | 不建议循环 | 收到 CSV 后一次性 `scripts/export_blind_eval.py --compare` | — | 外部人的时钟（教练），频率一次性，轮询无意义 |

## 优先级排名

**第一优先：verify skill。** 理由：论文将直接引用 baseline ladder 的数字（10.79/5.70/4.49/4.44），而 2026-07-06 已经发生过一次数字误记（"MLP 4.02"订正事件）。把"数字对表"固化成脚本，是防止写作期间代码微调悄悄改变论文数据的唯一机器保障。项目临近收尾，不值得建更重的循环。

## 前置条件

1. verify skill 里的三条基线数字先手工跑一遍确认当前仍复现。
2. conda 环境约束（numpy<2、mmcv/chumpy --no-build-isolation）写入 skill，避免未来重装环境时验证链断裂。
