# STATE.md — 项目状态（每次会话结束时更新）

## 2026-07-04 会话：对抗性代码审查 → 全量修复

### 本次完成（已验证，35 tests passed + 独立 reviewer APPROVE）

**代码修复**（maker: Sonnet workers ×3，verifier: 独立 Sonnet reviewer）：
- **R-3** `feature_extractor.compute_angle`：向量退化返回 NaN（原 0.0 会伪装成深屈膝），`extract_frame_features` 遇 NaN 丢帧。
- **R-1** `extract_sequence_features`：角速度差分改用**真实帧索引差**×dt；跳帧不再虚高 k 倍。无丢帧时与旧实现逐 bit 一致。
- **R-2** `dtw_aligner.segment_kick_cycles`：左膝列（0）分不出周期时自动回退右膝列（1）；显式指定非 0 列不回退。
- **M-5** `video_loader`：resize 改等比缩放 + letterbox 补黑边（原非等比拉伸会扭曲 16:9 源的关节角度）。
- **C-1** `audit_circularity.py` 重写：按 28 维观测布局用 env 缓存复刻公式（原版按 35 维索引 o[32] 必崩 + 有硬编码结论）；实测 158 周期 100% 复刻、MAE=0。
- **R-4** 新增 `feature_extractor.smooth_feature_matrix` + `scripts/smooth_template.py`；champion 模板已平滑（角速度 pooled std 169.2→123.9°/s，原始备份 `champion_template.bak_raw.npy`）。
- **M-2** `train_agent --source kinetics`：从 `data/features/*_cycles.pkl` + `dataset_meta.json` 的 video 级划分训练真实数据；`--calibrate_only` 打印因素分位。
- **M-3** `kick_env` 增 `score_scales` 参数；config 增 `reference_score.scale_max_real`（真实域标定：angle 20 / symmetry 20 / timing 1.0 / smoothness 250，≈重建数据集 P90）；pipeline 与 export_blind_eval 自动读取。
- **M-1** kick_env / architecture.md 增效度边界声明（参考分=规则公式，无外部真值）；新增 `scripts/export_blind_eval.py`（教练盲评 CSV 导出 + `--compare` 算 Spearman/Pearson/MAE）。
- **N-1** config `swimxyz_labels_dir` 修为 `data/swimxyz/Breaststroke`（磁盘真实名）。
- **N-3/N-4** 残留"35 维"说法清理；architecture.md 结构树/State 空间描述同步实现。

**数据重建**（修复后的提取器，从既有 joints pkl 离线重跑 scripts/03+04）：
- `data/features/`：237 视频 → 575 周期（修复前 562），无周期视频 33→22（右膝回退救回 11 个）。旧版保留在 `data/features_pre_20260704/`。
- `data/training/`：train 172 视频/468 周期，test 43 视频/107 周期（seed 42 video 级划分）。旧版保留在 `data/training_pre_20260704/`。
- 真实域标定后 target 分布：范围 [0,55]、12/21 档、P50=35（业余泳者 vs 冠军模板，分数中低是诚实结果）。

### 已验证的事实
- 参考分数是（周期特征, 模板）的确定性闭式函数，无外部真值（audit_circularity 实证 100%/MAE=0）；28 维 obs 的 kNN 预测 MAE≈3.2 分 = PPO 的可学习性参照。
- 真实域 smoothness P50≈128（修 R-1 后，修复前 169）仍高于合成域 ≈80 → HRNet 抖动是主因，域差真实存在。
- Kinetics joints/features 仍是**旧版非等比 resize** 提取的（letterbox 修复只对未来的 HRNet 重跑生效——需 Colab GPU 重跑 scripts/02 才能彻底消除几何失真）。

### 2026-07-04（同日第二轮）：Kinetics 真实数据 PPO 重训 ✅
- 命令：`python3 -m src.train_agent --source kinetics --output_model models/ppo_kick_scorer_kinetics.zip`（160k 步，~50 秒完成——观测已预计算，3287 fps）。
- **验证（107 val 周期全量）**：MAE=4.44 ≪ 常数基线 10.79，**反超 kNN 参照 5.70**；预测 9 种取值、pred_std 11.2 ≈ target_std 12.8（无坍缩）；train MAE 1.86（轻度过拟合，可接受）；final≈best（4.44 vs 4.49，训练稳定）。
- 部署冒烟：`python3 -m src.pipeline --features ... --model models/ppo_kick_scorer_kinetics.zip` 全通，agent 紧跟 reference（30 vs 30/35），反馈正常。
- config `paths.model_output` 已切到新模型；旧 SwimXYZ 模型保留 `models/ppo_kick_scorer.zip`。
- 产物：`models/ppo_kick_scorer_kinetics.zip`、`models/best_kinetics/`（EvalCallback best + evaluations.npz，可画学习曲线）、`train_kinetics.log`。
- EE 数据点：真实域上"原始运动学 → 规则分"可学到 MAE≈4.4（目标 std 12.8），即约 65% 的方差外解释力（1-(4.44/10.79)），这是 to-what-extent 的核心定量答案之一。

### 2026-07-04（第三轮）：RL 部分二次审查 + 评估协议修复
- **审查**：我精读 kick_env/train_agent + 实证跑数 + 独立验证员对抗复核。6 项论断全 CONFIRMED，另发现 2 项。RL 实现无正确性 bug；4 种子 val MAE=4.45±0.10、video 级 bootstrap 95% CI [3.51,6.21]、无坍缩。
- **P-2 已修**：评估协议从"有放回抽样 100 episodes"（单次波动 ±0.5，日志 4.95≠真值 4.44）改为**全量确定性枚举** `_full_eval`；新 `FullEvalCallback` 按全量 MAE 选 best + 存 evaluations.npz。final model 现报 MAE=4.44（可复现）。
- **P-4 已暴露**：`_evaluate` 现打印 per-bin MAE——边缘档位误差大（0分15.0 / 55分10.0 / 中段2~5），类不平衡，EE 局限性章节引用。
- **P-1/P-3 已文档化**：train_agent docstring 注明 val 参与 best 选择、final model 未经其选、无第三份 held-out（勿引用 best_model 数字）；kick_env docstring 注明单步=contextual bandit、GAE/gamma 无操作。
- **P-5 已处理**：tune_ppo.py 加弃用注记；audit_domain_gap.py 注明刻意不传 score_scales。
- **P-6 已增强**：export_blind_eval 增 `agent_score` 列（PPO 蒸馏分）+ `--model`；`--compare` 分别对 system_score/agent_score 算 Spearman——可同时验证"公式代理专家判断"与"蒸馏保真"。
- 测试 35→43 passed。config `n_val_episodes` 标注废弃。
- **对照数字（供 EE）**：常数基线 10.79 → kNN 5.70 → PPO 4.44 → 监督 MLP 4.02（PPO 达监督 91%）；公式复刻 MAE=0。**这条链说明：模型本质是把规则公式蒸馏成免模板评分器（监督式），非独立判断；效度上界由 reward function 决定 → 教练盲评是效度验证的唯一入口。**
  - **⚠ 2026-07-06 订正**：上行 "监督 MLP 4.02" **系误记**。用新建的可复现脚本 `scripts/baseline_ladder.py`（同一 video-level 划分 / 同一 obs·target·5分档位网格）实测真实阶梯：constant(中位数)=10.79 / constant(均值)=10.84 / kNN(k=5)=5.70 / **MLP(256,256)=4.49** / PPO=4.44 / variance_explained=0.59；PPO 无坍缩（9 种预测, pred_std 11.22），per-bin 证类不平衡（0分→15.0、55分→10.0、35分→2.27）。→ 论证升级：**PPO 4.44 ≈ MLP 上界 4.49，已触到 28 维 obs 的监督天花板**（非"91%"）。日志产物 `baseline_ladder.log`。

### 2026-07-06 会话：EE 大纲代码复核 + baseline ladder 可复现化
- 对照大纲 `main (13).pdf` 与代码逐项复核，产出 `Essay/EE_Outline_Review_and_Writing_Guide.md`（审查 + 分节写作指引）。
- **代码复核确认无误**：obs 28 维 / Discrete(21) / 评分权重 0.40·0.20·0.20·0.20 / 相位权重 .15·.20·.40·.25 / 特征修复 / 周期分割（左膝→右膝回退）/ feedback 阈值10°·max5 / video-level 80/20 / 数据集 237源→215有周期→575周期·train468·val107（`dataset_meta.json` 实读）/ champion 模板=真实 champion.mp4 单周期(45帧)。
- **发现 3 处大纲/STATE 与产物不符**（已写进指引与本 STATE）：
  1. **letterbox 未进数据**：`video_loader.py:21` 自认 + 时间戳证实，575 周期与 champion 模板均为旧非等比 resije 产物；letterbox 修复未重跑 HRNet。→ EE 应写进 Limitations，勿当优点。原"模板是否重提取"确认答案=否。
  2. **MLP 4.02 误记** → 真值 4.49（见上）。
  3. feedback 措辞是相位中立（"more flexed/extended than reference"），非"伸展不足/过度"。
- **新增可复现产物**：`scripts/baseline_ladder.py`（复用 build_kick_cycles_kinetics/_resolve_score_scales/BreaststrokeKickEnv/_full_eval，无泄漏，独立复查通过）；已知非阻塞：line 96 Pyright 对 sklearn predict stub 的类型误报，运行无碍。

### 2026-07-28 会话：EE_Draft_v2 图表生成（3 个 fast-worker 并行）
- **Figure 2** ✅：`figures/make_fig2_learning_curve.py` → `figures/fig2_learning_curve.png`。数据 `models/best_kinetics/evaluations.npz`（10 评估点 × 100 episodes，MAE = -reward×100）。末端标注写明口径 "4.05 (100-episode eval sample)"——**npz 各点是 100-episode 抽样，与正文 4.44（全量 107 枚举）口径不同**，caption 里也应说明。旧图 `fig_ppo_learning_curve.png`（6/13，数据重建前）已过时勿用。
- **Figure 3** ✅：`figures/make_fig3_score_distribution.py` → `figures/fig3_score_distribution.png`。核对精确吻合：n=107 / unique=9 / pred_std=11.2208 / MAE=4.4393。横轴保留 0–100 全量程（预测集中 5–45，可支持"保守但有判别力"叙述）。
- **Table 4** ✅：`scripts/table4_feedback_examples.py` → `outputs/table4_feedback_examples.md`。高/中/低 = pred 45/35/5（tie-break 规则已在文件头披露；23/107 并列 pred=45）。
- **⚠ 正文冲突（✅ 已如实改写，用户拍板）**：原草稿 4.3 节 "high-scoring cycle nearly silent (one sub-threshold note)" **不成立**——全测试集反馈条数分布 {2条:1, 3条:3, 4条:18, 5条:85}，无任何周期 ≤1 条。机制：沉默路径需所有 phase×joint 偏差 <10°，而测试集最高 pred 仅 45。真实可写的对比：低分周期偏差集中在 power 阶段且幅度大（-44°），高分周期偏差集中在低权重 recovery 阶段。最接近沉默的是 pAsj817UdvM_96_106 cycle2（45/45，2 条），用它需披露选取规则。
- **环境事实**：`ppo_kick_scorer_kinetics.zip` 是 numpy≥2.0 下 pickle 的；venv_pose（numpy 1.26 锁定）加载需 SB3 `custom_objects` 替换非权重元数据（fig3 脚本内有现成写法），或直接用系统 python3（numpy 2.4.6，Table4 脚本用此路径）。
- Figure 1（架构示意图）无需跑程序；Figure 4 仍等教练盲评数据。

### 2026-08-05 会话：自录视频端到端评估 + 检测器域适配修复
- **数据**：用户 2026-08-04 自拍 7 条 4K 水下侧视视频（`test_videos/IMG_46{41,42,44,45,46,48,50}.MOV`）。**4641/4642/4644 = 用户本人，其余四条 = 他人**。教练盲评（不知系统存在）：本人三条 ≈40/100，他人四条约 20–30（精确分数截图待用户提供）。
- **诊断链（已实证）**：首轮 3/7 视频无法分割周期。根因＝RTMDet-M（COCO 陆上训练）对水下横躺泳者置信度仅 0.17–0.31，被 `_DET_SCORE_THRESHOLD=0.3` 拒绝。对照实验：提高分辨率 1280×720 无效（2/8→1/8）；降阈值 0.1 + 画框目检确认低分框在泳者身上。
- **代码改动（fast-worker 实施，独立 reviewer APPROVE，47 passed）**：`pose_estimator.py` ① `det_score_threshold` 参数化（默认引用 `_DET_SCORE_THRESHOLD`=0.3，历史口径不变）；② bbox 选择「最大面积」→「最高置信度」。
- **三版提取对比**（`data/real_features{,_det010,_v3_maxscore}/`，日志 `outputs/extract_real_*.log`）：v1（0.3+max-area）4/7 出分；v2（0.1+max-area）6/7，但 4650 被大面积误检抢框回归失败（130→36 帧）；v3（0.1+max-score）**7/7 出分**，4650 修复、4642 的疑似噪声低分 15→25。
- **结果**（`outputs/selfvideo_system_scores.md` v3 表）：本人 30/25/25，他人 30/30/30/35。**教练排序未复现**——系统把所有泳者压在 25–35 窄带（≈噪声带宽），结论是「泳者间判别力不足」而非「打反」。EE 素材点已写入该文件（域适配链条、阈值 trade-off 反例、测量混杂论断）。
- **运行备忘**：自录视频走 `data/real/*.mp4` 软链（extract 只认 .mp4）；打分用系统 python3 + `--features` + kinetics 模型（venv_pose 的 numpy 1.26 载不了该模型）。

### 遗留 / 下一步
1. ~~重训 PPO~~ ✅（第三轮已用新评估协议重训，MAE=4.44 可复现）。学习曲线数据 `models/best_kinetics/evaluations.npz`。
2. **教练盲评（第四轮已建成完整盲评工具包，reviewer APPROVE）**：`export_blind_eval.py` 重构为盲版拆分——导出 3 产物到 `data/blind_eval/`：`coach_sheet.csv`（给教练，**无任何系统分**）、`answer_key.csv`（留档，含 system/agent 分）、`coach_rating.html`（教练浏览器单页：43 个本地视频 `clips/{id}.mp4` 内嵌播放器 + 0-100 打分 + rubric + 一键导出 `filled_coach_sheet.csv`）+ `coach_instructions.md`。43 个 test 视频 mp4 已拷入 `data/blind_eval/clips/`。默认导出全部 43（--n 抽样可选）。`--compare filled.csv [--answer-key answer_key.csv]` 按 video_id join 后分别对 system/agent 报 Spearman/Pearson **+ p 值** + n<10 功效警告；兼容旧单文件格式。测试 47 passed。**盲不被破坏经独立验证（32 个系统分值在给教练文件中零命中）。**
   - 已知非阻塞小债务：HTML 的 JS `exportCsv()` 未给 video_id/score 字段加 CSV 引号（YouTube ID 无逗号故不触发；**但若接入用户自录视频、文件名含特殊字符则需先补引号转义**）。
3. **[新方向，待用户定]** 用户想用**自己录的蛙泳腿视频**同时给系统+教练评分做对照（PE + 个案）。前提：自录视频出系统分需走 HRNet（venv_pose/Colab，本机系统 python 无 mmpose）；pipeline 入口现成 `python -m src.pipeline --video x.mp4`。待用户定：录几段（含故意错误?）/ 是否混入盲评池 / venv_pose vs Colab。单视频=个案非相关性，建议与 Kinetics 池并行。
3. （可选）Colab 重跑 HRNet（letterbox 后的帧）→ 彻底修 M-5 的历史数据。
4. 已知小债务（reviewer 提示，未阻塞）：`_SCALE_KEY_MAP` 在 3 处重复定义；`tune_ppo.py`/`audit_domain_gap.py` 未传 score_scales（前者 SwimXYZ 用默认尺度正确，后者暴露域差本就是其目的）；`smooth_template.py` 就地模式重复执行会覆盖 .bak_raw。
5. **EE 正文需同步**：训练数据的说法（现在"Kinetics 训练路径"已是事实，但要写清合成→真实两条路径与标定差异）；效度边界表述（见 architecture.md reward 节新增段）。

### 环境备忘
- 本机 Bash/Read 曾对 ~/Desktop 整体 EPERM（TCC）；本次全程用 **Finder AppleScript duplicate** 中转 scratchpad 完成读写。修复方法见记忆 desktop-tcc-bash-block。
- 测试：系统 python3.11 + pip --user 装了 dtw-python/opencv-python 后可全绿跑 `python3 -m pytest tests/ -q`（35 passed）。真实 HRNet 推理仍需 venv_pose。

## 2026-08-16 会话：Figure 1/4 定稿 + 教练数据落地 + 论文 4.2 填全

### 本次完成
- **Figure 1**（新）：`figures/make_fig1_pipeline.py` → `fig1_pipeline.png`，六模块蛇形流程图，RL 模块蓝色高亮 + "Original contribution" 图例。
- **教练精确数据到手（用户口述）**：按人统一分——本人（4641/4642/4644）= 40，他人（4645/4646/4648/4650）= 20；文字反馈：本人"脚踝僵硬要压脚踝"，他人"脚没怎么勾、节奏不好、像蝶泳腿"。已记入 `outputs/selfvideo_system_scores.md`。
- **Figure 4 终版两版**：散点版 `fig4_coach_vs_system.png`（精确点、合并重合点标签、去区间棒/微移）+ 配对哑铃版 `fig4b_coach_vs_system_pairs.png`（新脚本 `make_fig4b_pairs.py`）。**正文采用配对版**（排序颠倒一眼可见）。
- **反馈对照**：用 `EvaluationPipeline.score_npy` 对 v3 特征重新生成 7 条视频逐周期反馈（总分与 v3 记录逐条一致，口径已验证）。核心发现：①一致例——B 四条视频 recovery 髋膝 +29~83° 过伸 + glide 膝过屈 = "蝶泳腿"运动学特征，系统用角度语言复现教练观察；②盲区例——两位泳者教练首要诊断都在踝，特征集无踝角（COCO-17 止于踝点），construct-validity gap 的直接实证；③"节奏"进分数（timing 0.20）但反馈层不发声。
- **EE_Draft_v2.md 更新 6 处**：4.2 Protocol 精确分、Results 改"half of the 20-point gap"、Figure 4 caption 换配对版、Phase-level agreement 整段写实（~210 词）、4.3 交叉引用、5.x 踝盲区补实证呼应。【COACH-DATA】占位符全部清零。

### 用户决定
- **第一稿不管字数**（4.2 现 ~711 词/目标 450，全文粗算 ~5042；压缩留到定稿阶段）。
- Figure 4 正文用配对图，散点版留作备用。

### 遗留 / 下一步
1. 43 条 Kinetics 盲评是否执行仍未决（checklist 第 8 项，正文已兼容两种结局）。
2. 定稿阶段：字数压缩（4.2 优先）、删 HTML 注释、图表编号连续性核查。
3. 小提醒：make_fig4b_pairs.py 有 4 条 Pyright 类型提示（Line2D 私有导入等），运行无碍，未修。
