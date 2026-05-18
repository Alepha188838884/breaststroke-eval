# Extended Essay Detailed Outline

**Title:** Reinforcement Learning-Based Evaluation of Breaststroke Kick Technique Using Pose Estimation

**Research Question:** Reinforcement Learning-Based Evaluation of Breaststroke Kick Technique Using Pose Estimation

**Subject:** Computer Science

**Word Budget:** ~4000 words (hard cap)

---

## Assessment Criteria Reference (Current, through Nov 2026)

| Criterion | What examiners look for | Marks | Where it lives in your essay |
|-----------|------------------------|-------|------------------------------|
| A: Focus & Method | Clear RQ, justified methodology, maintained focus | 6 | Introduction + Methodology |
| B: Knowledge & Understanding | CS terminology, conceptual depth, source integration | 6 | Lit Review + Methodology |
| C: Critical Thinking | Analysis, evaluation, argument development, reasoned conclusions | 12 | Data Analysis + Limitations |
| D: Presentation | Structure, formatting, referencing, figures/tables | 4 | Throughout |
| E: Engagement | RPPF reflections (not in the essay itself) | 6 | RPPF document |

---

## 1. Introduction (~200 words)

**Purpose:** Establish context, state the RQ, and preview the approach.

### Content to include:
- **Hook / Context** (~50 words): Breaststroke kick is the most technically complex swimming kick; small deviations cause drag and injury. Traditional coaching relies on subjective observation. Computer vision + ML offer objective, scalable alternatives.
- **Gap identification** (~50 words): Existing pose-estimation systems for swimming focus on classification (stroke type recognition) or binary good/bad judgment rather than continuous technique quality scoring. No prior work combines RL-based scoring with DTW-aligned biomechanical features for breaststroke kick evaluation.
- **Research Question** (~30 words): State the RQ clearly and precisely. This is what Criterion A directly assesses.
- **Approach preview** (~70 words): Briefly outline the pipeline (pose estimation → feature extraction → DTW alignment → RL scoring → feedback) and state the core contribution: the custom RL environment and trained PPO agent that learns a scoring policy from biomechanical features, validated against expert coach evaluations.

### Criteria mapping:
- **Criterion A**: RQ stated, topic justified, scope defined
- **Criterion B**: Demonstrates awareness of the field

---

## 2. Literature Review (~300 words)

**Purpose:** Demonstrate knowledge of relevant prior work and position your contribution within the field.

### Content to include (3 clusters, ~100 words each):

#### 2.1 Pose Estimation in Aquatic Environments
- HRNet architecture and why it outperforms MediaPipe for underwater/poolside scenarios (cite: "Pose estimation for swimmers in video surveillance" — HRNet W48 achieving >97% on elbow/wrist)
- SwimmerNET's markerless approach and its 1mm average error
- SwimXYZ synthetic dataset demonstrating feasibility of training pose models for swimming
- **Your position:** You adopt HRNet via MMPose as the pose backbone, chosen for its multi-resolution feature fusion suited to partial occlusion in water

#### 2.2 RL in Sports Training & Feedback
- RL-CWtrans Net: RL agent for swimming coaching using Swin-Transformer + CLIP (most directly relevant)
- DRL-driven personalized training load optimization (DQN framework for sports performance)
- Cheerleading evaluation system using RL for curriculum difficulty adjustment
- **Your position:** Unlike these systems that use RL for action selection or load optimization, your RL agent learns a *scoring policy* — mapping biomechanical features to technique quality scores

#### 2.3 Action Quality Assessment (AQA) & Technique Scoring
- "3D Pose Based Feedback for Physical Exercises" — GCN-based error detection + correction
- "Interpretable two-stage AQA" — contrastive learning for scoring diving/gymnastics
- "Dynamic golf swing analysis framework" — phase-segmented DTW comparison (DMSM)
- Freestyle swimming posture analysis system — binary good/bad classification
- **Your position:** Your approach extends beyond binary classification to continuous scoring via RL, using DTW phase alignment to preserve temporal structure — a gap in current swimming-specific literature

### Criteria mapping:
- **Criterion B**: Subject-specific terminology, conceptual understanding, source integration
- **Criterion C**: Identifying gaps, positioning your work (analysis begins here)

---

## 3. Methodology (~1500–2000 words)

**Purpose:** Describe and justify every technical decision in the pipeline. This is where you earn Criterion A (method justification) and Criterion B (technical depth) simultaneously.

### 3.1 System Architecture Overview (~150 words)
- Present the 6-module pipeline diagram: VideoLoader → PoseEstimator → FeatureExtractor → DTWAligner → RL Environment → FeedbackGenerator
- State clearly: the RL environment is your core contribution; other modules leverage established tools (HRNet, DTW library, Stable-Baselines3)
- Justify the modular design: each module has a single responsibility, enabling independent testing and fail-closed error propagation

### 3.2 Video Processing & Pose Estimation (~200 words)
- VideoLoader: FPS-normalized frame extraction with letterbox resize (preserving aspect ratio for pose model input)
- PoseEstimator: HRNet-W48 via MMPose, top-down approach with RTMDet person detector
- **Justify HRNet over MediaPipe:** multi-resolution parallel branches maintain spatial precision; critical for detecting knee/ankle angles in poolside video where limbs may be partially submerged
- Confidence gating: min_confidence threshold (0.3) on individual joints; frames with <4 valid lower-body joints are discarded (fail-closed design)
- mid_shoulder virtual keypoint computed from left/right shoulder for hip angle calculation

### 3.3 Feature Extraction (~250 words)
- 10-dimensional feature vector per frame:
  - 4 joint angles: left/right knee (hip-knee-ankle), left/right hip (mid_shoulder-hip-knee)
  - 2 symmetry metrics: |left_knee - right_knee|, |left_hip - right_hip|
  - 4 angular velocities: central difference with gap-adjusted dt (correcting for dropped frames)
- **Justify feature selection:** These features directly encode the biomechanical criteria coaches assess — knee flexion/extension range, hip drive, bilateral symmetry, and kick tempo
- Dropped-frame handling: angular velocity computed using actual inter-frame time gap, not fixed dt, preventing artificial velocity amplification when frames are skipped

### 3.4 DTW Alignment & Cycle Segmentation (~300 words)
- **Template construction:** Extract features from world champion breaststroke video → segment kick cycles via peak detection on averaged knee angle signal → select cycle closest to median length as template
- **Cycle detection algorithm:** find_peaks on knee angle signal with configurable prominence and min/max cycle length constraints; fallback to valley-based segmentation for single-cycle videos
- **DTW alignment:** z-score normalization for path computation (equalizing feature scales), but aligned sequences returned in original physical units (preserving interpretability for scoring)
- **Phase segmentation:** Each aligned cycle divided into 4 phases — Recovery (0–25%), Catch (25–50%), Power (50–75%), Glide (75–100%) — with phase-specific RMSE computed only on angle columns (excluding angular velocity to avoid unit mixing)
- **Justify DTW over simple interpolation:** DTW handles tempo variations between swimmers without distorting the biomechanical structure of the kick cycle

### 3.5 RL Environment Design — Core Contribution (~400 words)
- **Framing as an RL problem:**
  - **State (observation):** For each kick cycle, compute a fixed-length observation vector from aligned query-template differences: per-feature statistics (mean, std, max, min) × 10 features + 4 phase deviations + DTW distance + 2 smoothness metrics (jerk RMS, velocity std) = 47 dimensions
  - **Action:** Discrete(20) — agent selects one of 20 bins, mapped to 0–100 score via (action + 0.5) × 5.0
  - **Reward:** Negative absolute error between predicted score and reference score: R = −|predicted − reference|
- **Reference score computation (the "ground truth" the agent learns to approximate):**
  - Weighted composite of 4 sub-scores:
    - Angle deviation (weight 0.40): RMSE of first 4 feature columns vs template
    - Symmetry (weight 0.20): mean absolute symmetry deviation
    - Timing (weight 0.20): phase-weighted deviation across Recovery/Catch/Power/Glide
    - Smoothness (weight 0.20): jerk magnitude from angular velocities
  - Each sub-score linearly mapped from raw metric to 0–100 via configurable thresholds
- **Justify PPO:** Stable training via clipped surrogate objective; handles discrete action spaces well; widely validated in Gymnasium environments; Stable-Baselines3 provides production-quality implementation
- **Justify Discrete(20) over continuous:** Technique scores are inherently categorical judgments (excellent/good/average/poor); discrete bins align with how coaches think; avoids regression instability in early training
- **Episode structure:** Single-step episodes (observe one cycle → output score → receive reward → episode ends). This is deliberate: the agent's task is per-cycle scoring, not sequential decision-making
- **Fail-closed design principles:** observation dimension validated at both training and inference time; missing required keys in cycle data raise explicit errors rather than defaulting to zeros

### 3.6 Training Configuration (~150 words)
- PPO hyperparameters: learning_rate, n_steps, batch_size, n_epochs, gamma, clip_range (all from config.yaml)
- Training data: kick cycles extracted from multiple swimmer videos, each stored as .npy with aligned sequences, phase deviations, DTW distance, and raw query
- Train/eval split via EvalCallback: best model saved based on evaluation reward
- Hardware: Google Colab T4 GPU

### 3.7 Feedback Generation (~100 words)
- Rule-based feedback triggered by signed (directional) per-phase feature deviations
- Positive deviation = angle exceeds template; negative = insufficient
- Phase-specific rules with configurable thresholds (e.g., "Power phase: knee extension insufficient" if knee angle < −10° from template)
- Bilateral coverage: rules check both left and right limbs independently

### Criteria mapping:
- **Criterion A**: Methodology clearly described and justified; scope maintained
- **Criterion B**: Demonstrates deep technical understanding; correct use of CS terminology (RL, PPO, DTW, observation space, reward shaping, etc.)
- **Criterion C**: Justification of design decisions = critical thinking throughout

---

## 4. Data Analysis (~1000–1200 words)

**Purpose:** Present results, analyze them critically, and develop your argument. This section carries the most weight for Criterion C (12 marks).

### 4.1 Training Performance (~200 words)
- Learning curve: plot mean reward over training timesteps
- Convergence behavior: how quickly does the agent's score error decrease?
- Final mean absolute error (MAE) on evaluation episodes (agent vs reference score during training)
- Distribution of predicted scores across evaluation cycles (histogram)

### 4.2 Validation Against Expert Coach Scores (~450 words)
**This is the core analysis.**
- **Data collection:** School swimming coach watches the same video clips used in the pipeline and assigns each kick cycle a technique score (0–100). This provides an independent human ground truth.
- **Correlation analysis:** Compare RL agent scores vs coach scores — Pearson/Spearman correlation, scatter plot, MAE
- **Per-phase analysis:** For cycles where the coach flags specific phase weaknesses (e.g., "insufficient knee extension in power phase"), does the agent's phase deviation data agree? Present specific examples with aligned numbers.
- **Failure case analysis:** Identify cycles where agent and coach scores disagree most. Investigate what features caused the agent to misjudge — was it a pose estimation error propagating through, a DTW alignment issue, or a genuine limitation of the feature set?
- **Key argument:** If the RL agent achieves strong correlation with coach scores, this demonstrates that the pipeline successfully encodes biomechanically meaningful information and the RL agent learns a scoring policy that generalizes beyond the rule-based reference it was trained on.

### 4.3 Feedback Quality Analysis (~200 words)
- Select 2–3 example cycles (high/medium/low score) and present the generated feedback
- Cross-reference with coach's verbal comments on the same cycles: does the system identify the same issues the coach identified?
- Assess whether the directional (signed) deviations produce actionable corrections (e.g., "knee extension insufficient in Power phase" when the coach also notes incomplete leg drive)

### 4.4 Feature Contribution Analysis (~200 words)
- Which observation features show the strongest separation between high-scoring and low-scoring cycles?
- Analyze observation vectors: which statistical dimensions (mean, std, max, min of angle/symmetry/velocity differences) best predict the coach's score?
- Relate back to biomechanics: does feature importance align with coaching priorities (e.g., Power phase angle deviation being the strongest predictor)?

### Criteria mapping:
- **Criterion C**: This entire section is critical thinking — analysis, interpretation, evaluation of evidence, reasoned argument. Every claim must be supported by data.
- Present figures and tables clearly labeled (Criterion D)

---

## 5. Limitations and Evaluation (~700 words)

**Purpose:** Evaluate the investigation's strengths, weaknesses, and implications. This is where you demonstrate mature academic judgment (Criterion C, top band).

### 5.1 Methodological Limitations (~300 words)
- **Training signal vs validation signal gap:** The RL agent is trained on a rule-based reference score (weighted composite of angle deviation, symmetry, timing, smoothness), but validated against coach scores. If the rule-based reference poorly represents what coaches actually value, the agent may learn a suboptimal policy. Discuss how the correlation results in Section 4.2 quantify this gap.
- **Single-template dependency:** DTW alignment against a single world champion template assumes that champion technique is the universal gold standard. Different body proportions may produce biomechanically valid but visually different kick patterns. Coach scores may partially account for this, but the feature extraction remains template-dependent.
- **2D pose limitation:** HRNet produces 2D keypoints from a single camera view. Depth information is lost, meaning out-of-plane movements (e.g., knee width during breaststroke catch) cannot be captured. 3D pose estimation or multi-camera setups would address this.
- **Sample size:** Training data derived from a limited number of swimmer videos; coach validation performed on a small set of cycles. Statistical significance of correlation results should be interpreted cautiously.
- **Single-step episodes:** The RL environment uses single-step episodes, meaning the agent cannot learn temporal dependencies across multiple kick cycles (e.g., fatigue-related technique deterioration within a set).

### 5.2 Strengths (~200 words)
- Modular, reproducible pipeline with clear separation of concerns
- Fail-closed design throughout: invalid frames, missing data, dimension mismatches all raise explicit errors
- DTW alignment preserves temporal structure while handling speed variations
- The RL formulation is extensible: reward function can be retrained with expert coach labels when available, without changing the observation pipeline
- Open-source stack (MMPose, Stable-Baselines3, Gymnasium) ensures reproducibility

### 5.3 Implications and Future Work (~200 words)
- **Practical value:** The system demonstrates that an RL agent can learn a scoring policy that aligns with expert coach judgment, enabling scalable, objective technique evaluation without requiring a coach to be present for every session
- **Training directly on coach scores:** Future work could use coach scores as the RL training signal instead of the rule-based reference, potentially improving alignment with expert judgment — this requires a larger labeled dataset
- **Extension to other strokes:** The feature extraction and DTW alignment modules are stroke-agnostic; only the template and phase definitions need updating
- **Real-time deployment:** The pipeline's per-frame inference latency (pose estimation ~25ms on T4 GPU) is compatible with near-real-time feedback during training sessions

### 5.4 Conclusion (~100 words, can be a subsection here or standalone)
- Directly address the RQ: summarize what the system achieves
- State the degree of agreement between RL agent scores and coach evaluations
- Acknowledge the key limitations concisely
- End with the broader significance: RL-based technique evaluation as a viable paradigm for automated, objective sports coaching feedback

### Criteria mapping:
- **Criterion C**: Evaluation of methodology, acknowledging limitations, suggesting improvements — this is where top-band C marks are earned
- **Criterion A**: Conclusions linked back to RQ (maintaining focus)

---

## Word Budget Summary

| Section | Target Words | Cumulative |
|---------|-------------|------------|
| Introduction | 200 | 200 |
| Literature Review | 300 | 500 |
| Methodology | 1500–1700 | 2000–2200 |
| Data Analysis | 1000–1100 | 3000–3300 |
| Limitations & Evaluation | 700 | 3700–4000 |

**Note:** Abstract, table of contents, bibliography, figures/tables, and appendices do NOT count toward the 4000-word limit. Code snippets in appendices are fine but examiners are not required to read them — any critical code logic must be explained in prose within the essay body. Include a ~300-word abstract summarizing the RQ, method, key results, and conclusion.

---

## Presentation Checklist (Criterion D — 4 easy marks)

- [ ] Title page with RQ, subject, word count
- [ ] Abstract (~300 words, not counted in word limit)
- [ ] Table of contents with page numbers
- [ ] Consistent heading hierarchy (numbered sections)
- [ ] All figures/tables numbered, captioned, and referenced in text
- [ ] System architecture diagram (pipeline overview)
- [ ] At least 1 training curve plot, 1 agent-vs-coach correlation scatter plot, 1 comparison table
- [ ] Example feedback outputs for selected cycles
- [ ] In-text citations (APA or IEEE, consistent throughout)
- [ ] Full bibliography
- [ ] Appendix: config.yaml structure, key code excerpts (optional but helpful)
- [ ] Word count stated on title page

---

## RPPF Notes (Criterion E — 6 marks, assessed separately)

The three reflections should cover:
1. **Initial:** Why this topic? Personal motivation from competitive swimming. Why RL for scoring rather than supervised classification? Early RQ formulation challenges.
2. **Interim:** Technical pivots — choosing HRNet over MediaPipe, designing the reward function, realizing the need for independent validation (coach scoring). How adversarial code review rounds shaped the architecture.
3. **Final (viva voce):** What you learned about owning architecture vs delegating implementation. How comparing agent output with coach judgment deepened your understanding of what RL can and cannot learn. Transferable skills: modular design thinking, fail-closed principles.

Use IB language: "engagement," "intellectual challenge," "decision-making," "growth as a learner."
