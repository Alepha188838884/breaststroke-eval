# Table 5. Baseline ladder

All models use the identical video-level split, identical 28-dim observations, and the identical rule-based target; predictions are rounded to the 5-point score grid. Metric: MAE on the 107 held-out test cycles.

| Model | Test MAE |
|---|---|
| Constant (train mean = 31.05) | 10.84 |
| Constant (train median = 35) | 10.79 |
| kNN (k = 5, z-normalised) | 5.70 |
| MLP (256, 256, z-normalised) | 4.49 |
| **PPO agent (this project)** | **4.44** |

Variance explained by PPO relative to the mean-constant baseline: 1 − 4.44/10.84 ≈ 0.59. The PPO agent uses 9 distinct score values (prediction SD 11.22), i.e. no prediction collapse.

*Source: `baseline_ladder.log` (`scripts/baseline_ladder.py`), model `models/ppo_kick_scorer_kinetics.zip`.*
