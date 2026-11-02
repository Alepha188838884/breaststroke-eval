# Table 2. PPO hyperparameters

| Hyperparameter | Value |
|---|---|
| total_timesteps | 160,000 |
| n_steps | 1024 |
| batch_size | 128 |
| n_epochs | 15 |
| learning_rate | 5 × 10⁻⁴ |
| ent_coef | 0.01 |
| net_arch | [256, 256] |
| seed | 42 |
| Observation space | 28-dim (precomputed cycle features) |
| Action space | 21 discrete actions × 5 points = scores 0–100 |
| Evaluation protocol | full deterministic enumeration of all validation cycles (no sampling) |

*Source: `configs/config.yaml` (`training.ppo`); consumed by `src/train_agent.py`. The entropy coefficient and widened network are the anti-collapse configuration discussed in Section 4.1.*
