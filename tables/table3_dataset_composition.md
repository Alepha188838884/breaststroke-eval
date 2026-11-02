# Table 3. Dataset composition and split

| Stage | Videos | Kick cycles |
|---|---|---|
| Kinetics-700 breaststroke source clips | 237 | — |
| With ≥ 1 valid segmented cycle (usable) | 215 | 575 |
| Training split | 172 | 468 |
| Held-out test split | 43 | 107 |

Split is at **video level** (80/20, seed 42): all cycles from one video fall on the same side, so no swimmer or clip appears in both training and test. 22 source videos yielded no valid cycle (pose failure or no periodic kick signal).

*Source: `data/training/dataset_meta.json` (n_source_videos 237, n_videos_with_cycles 215, total_cycles 575, train 172/468, test 43/107).*
