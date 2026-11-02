# Table 1. Feature-extraction and repair parameters

| Parameter | Value | Role |
|---|---|---|
| Sampling rate | 30 fps | frame extraction rate (VideoLoader) |
| Frame size | 640 × 480, aspect-preserving letterbox | input to HRNet detector |
| Min keypoint confidence | 0.3 | joints below threshold treated as missing |
| Feature vector | 10-dim | 4 joint angles + 2 symmetry terms + 4 angular velocities |
| Knee angle | included angle hip–knee–ankle | left and right |
| Hip angle | included angle mid-shoulder–hip–knee | left and right |
| Symmetry terms | \|left − right\| of knee and hip angles | bilateral asymmetry |
| Degenerate-angle guard | collinear / zero-length joint vectors → NaN | frame dropped rather than fabricating an angle |
| Angular-velocity Δt | true frame-index difference × (1/fps) | robust to dropped frames |
| Repair: spike detection | median filter, kernel 5 | locates single-frame spikes |
| Repair: spike threshold | > 40° deviation from local median | flags HRNet failure frames |
| Repair: method | linear interpolation across flagged frames | recovers otherwise-lost cycles |

*Source: `configs/config.yaml` (`video`, `pose`, `feature_extraction`) and module constants in `src/feature_extractor.py` (`_POSE_SMOOTH_KERNEL`, `_POSE_SPIKE_DEG`, `FEATURE_NAMES`, `_ANGLE_GROUPS`).*
