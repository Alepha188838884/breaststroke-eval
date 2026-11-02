from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kick_env import BreaststrokeKickEnv
from src.train_agent import _resolve_score_scales, build_kick_cycles_kinetics

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
paths = cfg.get("paths", {})

train_cycles, val_cycles, aligner = build_kick_cycles_kinetics(
    features_dir=paths.get("features_dir", "data/features"),
    meta_path=paths.get("dataset_meta", "data/training/dataset_meta.json"),
    template_path=paths.get("template", "data/templates/champion_template.npy"),
)
score_scales = _resolve_score_scales(cfg, "kinetics")

val_env = BreaststrokeKickEnv(val_cycles, aligner, seed=42, score_scales=score_scales)

model_path = ROOT / "models" / "ppo_kick_scorer_kinetics.zip"
# NOTE: this checkpoint was saved under numpy>=2.0, but venv_pose is pinned to
# numpy<2.0 for mmcv/mmpose ABI compatibility (see requirements.txt). Several
# entries in the SB3 "data" JSON blob (numpy arrays / RNG state) cannot be
# unpickled under numpy 1.26, so they are replaced via custom_objects. None
# of these are needed for deterministic inference with model.predict().
_custom_objects = {
    "observation_space": val_env.observation_space,
    "action_space": val_env.action_space,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "clip_range_vf": None,
    "target_kl": None,
    "tensorboard_log": None,
    "action_noise": None,
    "rollout_buffer_class": None,
    "rollout_buffer_kwargs": {},
    "_last_obs": None,
    "_last_episode_starts": None,
    "_last_original_obs": None,
    "ep_info_buffer": None,
    "ep_success_buffer": None,
}
ppo_model = PPO.load(str(model_path), custom_objects=_custom_objects, device="cpu")

targets = val_env._cached_target_score
preds = []
for obs in val_env._cached_obs:
    action, _ = ppo_model.predict(obs, deterministic=True)
    pred = float(int(action) * 5)
    preds.append(pred)
preds_arr = np.asarray(preds, dtype=np.float64)

# --- Diagnostics ---
n = len(val_cycles)
n_unique = len(np.unique(preds_arr))
pred_std = float(np.std(preds_arr))
mae = float(np.mean(np.abs(preds_arr - np.asarray(targets, dtype=np.float64))))
print(f"CHECK: n={n} unique={n_unique} pred_std={pred_std:.4f} mae={mae:.4f}")

# --- Plot ---
edges = np.arange(-2.5, 105, 5).tolist()

fig, ax = plt.subplots(figsize=(7, 4.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.hist(preds_arr, bins=edges, rwidth=0.9, color="#2563EB")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.yaxis.grid(True, linestyle="--", color="lightgray", alpha=0.7)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.set_xlabel("Predicted score")
ax.set_ylabel("Number of test cycles")
ax.set_xticks(np.arange(0, 101, 10))

ax.annotate(
    f"n = {n} cycles, SD = {pred_std:.1f}",
    xy=(0.97, 0.93),
    xycoords="axes fraction",
    horizontalalignment="right",
    verticalalignment="top",
)

out_path = ROOT / "figures" / "fig3_score_distribution.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")

print(f"Saved figure to: {out_path.resolve()}")
