"""Table 4：从真实 pipeline 输出中选取高/中/低分三个测试周期的 FeedbackGenerator
生成示例，写成 Markdown 表存到 outputs/table4_feedback_examples.md。

数据/口径与 scripts/baseline_ladder.py 一致：
    - train/val 周期划分：src.train_agent.build_kick_cycles_kinetics（video 级划分，
      val = dataset_meta.json 的 test_video_ids）
    - score_scales：src.train_agent._resolve_score_scales(cfg, "kinetics")
    - PPO 打分：models/ppo_kick_scorer_kinetics.zip，deterministic 预测
    - 反馈生成：与 src/pipeline.py EvaluationPipeline.score_features 相同的调用
      方式 —— aligner.align(cycle) → aligner.compute_signed_phase_feature_deviations
      → FeedbackGenerator.generate(agent_score, signed)

运行（CPU，无需 mmcfg_bridge）：
    python3 scripts/table4_feedback_examples.py
    注：不能用 venv_pose/bin/python —— 该 venv 的 numpy 锁定在 1.26.4（给 mmcv 用），
    而 models/ppo_kick_scorer_kinetics.zip 是在 numpy>=2.0 下 pickle 的，1.26.4 无
    numpy._core 命名空间会导致 PPO.load 抛 ModuleNotFoundError。系统 python3 装了
    numpy 2.4.6 + stable_baselines3，可正常加载。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feedback import FeedbackGenerator
from src.kick_env import BreaststrokeKickEnv
from src.train_agent import (
    _load_video_cycles,
    _resolve_score_scales,
    build_kick_cycles_kinetics,
)

logger = logging.getLogger(__name__)

_SEED = 42
_SCORE_PER_ACTION = 5
_OUT_PATH = ROOT / "outputs" / "table4_feedback_examples.md"


def _build_labels(meta_path: str, features_dir: str) -> list[tuple[str, int, np.ndarray]]:
    """在与 build_kick_cycles_kinetics 相同的顺序下重建 (video_id, local_idx, cycle) 列表，
    用于给 val_cycles 里的每个周期贴上 "video_id#cycle_idx" 标签。
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    test_ids = list(meta.get("test_video_ids", []))

    labeled: list[tuple[str, int, np.ndarray]] = []
    for vid in test_ids:
        loaded = _load_video_cycles(vid, features_dir)
        if loaded is None:
            continue
        for local_idx, cycle in enumerate(loaded):
            labeled.append((vid, local_idx, cycle))
    return labeled


def main(config_path: str = "configs/config.yaml") -> None:
    with open(ROOT / config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    fb_cfg = cfg.get("feedback", {})

    features_dir = str(ROOT / paths.get("features_dir", "data/features"))
    meta_path = str(ROOT / paths.get("dataset_meta", "data/training/dataset_meta.json"))
    template_path = str(ROOT / paths.get("template", "data/templates/champion_template.npy"))
    model_path = str(ROOT / paths.get("model_output", "models/ppo_kick_scorer_kinetics.zip"))

    _, val_cycles_check, aligner = build_kick_cycles_kinetics(
        features_dir=features_dir, meta_path=meta_path, template_path=template_path
    )

    # 独立重建带 (video_id, local_idx) 标签的 val 周期列表，并用它（而非上面
    # build_kick_cycles_kinetics 返回的 val_cycles_check）构建 env/aligner 输入 ——
    # 这样 id(array) 才能在后面匹配上标签（两次从磁盘 load 得到的是不同的 array
    # 对象，即便内容相同）。用长度核对两者确实一致。
    labeled = _build_labels(meta_path, features_dir)
    val_cycles = [c for _, _, c in labeled]
    label_by_id = {id(c): (vid, idx) for vid, idx, c in labeled}
    assert len(val_cycles) == len(val_cycles_check), (
        f"val 周期数不一致: labeled={len(val_cycles)} vs "
        f"build_kick_cycles_kinetics={len(val_cycles_check)}"
    )
    print(f"n_val_cycles = {len(val_cycles)}")

    score_scales = _resolve_score_scales(cfg, "kinetics")
    val_env = BreaststrokeKickEnv(
        val_cycles, aligner, seed=_SEED, score_scales=score_scales
    )

    model = PPO.load(model_path)

    obs_batch = np.asarray(val_env._cached_obs)
    actions, _ = model.predict(obs_batch, deterministic=True)
    pred_scores = np.asarray(actions, dtype=np.float64) * _SCORE_PER_ACTION
    ref_scores = np.asarray(val_env._cached_target_score, dtype=np.float64)

    n = len(pred_scores)

    # PPO 打分是 21 档离散网格（action×5），预测分严重打结（如本次 val 集上
    # pred=45 的周期有 23 个）。argsort 在结里的取法是任意的、不具代表性，
    # 因此对"最高/中位/最低预测分"三组分别打破结，二级 key 的方向与一级 key
    # 保持一致（而非统一取"参考分最高" —— 那样会让 low 组选出 pred=5/ref=25
    # 这种恰好是全测试集最大绝对误差(20)的样本，把"低分示例"变成"最差预测"）：
    #   - max pred 组：组内取参考分最高者（离满分最近的高分样本）
    #   - min pred 组：组内取参考分最低者（离 0 分最近的低分样本）
    #   - median pred 组：组内取参考分与该组参考分中位数最接近者
    # 同分再取原始 index 最小者，保证完全确定性。
    def _pick_extreme(mask: np.ndarray, want_max_ref: bool) -> int:
        cand = np.where(mask)[0]
        sign = 1 if want_max_ref else -1
        return int(max(cand, key=lambda i: (sign * ref_scores[i], -i)))

    def _pick_central(mask: np.ndarray) -> int:
        cand = np.where(mask)[0]
        group_median_ref = float(np.median(ref_scores[cand]))
        return int(
            min(cand, key=lambda i: (abs(ref_scores[i] - group_median_ref), i))
        )

    idx_high = _pick_extreme(pred_scores == pred_scores.max(), want_max_ref=True)
    idx_low = _pick_extreme(pred_scores == pred_scores.min(), want_max_ref=False)
    unique_vals = np.unique(pred_scores)
    median_grid_val = unique_vals[
        np.argmin(np.abs(unique_vals - np.median(pred_scores)))
    ]
    idx_mid = _pick_central(pred_scores == median_grid_val)

    feedback_gen = FeedbackGenerator(
        deviation_threshold_deg=float(fb_cfg.get("deviation_threshold_deg", 10.0)),
        max_items=int(fb_cfg.get("max_items", 5)),
    )

    def label_for(i: int) -> str:
        cyc = val_env.kick_cycles[i]
        key = id(cyc)
        if key in label_by_id:
            vid, local_idx = label_by_id[key]
            return f"{vid} (cycle {local_idx})"
        return f"val_cycle_{i}"

    rows = []
    report_lines = []
    for tag, i in (("high", idx_high), ("middle", idx_mid), ("low", idx_low)):
        cycle = val_env.kick_cycles[i]
        pred = float(pred_scores[i])
        ref = float(ref_scores[i])

        aligned_q, aligned_t, _ = aligner.align(cycle)
        signed = aligner.compute_signed_phase_feature_deviations(aligned_q, aligned_t)
        feedback = feedback_gen.generate(pred, signed)

        cyc_label = label_for(i)
        rows.append((tag, cyc_label, pred, ref, feedback))
        report_lines.append(
            f"[{tag}] {cyc_label}: predicted={pred:.0f} reference={ref:.1f} "
            f"n_feedback={len(feedback)}"
        )
        for line in feedback:
            report_lines.append(f"    - {line}")

    print("\n".join(report_lines))

    # ---- 写 Markdown 表 ----
    md_lines = []
    md_lines.append("# Table 4: FeedbackGenerator output for high/middle/low test cycles")
    md_lines.append("")
    md_lines.append(
        f"Source: PPO model `{Path(model_path).name}`, evaluated deterministically on the "
        f"full Kinetics test set ({n} cycles, video-level train/test split from "
        f"`dataset_meta.json`). The three cycles shown are the highest-, median-, and "
        f"lowest-predicted-score cycles in the test set. Predicted scores are a 21-point "
        f"discrete grid (action×5) and are heavily tied (e.g. 23/107 cycles tie at the "
        f"observed maximum, pred=45). Ties are broken with a secondary key that mirrors "
        f"the direction of the primary key: within the max-pred group, highest reference "
        f"score; within the min-pred group, lowest reference score; within the "
        f"median-pred group, reference score closest to that group's median (index order "
        f"as final tiebreak). A single uniform \"highest ref\" rule was rejected because, "
        f"applied to the min-pred group here, it would have selected pred=5/ref=25 — the "
        f"single largest prediction error (|20|) in the whole test set — as the "
        f"\"low-scoring\" exemplar. Feedback generated by `FeedbackGenerator` (`deviation_threshold_deg="
        f"{fb_cfg.get('deviation_threshold_deg', 10.0)}`, `max_items="
        f"{fb_cfg.get('max_items', 5)}`) from the same signed phase×joint deviations "
        f"used in `src/pipeline.py`'s `score_features`."
    )
    md_lines.append("")
    md_lines.append("| Cycle | Predicted score | Reference score | Generated feedback |")
    md_lines.append("|---|---|---|---|")
    for tag, cyc_label, pred, ref, feedback in rows:
        fb_text = "<br>".join(f"{j+1}. {line}" for j, line in enumerate(feedback))
        md_lines.append(
            f"| {cyc_label} ({tag}) | {pred:.0f} | {ref:.1f} | {fb_text} |"
        )
    md_lines.append("")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {_OUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
