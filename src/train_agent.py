"""PPO 训练脚本：从 SwimXYZ 蛙泳数据训练蹬腿评分 RL 智能体。

端到端流水线：

    SwimXYZ 关节序列 → FeatureExtractor → DTWAligner 周期分割
    → BreaststrokeKickEnv（训练集）→ PPO 训练 → 保存模型
    → 在验证集上评估 reward / MAE

验证协议说明（P-1）：val 集（kinetics 源下即 dataset_meta 的 test_video_ids）
同时用于训练期监控（:class:`FullEvalCallback`）与 best-model 选择；最终交付/
报告的是**未经 val 选择的 final model**，``best_model.zip`` 仅作训练曲线参考，
勿在论文中引用其数字。项目当前无第三份纯 held-out 集。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Any, Optional, cast

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES, _MIN_CYCLE_FRAMES as _ALIGNER_MIN_CYCLE_FRAMES
from src.feature_extractor import FeatureExtractor
from src.kick_env import BreaststrokeKickEnv
from src.swimxyz_loader import SwimXYZLoader


logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH: str = "configs/config.yaml"

# 以下为 config.yaml 缺对应键时的回退默认值；正常运行时超参数由 config 提供
# （见 config.yaml 的 training 小节，对应 EE §3.6“PPO 超参数全部来自 config.yaml”）。
_TRAIN_RATIO: float = 0.8
_PPO_DEFAULTS: dict[str, float] = {
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 15,
    "learning_rate": 5e-4,
    "ent_coef": 0.01,        # 防坍缩：单步打分任务上 ent_coef=0 会塌成常数预测
}
# 策略网络隐藏层（pi/vf 共用）；较大网络才能拟合 raw-kinematics → score 映射
_PPO_NET_ARCH_DEFAULT: list[int] = [256, 256]

# 蹬腿周期帧数下界，与 dtw_aligner / kick_env 保持一致
_MIN_CYCLE_FRAMES: int = _ALIGNER_MIN_CYCLE_FRAMES  # 10

# config.yaml reference_score.scale_max(_real) 的键 -> kick_env.score_scales 键
_SCALE_KEY_MAP: dict[str, str] = {
    "angle_dev": "angle",
    "symmetry_dev": "symmetry",
    "timing_warp": "timing",
    "smoothness": "smoothness",
}


def _load_config(path: str) -> dict:
    """读取 YAML 配置；文件不存在时抛 FileNotFoundError。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"配置文件不存在: {path}（可用 --config 指定，或确认 configs/config.yaml 存在）"
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _ensure_single_cycle_template(aligner: DTWAligner, template_path: str) -> None:
    """若 ``aligner.template`` 超过单周期帧数上界，就地切出代表性单周期替换。

    设计不变量：DTW 模板必须是"单个蹬腿周期"。compute_phase_deviations 按四分位
    切 recovery/catch/power/glide、kick_env 的 timing(warp) 因素都以模板长度为
    基准，均假设模板≈一个周期。但部分模板文件（如 extract_template.py 早期产出）
    保存的是整段冠军视频特征（多周期）。若拿整段模板对齐单个周期：timing 因素
    恒≈0（失效）、单周期被强行铺到多周期模板上，实测参考分数最多偏移约 30 分。
    此处兜底：模板明显超过单周期长度上界时，就地切出一个代表性（中位长度）单
    周期替换 ``aligner.template``；不改写磁盘上的模板文件。

    Raises:
        RuntimeError: 模板无法分割出任何单个蹬腿周期。
    """
    if aligner.template.shape[0] <= _MAX_CYCLE_FRAMES:
        return
    template_cycles = aligner.segment_kick_cycles(aligner.template)
    if not template_cycles:
        raise RuntimeError(
            f"模板 {template_path} 含 {aligner.template.shape[0]} 帧但无法分割出"
            f"单个蹬腿周期；请用 extract_template.py 重新生成单周期模板。"
        )
    lengths = [c.shape[0] for c in template_cycles]
    rep = template_cycles[int(np.argsort(lengths)[len(lengths) // 2])]
    logger.warning(
        "模板 %s 含 %d 帧（检出 %d 个周期），疑为整段序列而非单周期；已就地切出"
        "中位长度单周期 (%d 帧) 用于对齐（磁盘文件未改动）。",
        template_path,
        aligner.template.shape[0],
        len(template_cycles),
        rep.shape[0],
    )
    aligner.template = rep


def build_kick_cycles(
    labels_dir: str,
    template_path: str,
    fps: int = 30,
    max_sequences: Optional[int] = None,
) -> tuple[list[list[np.ndarray]], DTWAligner]:
    """从 SwimXYZ 标注构建按序列分组的蹬腿周期池 + 加载好模板的 DTWAligner。

    流程：

        1. 用 :class:`SwimXYZLoader` 加载关节序列。
        2. 用 :class:`FeatureExtractor` 转为 10 列特征矩阵。
        3. 若 ``template_path`` 不存在：取第一个序列的特征/首个有效周期作为
           冠军模板并保存。
        4. 用 :class:`DTWAligner` 对每个序列 ``segment_kick_cycles``。
        5. 按 **序列** 分组返回（避免 train/val 切分时跨序列泄露）。

    Args:
        labels_dir: SwimXYZ ``Breaststroke`` 标注目录。
        template_path: 冠军模板 ``.npy`` 路径，不存在时自动生成。
        fps: 视频帧率，用于角速度计算。
        max_sequences: 调试时限制加载的序列数；``None`` 表示全部加载。

    Returns:
        ``(per_sequence_cycles, aligner)`` 元组。
        ``per_sequence_cycles`` 是 list[list[np.ndarray]]，外层按源序列分组，
        内层每项 shape ``(cycle_len, 10)``。

    Raises:
        RuntimeError: 没有任何有效序列或未能分割出任何周期。
    """
    loader = SwimXYZLoader(labels_dir, stroke="breaststroke")
    sequence_paths = loader.list_sequences()
    if not sequence_paths:
        raise RuntimeError(f"未在 {labels_dir} 找到任何标注文件")
    if max_sequences is not None:
        sequence_paths = sequence_paths[:max_sequences]

    extractor = FeatureExtractor(fps=fps)
    all_features: list[np.ndarray] = []
    for path in sequence_paths:
        try:
            joint_seq = loader.load_sequence(path)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("跳过无效标注 %s: %s", path, exc)
            continue
        # extract_sequence_features 接受 list[Optional[dict]]（允许失败帧为 None）；
        # SwimXYZ 合成真值不会缺帧，仅在类型上做兼容性转换。
        joint_seq_opt = cast(
            list[Optional[dict[str, tuple[float, float, float]]]],
            joint_seq,
        )
        feats = extractor.extract_sequence_features(joint_seq_opt)
        if feats.shape[0] == 0:
            logger.warning("序列 %s 无有效帧，跳过", path)
            continue
        all_features.append(feats)

    if not all_features:
        raise RuntimeError("没有任何序列产出有效特征矩阵")
    logger.info(
        "特征提取完成 | 序列数=%d | 总帧数=%d",
        len(all_features),
        sum(f.shape[0] for f in all_features),
    )

    # 模板：不存在则用第一个序列的首个蹬腿周期生成；该序列随后从训练 / 验证池
    # 中剔除，避免 cycle 与模板同源造成的 DTW 自我对齐泄露。
    template_source_idx: Optional[int] = None
    if not os.path.isfile(template_path):
        parent = os.path.dirname(os.path.abspath(template_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        first_feats = all_features[0]
        # 先以整段特征作为临时模板初始化 aligner，再尝试分割出首个周期
        DTWAligner.save_template(first_feats, template_path)
        temp_aligner = DTWAligner(template_path)
        first_cycles = temp_aligner.segment_kick_cycles(first_feats)
        if not first_cycles:
            raise RuntimeError(
                f"模板生成失败：序列 {sequence_paths[0]} 无法分割出有效蹬腿周期。"
                f"请改用 --template_path 指向预生成的冠军模板 .npy 文件，"
                f"或先用更规整的序列引导模板。"
            )
        DTWAligner.save_template(first_cycles[0], template_path)
        template_source_idx = 0
        logger.info(
            "已用第一个序列的首个蹬腿周期生成模板 %s | shape=%s | "
            "该序列将从训练 / 验证池中剔除以避免泄露",
            template_path,
            tuple(first_cycles[0].shape),
        )

    aligner = DTWAligner(template_path)
    _ensure_single_cycle_template(aligner, template_path)

    per_sequence_cycles: list[list[np.ndarray]] = []
    total_cycles = 0
    skipped_template = 0
    for i, feats in enumerate(all_features):
        if i == template_source_idx:
            skipped_template += 1
            continue
        cycles = aligner.segment_kick_cycles(feats)
        if cycles:
            per_sequence_cycles.append(cycles)
            total_cycles += len(cycles)

    if not per_sequence_cycles:
        raise RuntimeError("未能从任何序列中分割出有效蹬腿周期")

    logger.info(
        "共收集到 %d 个蹬腿周期，分布在 %d 个序列（剔除模板源序列 %d 个）",
        total_cycles,
        len(per_sequence_cycles),
        skipped_template,
    )
    return per_sequence_cycles, aligner


def _load_video_cycles(
    video_id: str, features_dir: str
) -> Optional[list[np.ndarray]]:
    """加载单个视频的 ``{video_id}_cycles.pkl``，过滤 shape / 帧数不合法的周期。

    Returns:
        过滤后的周期列表；文件不存在时返回 ``None``（调用方已收到 warning）。
    """
    pkl_path = os.path.join(features_dir, f"{video_id}_cycles.pkl")
    if not os.path.isfile(pkl_path):
        logger.warning("特征文件不存在，跳过视频 %s: %s", video_id, pkl_path)
        return None
    with open(pkl_path, "rb") as f:
        raw_cycles = pickle.load(f)

    valid: list[np.ndarray] = []
    n_dropped = 0
    for c in raw_cycles:
        arr = np.asarray(c)
        if arr.ndim == 2 and arr.shape[1] == 10 and arr.shape[0] >= _MIN_CYCLE_FRAMES:
            valid.append(arr)
        else:
            n_dropped += 1
    if n_dropped:
        logger.info(
            "视频 %s 过滤掉 %d/%d 个不合法周期（shape 非 (n,10) 或帧数 < %d）",
            video_id,
            n_dropped,
            len(raw_cycles),
            _MIN_CYCLE_FRAMES,
        )
    return valid


def build_kick_cycles_kinetics(
    features_dir: str, meta_path: str, template_path: str
) -> tuple[list[np.ndarray], list[np.ndarray], DTWAligner]:
    """从真实 Kinetics 处理产物构建 train / val 蹬腿周期池。

    数据来源：``scripts/02``（姿态估计）与 ``scripts/03``（特征提取 + DTW 周期
    分割）已把每个视频处理为 ``{features_dir}/{video_id}_cycles.pkl``
    （``list[np.ndarray(cycle_len, 10)]``）。train/val 划分**直接沿用**
    ``meta_path`` 里记录的 ``train_video_ids`` / ``test_video_ids``（video 级
    划分），与已发布的 observations 数据集保持同一划分，不在此处重新洗牌。

    Args:
        features_dir: 含 ``*_cycles.pkl`` 的目录（如 ``data/features``）。
        meta_path: ``dataset_meta.json`` 路径，提供 ``train_video_ids`` /
            ``test_video_ids``。
        template_path: 冠军模板 ``.npy`` 路径（须已存在，本函数不会自动生成）。

    Returns:
        ``(train_cycles, val_cycles, aligner)``。

    Raises:
        FileNotFoundError: ``meta_path`` 不存在。
        RuntimeError: train 或 val 侧收集到的周期为空。
    """
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"dataset_meta.json 不存在: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_ids = list(meta.get("train_video_ids", []))
    test_ids = list(meta.get("test_video_ids", []))

    aligner = DTWAligner(template_path)
    _ensure_single_cycle_template(aligner, template_path)

    def _collect(video_ids: list[str]) -> list[np.ndarray]:
        cycles: list[np.ndarray] = []
        for vid in video_ids:
            loaded = _load_video_cycles(vid, features_dir)
            if loaded is not None:
                cycles.extend(loaded)
        return cycles

    train_cycles = _collect(train_ids)
    val_cycles = _collect(test_ids)

    logger.info(
        "Kinetics 数据集 | train_videos=%d train_cycles=%d | "
        "val_videos=%d val_cycles=%d",
        len(train_ids),
        len(train_cycles),
        len(test_ids),
        len(val_cycles),
    )
    if not train_cycles:
        raise RuntimeError(f"train 侧未收集到任何有效周期（features_dir={features_dir}）")
    if not val_cycles:
        raise RuntimeError(f"val 侧未收集到任何有效周期（features_dir={features_dir}）")

    return train_cycles, val_cycles, aligner


def _split_sequences(
    per_sequence_cycles: list[list[np.ndarray]],
    train_ratio: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """**在序列级别** 洗牌后切分 train / val，避免同一视频跨集泄露。

    切分后再 flatten 为 cycle 列表。

    Raises:
        ValueError: 序列总数 < 2（无法形成 train + val）；
            或 ``train_ratio`` 不在 ``(0, 1]`` 范围。

    说明：
        - ``train_ratio == 1.0``：所有序列归 train，``val_cycles`` 返回 ``[]``。
        - ``train_ratio < 1.0``：``min(split, n_seq-1)`` 确保 val 至少 1 个序列。
    """
    n_seq = len(per_sequence_cycles)
    if n_seq < 2:
        raise ValueError(
            f"_split_sequences 需要至少 2 个序列才能构造 train + val，"
            f"得到 {n_seq}。请增加 --max_sequences 或检查数据集。"
        )
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(
            f"train_ratio 必须在 (0, 1] 区间，得到 {train_ratio}"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n_seq)
    rng.shuffle(indices)

    if train_ratio >= 1.0:
        # 用户显式要求全量训练；不强行偷一个序列做 val
        train_cycles_full: list[np.ndarray] = []
        for i in indices:
            train_cycles_full.extend(per_sequence_cycles[int(i)])
        logger.info(
            "train_ratio=1.0，全部 %d 个序列归 train (%d cycles)，val 为空",
            n_seq,
            len(train_cycles_full),
        )
        return train_cycles_full, []

    split = max(1, int(round(train_ratio * n_seq)))
    split = min(split, n_seq - 1)
    train_seq_idx = indices[:split]
    val_seq_idx = indices[split:]

    train_cycles: list[np.ndarray] = []
    for i in train_seq_idx:
        train_cycles.extend(per_sequence_cycles[int(i)])
    val_cycles: list[np.ndarray] = []
    for i in val_seq_idx:
        val_cycles.extend(per_sequence_cycles[int(i)])

    logger.info(
        "序列级 train / val 划分 | 序列总=%d, train_seq=%d (%d cycles), "
        "val_seq=%d (%d cycles)",
        n_seq,
        len(train_seq_idx),
        len(train_cycles),
        len(val_seq_idx),
        len(val_cycles),
    )
    return train_cycles, val_cycles


def _resolve_score_scales(
    cfg: dict, source: str
) -> Optional[dict[str, float]]:
    """按数据来源解析传给 :class:`BreaststrokeKickEnv` 的 ``score_scales``。

    仅 ``source == "kinetics"`` 时生效：读取 config 的
    ``reference_score.scale_max_real``（键 angle_dev/symmetry_dev/timing_warp/
    smoothness）并映射为 kick_env 的 score_scales 键（angle/symmetry/timing/
    smoothness）。未配置时告警并回退默认（None，沿用 SwimXYZ 标定）。
    ``source == "swimxyz"`` 时恒返回 ``None``，行为完全不变。
    """
    if source != "kinetics":
        return None
    scale_max_real = cfg.get("reference_score", {}).get("scale_max_real")
    if not scale_max_real:
        logger.warning(
            "未配置 scale_max_real，真实数据将沿用合成标定，因素可能饱和；"
            "先跑 --calibrate_only 标定"
        )
        return None
    return {
        _SCALE_KEY_MAP[k]: float(v)
        for k, v in scale_max_real.items()
        if k in _SCALE_KEY_MAP
    }


def _full_eval(model: PPO, env: BreaststrokeKickEnv) -> dict[str, Any]:
    """确定性全量枚举评估：对 ``env`` 缓存的每个周期恰好评估一次。

    替代旧的"``env.reset()`` 有放回随机抽样 n_episodes 次"评估协议——旧协议在
    周期池远小于抽样次数时（如 107 个周期抽 100 次）单次实现的 MAE 方差达
    ±0.5~0.7 分量级、且不可复现（同一模型两次运行给出不同数字）。本函数直接
    遍历 ``env._cached_obs``（预计算阶段已为每个周期算好观测与参考分），每个
    周期恰好参与一次评估，结果在同一模型上逐位可复现。

    Args:
        model: 提供 ``predict(obs, deterministic)`` 的策略（PPO 或兼容鸭子类型）。
        env: 已完成预计算的 :class:`BreaststrokeKickEnv`。

    Returns:
        dict，含：

        - ``mae``: 全量周期的平均绝对误差。
        - ``max_abs_error``: 最大绝对误差。
        - ``mean_reward``: ``-mae / 100``，与 env 的 reward 定义
          （``reward = -|agent_score - target_score| / 100``）一致。
        - ``n_unique_predictions``: 预测值的不同取值数（坍缩诊断：为 1 说明
          智能体退化成常数预测）。
        - ``pred_std``: 预测值标准差（坍缩诊断）。
        - ``per_bin_mae``: ``{target 档位(int): 该档位周期的 MAE}``。
    """
    targets = env._cached_target_score
    preds: list[float] = []
    abs_errors: list[float] = []
    for obs, target in zip(env._cached_obs, targets):
        action, _ = model.predict(obs, deterministic=True)
        pred = float(int(action) * 5)
        preds.append(pred)
        abs_errors.append(abs(pred - float(target)))

    preds_arr = np.asarray(preds, dtype=np.float64)
    errors_arr = np.asarray(abs_errors, dtype=np.float64)
    mae = float(np.mean(errors_arr))

    per_bin_errors: dict[int, list[float]] = {}
    for target, err in zip(targets, abs_errors):
        per_bin_errors.setdefault(int(target), []).append(err)
    per_bin_mae = {k: float(np.mean(v)) for k, v in per_bin_errors.items()}

    return {
        "mae": mae,
        "max_abs_error": float(np.max(errors_arr)),
        "mean_reward": float(-mae / 100.0),
        "n_unique_predictions": int(len(set(preds))),
        "pred_std": float(np.std(preds_arr)),
        "per_bin_mae": per_bin_mae,
    }


def _evaluate(
    model: PPO,
    val_cycles: list[np.ndarray],
    aligner: DTWAligner,
    seed: int,
    score_scales: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """在验证集上做全量确定性枚举评估（见 :func:`_full_eval`）。"""
    if not val_cycles:
        logger.warning("验证集为空，跳过评估")
        return {"mean_reward": float("nan"), "mae_score": float("nan")}

    env = BreaststrokeKickEnv(val_cycles, aligner, seed=seed, score_scales=score_scales)
    full_stats = _full_eval(model, env)
    return {
        "mean_reward": full_stats["mean_reward"],
        "mae_score": full_stats["mae"],
        "max_abs_error": full_stats["max_abs_error"],
        "n_unique_predictions": full_stats["n_unique_predictions"],
        "pred_std": full_stats["pred_std"],
        "per_bin_mae": full_stats["per_bin_mae"],
        "n_cycles": len(env.kick_cycles),
    }


class FullEvalCallback(BaseCallback):
    """训练期周期性全量确定性枚举评估 + best-model 保存的回调。

    替代 ``stable_baselines3.EvalCallback`` 的有放回随机抽样评估
    （``n_eval_episodes`` 次 ``env.reset()``）：本回调复用 :func:`_full_eval`，
    每次评估对 ``eval_env`` 缓存的全部周期各评估一次，MAE 数值确定、可复现，
    best-model 选择不再取决于"抽样运气"。
    """

    def __init__(
        self,
        eval_env: BreaststrokeKickEnv,
        eval_freq: int,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.best_mae: float = float("inf")
        self.evaluations: list[tuple[int, float]] = []

    def _run_eval(self) -> dict[str, Any]:
        """跑一次全量评估，打印诊断日志，MAE 改善时保存 best_model.zip。"""
        stats = _full_eval(self.model, self.eval_env)
        self.evaluations.append((self.num_timesteps, stats["mae"]))
        logger.info(
            "FullEvalCallback | step=%d | 全量 %d 周期 MAE=%.2f | "
            "n_unique_predictions=%d | pred_std=%.2f",
            self.num_timesteps,
            len(self.eval_env.kick_cycles),
            stats["mae"],
            stats["n_unique_predictions"],
            stats["pred_std"],
        )
        if stats["mae"] < self.best_mae:
            self.best_mae = stats["mae"]
            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                save_path = os.path.join(self.best_model_save_path, "best_model.zip")
                self.model.save(save_path)
                if self.verbose:
                    logger.info(
                        "FullEvalCallback | 新最优 MAE=%.2f，已保存 %s",
                        self.best_mae,
                        save_path,
                    )
        return stats

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        if self.best_model_save_path and self.evaluations:
            os.makedirs(self.best_model_save_path, exist_ok=True)
            ts = np.asarray([e[0] for e in self.evaluations], dtype=np.int64)
            maes = np.asarray([e[1] for e in self.evaluations], dtype=np.float64)
            np.savez(
                os.path.join(self.best_model_save_path, "evaluations.npz"),
                ts=ts,
                maes=maes,
            )


def main(args: argparse.Namespace) -> None:
    """主训练流程：数据准备 → PPO 训练 → 保存模型 → 验证集评估。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # 加载配置；CLI 显式给出的参数（非 None）覆盖 config 对应项。
    cfg = _load_config(args.config)
    training = cfg.get("training", {})
    ppo_cfg = training.get("ppo", {})
    paths = cfg.get("paths", {})

    source = args.source or training.get("source", "swimxyz")

    template_path = args.template_path or paths.get(
        "template", "data/templates/champion_template.npy"
    )
    output_model = args.output_model or paths.get(
        "model_output", "models/ppo_kick_scorer.zip"
    )
    fps = args.fps if args.fps is not None else int(training.get("fps", 30))
    seed = args.seed if args.seed is not None else int(training.get("seed", 42))
    total_timesteps = (
        args.total_timesteps
        if args.total_timesteps is not None
        else int(training.get("total_timesteps", 50_000))
    )
    train_ratio = float(training.get("train_ratio", _TRAIN_RATIO))
    ppo_n_steps = int(ppo_cfg.get("n_steps", _PPO_DEFAULTS["n_steps"]))
    ppo_batch_size = int(ppo_cfg.get("batch_size", _PPO_DEFAULTS["batch_size"]))
    ppo_n_epochs = int(ppo_cfg.get("n_epochs", _PPO_DEFAULTS["n_epochs"]))
    ppo_lr = float(ppo_cfg.get("learning_rate", _PPO_DEFAULTS["learning_rate"]))
    ppo_ent_coef = float(ppo_cfg.get("ent_coef", _PPO_DEFAULTS["ent_coef"]))
    ppo_net_arch = [int(x) for x in ppo_cfg.get("net_arch", _PPO_NET_ARCH_DEFAULT)]
    logger.info(
        "配置 %s | PPO: n_steps=%d batch=%d epochs=%d lr=%g ent_coef=%g net=%s | "
        "total_timesteps=%d train_ratio=%.2f",
        args.config,
        ppo_n_steps,
        ppo_batch_size,
        ppo_n_epochs,
        ppo_lr,
        ppo_ent_coef,
        ppo_net_arch,
        total_timesteps,
        train_ratio,
    )

    if source == "kinetics":
        features_dir = args.features_dir or paths.get("features_dir", "data/features")
        dataset_meta = args.dataset_meta or paths.get(
            "dataset_meta", "data/training/dataset_meta.json"
        )
        train_cycles, val_cycles, aligner = build_kick_cycles_kinetics(
            features_dir=features_dir,
            meta_path=dataset_meta,
            template_path=template_path,
        )
    elif source == "swimxyz":
        labels_dir = args.labels_dir or paths.get("swimxyz_labels_dir")
        if not labels_dir:
            raise ValueError(
                "未提供 SwimXYZ 标注目录：请用 --labels_dir 指定，"
                "或在 config.yaml 的 paths.swimxyz_labels_dir 中配置。"
            )
        per_sequence_cycles, aligner = build_kick_cycles(
            labels_dir=labels_dir,
            template_path=template_path,
            fps=fps,
            max_sequences=args.max_sequences,
        )
        train_cycles, val_cycles = _split_sequences(
            per_sequence_cycles, train_ratio=train_ratio, seed=seed
        )
    else:
        raise ValueError(f"未知 source: {source!r}，应为 'swimxyz' 或 'kinetics'")

    score_scales = _resolve_score_scales(cfg, source)

    train_env = BreaststrokeKickEnv(
        train_cycles, aligner, seed=seed, score_scales=score_scales
    )

    # 标定诊断：暴露参考分数各因素分布与动作档位利用率。若 n_bins_used 远小于
    # n_bins_total、或某因素 P90 远小于其 *_MAX，说明分数被压缩，可据此下调对应
    # *_MAX 阈值（见 kick_env._ANGLE_DEV_MAX 等）。此处只打印不改评分。
    dist = train_env.factor_distribution()
    ts = dist["target_score"]
    logger.info(
        "标定诊断 | target 档位 %d/%d 被使用, 范围 [%.0f, %.0f]",
        ts["n_bins_used"],
        ts["n_bins_total"],
        ts["min"],
        ts["max"],
    )
    for name, key in (
        ("angle", "angle_overall_deviation"),
        ("symmetry", "symmetry_mean"),
        ("timing", "timing_warp"),
        ("smoothness", "smoothness_vel_std"),
    ):
        logger.info(
            "标定诊断 | %-10s P50=%.2f P90=%.2f max=%.2f  当前 *_MAX=%.2f",
            name,
            dist[key]["p50"],
            dist[key]["p90"],
            dist[key]["max"],
            dist["current_scale_maxes"][name],
        )

    if args.calibrate_only:
        logger.info("--calibrate_only 已开启：仅打印标定诊断，不构建/训练 PPO 模型")
        return

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        learning_rate=ppo_lr,
        ent_coef=ppo_ent_coef,
        policy_kwargs=dict(net_arch=dict(pi=ppo_net_arch, vf=ppo_net_arch)),
        verbose=1,
        seed=seed,
    )
    output_parent = os.path.dirname(os.path.abspath(output_model))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    # 验证集非空时挂 FullEvalCallback：周期性在 val 上做全量确定性枚举评估、
    # 保存 best model，便于监控收敛并避免只靠固定步数训练而过 / 欠拟合不自知。
    # 见模块 docstring：val 集同时用于监控与 best-model 选择，best_model.zip
    # 数字不作为论文交付结果，最终交付的是本函数末尾保存的 final model。
    callback = None
    if val_cycles:
        eval_env = BreaststrokeKickEnv(
            val_cycles, aligner, seed=seed + 1, score_scales=score_scales
        )
        best_dir = os.path.join(output_parent or ".", "best")
        eval_freq = max(ppo_n_steps, total_timesteps // 10)
        callback = FullEvalCallback(
            eval_env,
            eval_freq=eval_freq,
            best_model_save_path=best_dir,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(output_model)
    logger.info("已保存模型 %s", output_model)

    stats = _evaluate(
        model,
        val_cycles=val_cycles,
        aligner=aligner,
        seed=seed + 1,
        score_scales=score_scales,
    )
    n_cycles_eval = stats.get("n_cycles", 0)
    logger.info(
        "验证集评估 (全量 %d 周期枚举) | mean_reward=%.4f | MAE=%.2f | max_err=%.2f",
        n_cycles_eval,
        stats["mean_reward"],
        stats["mae_score"],
        stats.get("max_abs_error", float("nan")),
    )
    print(
        "Validation summary | "
        f"mean_reward={stats['mean_reward']:.4f} | "
        f"MAE={stats['mae_score']:.2f} | "
        f"max_abs_error={stats.get('max_abs_error', float('nan')):.2f}"
    )
    # per_bin_mae：逐档位打印供 EE 引用；n_unique_predictions/pred_std 让坍缩
    # 一目了然（==1 即坍缩成常数预测）。
    for bin_val in sorted(stats.get("per_bin_mae", {})):
        logger.info(
            "验证集评估 | 档位=%d | MAE=%.2f",
            bin_val,
            stats["per_bin_mae"][bin_val],
        )
    logger.info(
        "验证集评估 | n_unique_predictions=%d | pred_std=%.2f",
        stats.get("n_unique_predictions", 0),
        stats.get("pred_std", float("nan")),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Train PPO breaststroke kick scoring agent on SwimXYZ data."
    )
    # 多数参数默认 None：未显式给出时由 config.yaml 提供（见 main 中的解析逻辑）。
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="config.yaml 路径（超参数 / 路径来源；CLI 参数可覆盖其中对应项）",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="SwimXYZ Breaststroke 标注目录；缺省时取 config paths.swimxyz_labels_dir",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default=None,
        help="冠军模板 .npy 路径；缺省时取 config paths.template",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="PPO 训练总步数；缺省时取 config training.total_timesteps",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="调试用：限制加载的序列数",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=None,
        help="训练完成后保存模型的路径；缺省时取 config paths.model_output",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "特征提取帧率（角速度 = Δ角度 × fps）；缺省时取 config training.fps。"
            "注意：SwimXYZ 合成集真实渲染帧率未在标注中给出，30 为假设值；仅影响"
            "smoothness 因素与速度观测维的绝对尺度（train/val 同 fps 故内部自洽）。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；缺省时取 config training.seed",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["swimxyz", "kinetics"],
        help=(
            "训练数据来源；缺省时取 config training.source（缺省 'swimxyz'）。"
            "swimxyz=合成数据（现状默认）；kinetics=真实数据，用 --features_dir + "
            "--dataset_meta 的既有 video 级划分。"
        ),
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help=(
            "source=kinetics 时，含 *_cycles.pkl 的特征目录；"
            "缺省时取 config paths.features_dir"
        ),
    )
    parser.add_argument(
        "--dataset_meta",
        type=str,
        default=None,
        help=(
            "source=kinetics 时，dataset_meta.json 路径（train/test_video_ids）；"
            "缺省时取 config paths.dataset_meta"
        ),
    )
    parser.add_argument(
        "--calibrate_only",
        action="store_true",
        help=(
            "只构建数据与 train_env 并打印标定诊断（各因素 P50/P90/max、"
            "target 档位利用率）后退出，不构建/训练 PPO 模型。用于在真实数据上"
            "标定 reference_score.scale_max_real。"
        ),
    )
    return parser


if __name__ == "__main__":
    cli_args = _build_arg_parser().parse_args()
    main(cli_args)
