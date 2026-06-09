"""PPO 训练脚本：从 SwimXYZ 蛙泳数据训练蹬腿评分 RL 智能体。

端到端流水线：

    SwimXYZ 关节序列 → FeatureExtractor → DTWAligner 周期分割
    → BreaststrokeKickEnv（训练集）→ PPO 训练 → 保存模型
    → 在验证集上评估 reward / MAE
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, cast

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES
from src.feature_extractor import FeatureExtractor
from src.kick_env import BreaststrokeKickEnv
from src.swimxyz_loader import SwimXYZLoader


logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH: str = "configs/config.yaml"

# 以下为 config.yaml 缺对应键时的回退默认值；正常运行时超参数由 config 提供
# （见 config.yaml 的 training 小节，对应 EE §3.6“PPO 超参数全部来自 config.yaml”）。
_TRAIN_RATIO: float = 0.8
_N_VAL_EPISODES: int = 100
_PPO_DEFAULTS: dict[str, float] = {
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
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
        labels_dir: SwimXYZ ``Breaststroke_labels`` 目录。
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

    # 设计不变量：DTW 模板必须是“单个蹬腿周期”。compute_phase_deviations 按四分位切
    # recovery/catch/power/glide、kick_env 的 timing(warp) 因素都以模板长度为基准，
    # 均假设模板≈一个周期。但 extract_template.py 保存的是整段冠军视频特征（多周期）。
    # 若拿整段模板对齐单个周期：timing 因素恒≈0（失效）、单周期被强行铺到多周期模板上，
    # 实测参考分数最多偏移约 30 分。此处兜底：模板明显超过单周期长度上界时，就地切出一个
    # 代表性（中位长度）单周期替换；不改写用户磁盘上的模板文件。
    if aligner.template.shape[0] > _MAX_CYCLE_FRAMES:
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


def _evaluate(
    model: PPO,
    val_cycles: list[np.ndarray],
    aligner: DTWAligner,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """在验证集上跑 ``n_episodes`` 个 episode，返回 reward / 误差统计。"""
    if not val_cycles:
        logger.warning("验证集为空，跳过评估")
        return {"mean_reward": float("nan"), "mae_score": float("nan")}

    env = BreaststrokeKickEnv(val_cycles, aligner, seed=seed)
    rewards: list[float] = []
    abs_errors: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(int(action))
        rewards.append(float(reward))
        abs_errors.append(
            abs(float(info["target_score"]) - float(info["agent_score"]))
        )

    stats = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mae_score": float(np.mean(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
    }
    return stats


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

    labels_dir = args.labels_dir or paths.get("swimxyz_labels_dir")
    if not labels_dir:
        raise ValueError(
            "未提供 SwimXYZ 标注目录：请用 --labels_dir 指定，"
            "或在 config.yaml 的 paths.swimxyz_labels_dir 中配置。"
        )
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
    n_val_episodes = int(training.get("n_val_episodes", _N_VAL_EPISODES))
    ppo_n_steps = int(ppo_cfg.get("n_steps", _PPO_DEFAULTS["n_steps"]))
    ppo_batch_size = int(ppo_cfg.get("batch_size", _PPO_DEFAULTS["batch_size"]))
    ppo_n_epochs = int(ppo_cfg.get("n_epochs", _PPO_DEFAULTS["n_epochs"]))
    ppo_lr = float(ppo_cfg.get("learning_rate", _PPO_DEFAULTS["learning_rate"]))
    logger.info(
        "配置 %s | PPO: n_steps=%d batch=%d epochs=%d lr=%g | "
        "total_timesteps=%d train_ratio=%.2f n_val=%d",
        args.config,
        ppo_n_steps,
        ppo_batch_size,
        ppo_n_epochs,
        ppo_lr,
        total_timesteps,
        train_ratio,
        n_val_episodes,
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

    train_env = BreaststrokeKickEnv(train_cycles, aligner, seed=seed)

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

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        n_epochs=ppo_n_epochs,
        learning_rate=ppo_lr,
        verbose=1,
        seed=seed,
    )
    output_parent = os.path.dirname(os.path.abspath(output_model))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    # 验证集非空时挂 EvalCallback：周期性在 val 上评估、保存 best model，
    # 便于监控收敛并避免只靠固定步数训练而过 / 欠拟合不自知。
    callback = None
    if val_cycles:
        eval_env = Monitor(
            BreaststrokeKickEnv(val_cycles, aligner, seed=seed + 1)
        )
        best_dir = os.path.join(output_parent or ".", "best")
        eval_freq = max(ppo_n_steps, total_timesteps // 10)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=best_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_val_episodes,
            deterministic=True,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(output_model)
    logger.info("已保存模型 %s", output_model)

    stats = _evaluate(
        model,
        val_cycles=val_cycles,
        aligner=aligner,
        n_episodes=n_val_episodes,
        seed=seed + 1,
    )
    logger.info(
        "验证集评估 (%d episodes) | mean_reward=%.4f | MAE=%.2f | max_err=%.2f",
        n_val_episodes,
        stats["mean_reward"],
        stats["mae_score"],
        stats.get("max_abs_error", float("nan")),
    )
    print(
        "Validation summary | "
        f"mean_reward={stats['mean_reward']:.4f} | "
        f"std_reward={stats.get('std_reward', float('nan')):.4f} | "
        f"MAE={stats['mae_score']:.2f} | "
        f"max_abs_error={stats.get('max_abs_error', float('nan')):.2f}"
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
        help="SwimXYZ Breaststroke_labels 目录；缺省时取 config paths.swimxyz_labels_dir",
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
    return parser


if __name__ == "__main__":
    cli_args = _build_arg_parser().parse_args()
    main(cli_args)
