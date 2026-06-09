"""端到端评估流水线：串联模块 1–6，对蛙泳视频/特征打分并生成建议。

数据流（architecture.md §Pipeline）：

    视频 → VideoLoader → PoseEstimator → FeatureExtractor → DTWAligner 周期分割
    → 逐周期 BreaststrokeKickEnv 取观测 + 参考分 → PPO 模型预测分数
    → FeedbackGenerator 生成方向性建议

设计取舍：
    - 核心 :meth:`score_features` 从**特征矩阵**工作，不依赖 mmpose，便于离线复用
      已抽好的 ``.npy``（如教练效度对比的样本）与单元测试；
    - :meth:`score_video` 惰性导入姿态估计后调用 ``score_features``，使本模块在
      未安装 mmpose 的环境下仍可导入；
    - 逐周期通过 :class:`BreaststrokeKickEnv` 的**公共接口**（``reset``）取观测与
      参考分，不触碰其内部缓存，避免与已测试代码耦合。
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional

import numpy as np
import yaml

from src.dtw_aligner import DTWAligner
from src.feature_extractor import FeatureExtractor
from src.feedback import FeedbackGenerator
from src.kick_env import BreaststrokeKickEnv

logger = logging.getLogger(__name__)


_DEFAULT_CONFIG_PATH: str = "configs/config.yaml"
_SCORE_PER_ACTION: int = 5  # 与 kick_env 动作档位一致（action × 5 = 分数）


def load_config(config_path: str) -> dict[str, Any]:
    """读取 YAML 配置。

    Raises:
        FileNotFoundError: 配置文件不存在。
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件 {config_path} 顶层应为映射，得到 {type(cfg)}")
    return cfg


class EvaluationPipeline:
    """端到端评估流水线。"""

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG_PATH,
        model_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ) -> None:
        """初始化流水线。

        Args:
            config_path: ``config.yaml`` 路径。
            model_path: 训练好的 PPO 模型 ``.zip``；``None`` 时取配置中的
                ``paths.model_output``。模型采用**惰性加载**，仅在打分时才读取，
                因此仅做 :meth:`build_template` 时无需模型存在。
            template_path: 冠军模板 ``.npy``；``None`` 时取配置 ``paths.template``。

        Raises:
            FileNotFoundError: 配置或模板文件不存在。
        """
        self.cfg = load_config(config_path)
        paths = self.cfg.get("paths", {})
        video_cfg = self.cfg.get("video", {})
        pose_cfg = self.cfg.get("pose", {})
        fb_cfg = self.cfg.get("feedback", {})

        self.template_path: str = template_path or paths.get(
            "template", "data/templates/champion_template.npy"
        )
        self.model_path: str = model_path or paths.get(
            "model_output", "models/ppo_kick_scorer.zip"
        )

        self.fps = int(video_cfg.get("fps", 30))
        self.min_confidence = float(pose_cfg.get("min_confidence", 0.3))

        self.aligner = DTWAligner(self.template_path)
        self.extractor = FeatureExtractor(
            fps=self.fps, min_confidence=self.min_confidence
        )
        self.feedback_gen = FeedbackGenerator(
            deviation_threshold_deg=float(
                fb_cfg.get("deviation_threshold_deg", 10.0)
            ),
            max_items=int(fb_cfg.get("max_items", 5)),
        )
        self._model: Any = None  # 惰性加载

    # ------------------------------------------------------------------ #
    # 模型（惰性）
    # ------------------------------------------------------------------ #

    def _get_model(self) -> Any:
        """惰性加载 PPO 模型。

        Raises:
            FileNotFoundError: 模型文件不存在（需先训练）。
        """
        if self._model is None:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"模型文件不存在: {self.model_path}。请先运行 train_agent.py 训练，"
                    f"或在构造 EvaluationPipeline 时传入正确的 model_path。"
                )
            from stable_baselines3 import PPO  # 惰性导入，避免无谓依赖

            self._model = PPO.load(self.model_path)
            logger.info("已加载 PPO 模型 %s", self.model_path)
        return self._model

    # ------------------------------------------------------------------ #
    # 打分（核心：从特征矩阵）
    # ------------------------------------------------------------------ #

    def score_features(self, features: np.ndarray) -> dict[str, Any]:
        """对一段连续特征矩阵打分（先分割周期，再逐周期评估）。

        Args:
            features: shape ``(N, 10)`` 的特征矩阵。

        Returns:
            ``{"score": int, "n_cycles": int, "cycles": [per-cycle dict...]}``。
            每个 per-cycle dict 含 ``agent_score`` / ``reference_score`` /
            ``phase_deviations`` / ``feedback``。``score`` 为各周期 agent 分均值
            （四舍五入到 5 的倍数）。

        Raises:
            ValueError: 无法从特征中分割出任何蹬腿周期。
        """
        cycles = self.aligner.segment_kick_cycles(features)
        if not cycles:
            raise ValueError(
                "未能从特征中分割出任何蹬腿周期；请检查输入是否包含完整、规律的"
                "蹬腿动作，或调整分割参数。"
            )

        model = self._get_model()
        per_cycle: list[dict[str, Any]] = []
        agent_scores: list[float] = []
        for idx, cycle in enumerate(cycles):
            # 用单周期 env 的公共接口取观测与参考分（reset 必返回该唯一周期）
            env = BreaststrokeKickEnv([cycle], self.aligner, seed=0)
            obs, info = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            agent_score = float(int(action) * _SCORE_PER_ACTION)

            # 有符号阶段×关节偏差 → 方向性反馈
            aligned_q, aligned_t, _ = self.aligner.align(cycle)
            signed = self.aligner.compute_signed_phase_feature_deviations(
                aligned_q, aligned_t
            )
            feedback = self.feedback_gen.generate(agent_score, signed)

            agent_scores.append(agent_score)
            per_cycle.append(
                {
                    "cycle_index": idx,
                    "cycle_length": int(cycle.shape[0]),
                    "agent_score": agent_score,
                    "reference_score": float(info["target_score"]),
                    "phase_deviations": dict(info["phase_deviations"]),
                    "feedback": feedback,
                }
            )

        mean_score = float(np.mean(agent_scores))
        overall = int(round(mean_score / _SCORE_PER_ACTION) * _SCORE_PER_ACTION)
        return {
            "score": overall,
            "n_cycles": len(per_cycle),
            "cycles": per_cycle,
        }

    def score_npy(self, features_path: str) -> dict[str, Any]:
        """从已抽取的 ``.npy`` 特征文件打分。"""
        if not os.path.isfile(features_path):
            raise FileNotFoundError(f"特征文件不存在: {features_path}")
        return self.score_features(np.load(features_path))

    def score_video(self, video_path: str) -> dict[str, Any]:
        """对原始视频打分：M1→2→3 抽特征后调用 :meth:`score_features`。"""
        features = self._video_to_features(video_path)
        return self.score_features(features)

    # ------------------------------------------------------------------ #
    # 模板构建
    # ------------------------------------------------------------------ #

    def build_template(
        self, video_path: str, out_path: Optional[str] = None
    ) -> np.ndarray:
        """从冠军视频构建**单周期**模板并保存，同时更新本实例的对齐模板。

        Args:
            video_path: 冠军视频路径。
            out_path: 输出 ``.npy``；``None`` 时覆盖配置中的模板路径。

        Returns:
            选中的单周期模板特征矩阵。

        Raises:
            ValueError: 无法从视频分割出蹬腿周期。
        """
        target_path: str = out_path if out_path is not None else self.template_path
        features = self._video_to_features(video_path)
        cycles = self.aligner.segment_kick_cycles(features)
        if not cycles:
            raise ValueError(
                f"无法从 {video_path} 分割出蹬腿周期，模板构建失败。"
            )
        lengths = [c.shape[0] for c in cycles]
        rep = cycles[int(np.argsort(lengths)[len(lengths) // 2])]
        DTWAligner.save_template(rep, target_path)
        # 更新当前实例使用的模板，便于构建后立即打分
        self.aligner.template = rep
        logger.info(
            "已从 %d 个周期中选中位长度单周期 (%d 帧) 写入 %s",
            len(cycles),
            rep.shape[0],
            target_path,
        )
        return rep

    # ------------------------------------------------------------------ #
    # 内部：视频 → 特征（惰性加载姿态估计）
    # ------------------------------------------------------------------ #

    def _video_to_features(self, video_path: str) -> np.ndarray:
        """M1→2→3：抽帧 → HRNet 关节点 → 特征矩阵。惰性导入 mmpose 相关依赖。"""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        import mmcfg_bridge  # noqa: F401 — patch Config.fromfile 路径重定向
        from src.pose_estimator import PoseEstimator, PoseEstimationError
        from src.video_loader import VideoLoader

        video_cfg = self.cfg.get("video", {})
        pose_cfg = self.cfg.get("pose", {})

        loader = VideoLoader(
            video_path,
            fps=self.fps,
            resize_width=int(video_cfg.get("resize_width", 640)),
            resize_height=int(video_cfg.get("resize_height", 480)),
        )
        try:
            frames = loader.extract_all_frames()
        finally:
            loader.release()
        if frames.shape[0] == 0:
            raise ValueError(f"视频 {video_path} 抽帧为空")

        device = str(pose_cfg.get("device", "mps"))
        try:
            estimator = PoseEstimator(
                model_name=str(pose_cfg.get("model_name", "hrnet_w48")),
                device=device,
                min_confidence=self.min_confidence,
            )
            try:
                estimator.estimate_single(frames[0])
            except PoseEstimationError:
                pass  # 探针帧无人体不算设备故障
        except Exception as exc:  # noqa: BLE001 — 设备不可用时回退 CPU
            logger.warning(
                "姿态估计设备 %s 不可用 (%s)，回退 CPU", device, type(exc).__name__
            )
            estimator = PoseEstimator(
                model_name=str(pose_cfg.get("model_name", "hrnet_w48")),
                device="cpu",
                min_confidence=self.min_confidence,
            )

        joint_sequence = estimator.estimate_batch(frames)
        return self.extractor.extract_sequence_features(joint_sequence)


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="蛙泳蹬腿评估端到端流水线：对视频/特征打分或构建模板。"
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH, help="config.yaml 路径")
    parser.add_argument("--model", default=None, help="PPO 模型 .zip 路径（打分用）")
    parser.add_argument("--video", default=None, help="待评估视频路径")
    parser.add_argument("--features", default=None, help="待评估特征 .npy 路径")
    parser.add_argument(
        "--build_template", default=None, help="从该冠军视频构建单周期模板"
    )
    return parser


def main(args: argparse.Namespace) -> int:
    """CLI 入口：构建模板，或对视频/特征打分。"""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    pipeline = EvaluationPipeline(config_path=args.config, model_path=args.model)

    if args.build_template:
        rep = pipeline.build_template(args.build_template)
        print(f"已构建单周期模板，shape={rep.shape}")
        return 0

    if args.video:
        result = pipeline.score_video(args.video)
    elif args.features:
        result = pipeline.score_npy(args.features)
    else:
        print("请提供 --video / --features / --build_template 之一")
        return 1

    print(f"\n=== 总评分: {result['score']} | 周期数: {result['n_cycles']} ===")
    for c in result["cycles"]:
        print(
            f"\n周期 {c['cycle_index']} (长度 {c['cycle_length']} 帧) | "
            f"agent={c['agent_score']:.0f} reference={c['reference_score']:.0f}"
        )
        for line in c["feedback"]:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(_build_arg_parser().parse_args()))
