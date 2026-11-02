"""CLI 工具：对已保存的模板 .npy 做角度平滑，抑制 HRNet 逐帧抖动带来的虚高角速度。

背景：模板角速度的 pooled std 若接近 HRNet 姿态估计噪声量级（实测 champion 模板
≈169°/s，远超 SwimXYZ 合成数据 ≈80°/s），说明模板把估计噪声当成了技术基准，
会污染 DTW 对齐与参考分数的 smoothness 因素。本工具对模板前 4 列角度做滑动均值
平滑（:func:`src.feature_extractor.smooth_feature_matrix`），并据此重算对称性
与角速度列。

用法：
    python scripts/smooth_template.py \
        --in-path data/templates/champion_template.npy \
        --window 5 --fps 30

默认就地覆盖 ``--in-path``；覆盖前会把原始文件备份为 ``<name>.bak_raw``。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from common import add_repo_root_to_path, setup_logging

add_repo_root_to_path()
from src.feature_extractor import smooth_feature_matrix  # noqa: E402

logger = logging.getLogger(__name__)

_ANGLE_COLS = slice(0, 4)
_VELOCITY_COLS = slice(6, 10)


def main(args: argparse.Namespace) -> None:
    setup_logging()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path

    if not in_path.is_file():
        logger.error("模板文件不存在: %s", in_path)
        return

    raw = np.load(in_path)
    logger.info("已加载模板 %s | shape=%s", in_path, tuple(raw.shape))

    smoothed = smooth_feature_matrix(raw, window=args.window, fps=args.fps)

    angle_std_before = raw[:, _ANGLE_COLS].std(axis=0)
    angle_std_after = smoothed[:, _ANGLE_COLS].std(axis=0)
    vel_pooled_std_before = float(raw[:, _VELOCITY_COLS].std())
    vel_pooled_std_after = float(smoothed[:, _VELOCITY_COLS].std())

    logger.info(
        "角度 std（4列，平滑前→后，度）: %s → %s",
        np.round(angle_std_before, 1).tolist(),
        np.round(angle_std_after, 1).tolist(),
    )
    logger.info(
        "角速度 pooled std（平滑前→后，度/秒）: %.1f → %.1f",
        vel_pooled_std_before,
        vel_pooled_std_after,
    )

    # 就地覆盖前备份原始文件，避免误操作丢失原始（未平滑）模板
    if out_path == in_path:
        backup_path = in_path.with_name(in_path.stem + ".bak_raw" + in_path.suffix)
        np.save(backup_path, raw)
        logger.info("原始模板已备份至 %s", backup_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, smoothed)
    logger.info("已保存平滑后模板 %s | shape=%s", out_path, tuple(smoothed.shape))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="对模板 .npy 做角度平滑，抑制 HRNet 逐帧抖动带来的虚高角速度"
    )
    p.add_argument(
        "--in-path",
        type=str,
        default="data/templates/champion_template.npy",
        help="待平滑的模板 .npy 路径",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="输出路径；缺省时与 --in-path 相同（就地覆盖，覆盖前自动备份为 "
        "<name>.bak_raw.npy）",
    )
    p.add_argument(
        "--window",
        type=int,
        default=5,
        help="滑动均值窗宽（帧），<=1 表示不平滑",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="帧率，用于重算角速度 dt = 1/fps",
    )
    return p


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
