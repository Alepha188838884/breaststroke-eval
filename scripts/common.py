"""批量处理 pipeline 的共享工具：路径引导、日志配置、设备检测、目录准备。

四个脚本（01_download / 02_pose / 03_feature / 04_dataset）都 import 本模块。
放在 ``scripts/`` 下；各脚本以 ``python scripts/0X_xxx.py`` 方式独立运行时，
``scripts/`` 已在 ``sys.path[0]``，故 ``import common`` 可直接生效。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 仓库根目录 = 本文件所在 scripts/ 的上一级。用于把根目录加进 sys.path，
# 从而能 ``import src.xxx`` 与 ``import mmcfg_bridge``（姿态估计的 config 重定向）。
REPO_ROOT: Path = Path(__file__).resolve().parent.parent


def add_repo_root_to_path() -> None:
    """把仓库根目录插到 sys.path 首位，使 src.* / mmcfg_bridge 可导入。"""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dir(path: str | os.PathLike) -> Path:
    """确保目录存在（等价 ``mkdir -p``），返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_file: str | os.PathLike | None = None) -> logging.Logger:
    """统一日志配置：INFO 级别，同时输出到 stdout 与（可选）日志文件。

    Args:
        log_file: 日志文件路径；None 表示只输出到 stdout。父目录自动创建。

    Returns:
        根 logger。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,  # 覆盖库（mmengine 等）可能已装的 handler，保证格式统一
    )
    return logging.getLogger()


def auto_detect_device(requested: str = "auto") -> str:
    """选择推理设备。``auto`` → cuda（若可用）否则 cpu。

    刻意**不**自动选择 mps：本项目实测 mmcv 的 NMS 在 Apple MPS 上会崩溃
    （见项目记忆 pose-inference-runtime）。Mac 上默认回退 CPU；确需尝试
    可显式传 ``--device mps``。

    Args:
        requested: "auto" / "cuda" / "mps" / "cpu"。非 auto 时原样返回。

    Returns:
        实际使用的设备字符串。
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
