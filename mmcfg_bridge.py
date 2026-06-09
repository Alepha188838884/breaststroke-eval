"""把硬编码的相对 config 路径重定向到 mmdet/mmpose 包内的 .mim/configs 绝对路径。

src/pose_estimator.py 写死了：
  - configs/rtmdet/rtmdet_m_8xb32-300e_coco.py
  - configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py

mmdet 与 mmpose 都有 `_base_/default_runtime.py` 但内容不同
（default_scope='mmdet' vs 'mmpose'），所以不能合并到单一 ./configs/。

本模块在被 import 时给 mmengine.config.Config.fromfile 打 patch，
检测到上述两个相对路径就改写为对应包内的绝对路径，
后续 _base_ 链接就在各自的 .mim/configs 树内解析，互不干扰。

用法：在 extract_all.py / extract_template.py 顶部 `import mmcfg_bridge` 即可。
"""

from __future__ import annotations

from pathlib import Path

import mmdet
import mmpose
from mmengine.config import Config
from mmengine.registry import init_default_scope


def _bundled(pkg, rel: str) -> str:
    p = Path(pkg.__file__).parent / ".mim" / rel
    if not p.is_file():
        raise FileNotFoundError(f"未找到 mim 自带 config: {p}")
    return str(p)


_REDIRECTS: dict[str, str] = {
    "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py": _bundled(
        mmdet, "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
    ),
    "configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py": _bundled(
        mmpose,
        "configs/body_2d_keypoint/topdown_heatmap/coco/"
        "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
    ),
}

_orig_fromfile = Config.fromfile


def _patched_fromfile(filename, *args, **kwargs):
    key = str(filename)
    if key in _REDIRECTS:
        filename = _REDIRECTS[key]
    return _orig_fromfile(filename, *args, **kwargs)


# Config.fromfile 是 staticmethod，直接覆盖即可
Config.fromfile = staticmethod(_patched_fromfile)


# --- 同进程内 mmdet+mmpose 串台修复 ---
# init_pose_model / init_detector 都会把 default_scope 改成自己，
# 后续推理时如果 scope 不对，registry 找不到对应的 transforms。
# 这里把两个 inference 函数包一层，调用前强制切回正确 scope。

import mmdet.apis as _mmdet_apis
import mmpose.apis as _mmpose_apis

_orig_inference_detector = _mmdet_apis.inference_detector
_orig_inference_topdown = _mmpose_apis.inference_topdown


def _inference_detector_scoped(*args, **kwargs):
    init_default_scope("mmdet")
    return _orig_inference_detector(*args, **kwargs)


def _inference_topdown_scoped(*args, **kwargs):
    init_default_scope("mmpose")
    return _orig_inference_topdown(*args, **kwargs)


_mmdet_apis.inference_detector = _inference_detector_scoped
_mmpose_apis.inference_topdown = _inference_topdown_scoped
