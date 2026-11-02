"""脚本 1：从 Kinetics-700 下载蛙泳相关视频片段。

流程：
    1. 下载 Kinetics-700-2020 官方 annotation CSV（train / val / test）。
    2. 筛选 label 模糊匹配（同时含 "breast" 与 "swim"）的条目，如
       "swimming breast stroke"。
    3. 打印筛选统计（总数 / 各 split）。
    4. 用 yt-dlp 逐条下载 time_start–time_end 片段（≤720p mp4）。
    5. 断点续传（跳过已存在文件），失败 ID 记录到 failed_downloads.txt。

依赖（非 Python 库，需系统安装）：
    - yt-dlp:  pip install -U yt-dlp
    - ffmpeg:  yt-dlp 的 --download-sections 精确切片需要 ffmpeg

用法：
    python scripts/01_download_kinetics_breaststroke.py \
        --data-dir data/kinetics --limit 100
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from common import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def _resolve_ytdlp_cmd() -> list[str] | None:
    """返回调用 yt-dlp 的基础命令，找不到则返回 None。

    优先 ``python -m yt_dlp``（用当前解释器所在环境的 yt-dlp，不依赖 PATH，
    venv / Colab 均可靠）；其次回退到 PATH 上的 yt-dlp 可执行文件。
    """
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    exe = shutil.which("yt-dlp")
    return [exe] if exe else None

# Kinetics-700-2020 官方 annotation CSV（cvdfoundation/kinetics-dataset 指向的 S3）。
_ANNOTATION_BASE: str = "https://s3.amazonaws.com/kinetics/700_2020/annotations"
_SPLITS: tuple[str, ...] = ("train", "val", "test")

# 标签模糊匹配关键词：两者都命中才算蛙泳（Kinetics-700 中为 "swimming breast stroke"）。
_LABEL_KEYWORDS: tuple[str, str] = ("breast", "swim")


class Clip(NamedTuple):
    """一条待下载的视频片段标注。"""

    youtube_id: str
    time_start: int
    time_end: int
    split: str
    label: str


def download_annotation_csvs(anno_dir: Path) -> dict[str, Path]:
    """下载三个 split 的 annotation CSV（已存在则跳过）。

    Returns:
        {split: 本地 CSV 路径}，仅含成功获取的 split。
    """
    ensure_dir(anno_dir)
    paths: dict[str, Path] = {}
    for split in _SPLITS:
        dest = anno_dir / f"{split}.csv"
        if dest.is_file() and dest.stat().st_size > 0:
            logger.info("annotation 已存在，跳过下载: %s", dest)
            paths[split] = dest
            continue
        url = f"{_ANNOTATION_BASE}/{split}.csv"
        try:
            logger.info("下载 annotation: %s", url)
            urllib.request.urlretrieve(url, dest)
            paths[split] = dest
        except Exception as exc:  # noqa: BLE001 — 单个 split 失败不应中断其它
            logger.error("下载 annotation 失败 (%s): %s", split, exc)
    return paths


def _is_breaststroke(label: str) -> bool:
    """标签是否为蛙泳（小写后同时包含全部关键词）。"""
    low = label.lower()
    return all(kw in low for kw in _LABEL_KEYWORDS)


def filter_breaststroke_clips(csv_paths: dict[str, Path]) -> list[Clip]:
    """从各 split CSV 中筛选蛙泳片段。

    Kinetics-700-2020 的 train/val CSV 列为
    ``label,youtube_id,time_start,time_end,split``；test.csv 通常**无 label 列**
    （测试集不公开标签），此时无法按标签筛选，记录告警并跳过该 split。
    """
    clips: list[Clip] = []
    for split, path in csv_paths.items():
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if "label" not in fields:
                    logger.warning(
                        "split=%s 的 CSV 无 'label' 列（列=%s），无法按标签筛选，跳过",
                        split,
                        fields,
                    )
                    continue
                n_split = 0
                for row in reader:
                    label = (row.get("label") or "").strip()
                    if not _is_breaststroke(label):
                        continue
                    try:
                        clips.append(
                            Clip(
                                youtube_id=row["youtube_id"].strip(),
                                time_start=int(float(row["time_start"])),
                                time_end=int(float(row["time_end"])),
                                split=row.get("split", split).strip() or split,
                                label=label,
                            )
                        )
                        n_split += 1
                    except (KeyError, ValueError) as exc:
                        logger.warning("跳过异常行 (%s): %s | 行=%s", split, exc, row)
                logger.info("split=%s 命中蛙泳片段 %d 条", split, n_split)
        except Exception as exc:  # noqa: BLE001
            logger.error("读取 CSV 失败 (%s): %s", path, exc)
    return clips


def _download_one(
    clip: Clip,
    out_path: Path,
    sleep_min: int,
    sleep_max: int,
    limit_rate: str | None,
    ytdlp_cmd: list[str],
) -> bool:
    """用 yt-dlp 下载单个片段。成功返回 True。

    只下 ≤720p 的 mp4；用 --download-sections 精确切 time_start–time_end；
    --force-keyframes-at-cuts 保证切点准确（需 ffmpeg）。
    """
    url = f"https://www.youtube.com/watch?v={clip.youtube_id}"
    section = f"*{clip.time_start}-{clip.time_end}"
    cmd: list[str] = [
        *ytdlp_cmd,
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f",
        # 优先 ≤720p 的 mp4 视频+m4a 音频，回退到任意 ≤720p 最佳格式
        "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        section,
        # 不用 --force-keyframes-at-cuts：它在切点重编码，对部分流会触发
        # "ffmpeg exited with code 8" 的假失败。改为在最近关键帧切割（±数秒），
        # 对 10s 片段的姿态训练完全够用，且大幅提高成功率。
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        # rate limit：随机 sleep + 可选限速，降低被 YouTube 封 IP 的概率
        "--sleep-interval",
        str(sleep_min),
        "--max-sleep-interval",
        str(sleep_max),
        "-o",
        str(out_path),
        url,
    ]
    if limit_rate:
        cmd += ["--limit-rate", limit_rate]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        logger.warning("下载超时: %s", clip.youtube_id)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("下载异常 (%s): %s", clip.youtube_id, exc)
        return False

    if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
        return True
    logger.warning(
        "下载失败 (%s, rc=%d): %s",
        clip.youtube_id,
        proc.returncode,
        (proc.stderr or "").strip().splitlines()[-1:] or "无 stderr",
    )
    # 清理可能残留的 .part / 空文件
    if out_path.is_file() and out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
    return False


def main(args: argparse.Namespace) -> None:
    data_dir = ensure_dir(args.data_dir)
    anno_dir = ensure_dir(data_dir / "annotations")
    setup_logging(data_dir / "download.log")

    ytdlp_cmd = _resolve_ytdlp_cmd()
    if ytdlp_cmd is None:
        logger.error("未找到 yt-dlp，请先安装：pip install -U yt-dlp（切片还需 ffmpeg）")
        return
    logger.info("yt-dlp 调用方式: %s", " ".join(ytdlp_cmd))

    # 1–2. 下载并筛选 annotation
    csv_paths = download_annotation_csvs(anno_dir)
    if not csv_paths:
        logger.error("未能获取任何 annotation CSV，终止")
        return
    clips = filter_breaststroke_clips(csv_paths)

    # 3. 筛选统计
    per_split: dict[str, int] = {}
    for c in clips:
        per_split[c.split] = per_split.get(c.split, 0) + 1
    logger.info("蛙泳片段筛选完成 | 总计 %d 条 | 各 split: %s", len(clips), per_split)
    if not clips:
        logger.warning("无蛙泳片段可下载（test 集通常无标签）")
        return

    if args.limit is not None:
        clips = clips[: args.limit]
        logger.info("--limit=%d，仅下载前 %d 条", args.limit, len(clips))

    # 分片：多进程并行下载时，第 shard_id 个进程只取 clips[shard_id::num_shards]
    # （不相交子集）。下载是网络/IO 密集、不吃内存，故可比姿态开更多并发；但并发越高
    # 越可能触发 YouTube 限流/封 IP → 失败率飙升，需在并发数与温和 sleep 间取平衡。
    if args.num_shards > 1:
        clips = clips[args.shard_id :: args.num_shards]
        logger.info(
            "分片 %d/%d | 本进程负责 %d 条", args.shard_id, args.num_shards, len(clips)
        )

    # 4–5. 逐条下载（断点续传 + 记录失败）；分片时各写独立失败清单避免并发写冲突
    failed_name = (
        "failed_downloads.txt"
        if args.num_shards <= 1
        else f"failed_downloads_shard{args.shard_id}.txt"
    )
    failed_path = data_dir / failed_name
    n_ok = n_skip = n_fail = 0
    failed_ids: list[str] = []
    for clip in tqdm(clips, desc="下载蛙泳片段"):
        fname = f"{clip.youtube_id}_{clip.time_start}_{clip.time_end}.mp4"
        out_path = data_dir / fname
        if out_path.is_file() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        if _download_one(
            clip, out_path, args.sleep_min, args.sleep_max, args.limit_rate, ytdlp_cmd
        ):
            n_ok += 1
        else:
            n_fail += 1
            failed_ids.append(f"{clip.youtube_id}_{clip.time_start}_{clip.time_end}")
        # 视频间额外 sleep，进一步降低封 IP 风险
        if args.between_sleep > 0:
            time.sleep(args.between_sleep)

    if failed_ids:
        with open(failed_path, "a", encoding="utf-8") as f:
            f.write("\n".join(failed_ids) + "\n")

    # 结尾总结
    total_present = len(list(data_dir.glob("*.mp4")))
    logger.info(
        "下载完成 | 成功 %d | 跳过(已存在) %d | 失败 %d | 目录现有 mp4 %d 个",
        n_ok,
        n_skip,
        n_fail,
        total_present,
    )
    logger.info(
        "失败率 %.1f%%（YouTube 视频被删/私有属正常，预期 30–50%%）| 失败清单: %s",
        100.0 * n_fail / max(n_ok + n_fail, 1),
        failed_path if failed_ids else "无",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="下载 Kinetics-700 蛙泳视频片段")
    p.add_argument("--data-dir", type=Path, default=Path("data/kinetics"),
                   help="视频与 annotation 保存目录")
    p.add_argument("--limit", type=int, default=None,
                   help="仅下载前 N 条（调试用；缺省下载全部）")
    p.add_argument("--sleep-min", type=int, default=1, help="yt-dlp 请求间最小 sleep 秒")
    p.add_argument("--sleep-max", type=int, default=5, help="yt-dlp 请求间最大 sleep 秒")
    p.add_argument("--between-sleep", type=float, default=1.0,
                   help="每个视频下载完后额外 sleep 秒（防封 IP）")
    p.add_argument("--limit-rate", type=str, default=None,
                   help="yt-dlp 限速，如 '2M'；缺省不限速")
    p.add_argument("--num-shards", type=int, default=1,
                   help="分片总数：多进程并行下载时设为进程数（每进程下载不相交子集）")
    p.add_argument("--shard-id", type=int, default=0,
                   help="本进程分片编号，0..num_shards-1")
    return p


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
