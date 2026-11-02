"""导出/比对教练盲评样本：为 reference_score 的外部效度提供人工对照。

背景（见 src/kick_env.py 模块 docstring 的"效度边界"说明）：reference_score
是规则公式（角度/对称/时序/平滑加权），不含任何外部人工真值，只能证明系统
自洽（原始运动学能多大程度重建该规则分）。要评估外部效度（规则分是否与
教练眼中的技术正确性一致），需要独立于该公式的人工评分做对照。

**关键方法论要求：盲评必须真盲**——教练不能看到系统分/agent分，否则会
锚定评分，相关性失去意义。因此导出产物拆成两份：

    - coach_sheet.csv / coach_rating.html：只给教练，绝不含任何系统分数值。
    - answer_key.csv：自己留存的系统分/agent分，教练看不到。

两种用法：

    1. 导出盲评材料（默认导出全部测试集视频；显式 --n 时才随机抽样）：
           python scripts/export_blind_eval.py --config configs/config.yaml
       产出 data/blind_eval/{coach_sheet.csv, answer_key.csv,
       coach_rating.html, coach_instructions.md}。

    2. 教练在浏览器打开 coach_rating.html 逐个视频打分，点「导出评分 CSV」
       下载 filled_coach_sheet.csv 后回传，比对系统分/蒸馏分与教练分的一致性：
           python scripts/export_blind_eval.py \\
               --compare data/blind_eval/filled_coach_sheet.csv \\
               --answer-key data/blind_eval/answer_key.csv
       （亦兼容比对旧版单文件——即本身就含 system_score 列的 CSV——此时无需
       --answer-key。）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from common import add_repo_root_to_path, ensure_dir, setup_logging

add_repo_root_to_path()
from src.dtw_aligner import DTWAligner, _MAX_CYCLE_FRAMES, _MIN_CYCLE_FRAMES  # noqa: E402
from src.kick_env import BreaststrokeKickEnv  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_META = "data/training/dataset_meta.json"
_DEFAULT_FEATURES_DIR = "data/features"
_DEFAULT_TEMPLATE = "data/templates/champion_template.npy"
_DEFAULT_OUT_DIR = "data/blind_eval"
_DEFAULT_CONFIG = "configs/config.yaml"

_COACH_SHEET_NAME = "coach_sheet.csv"
_ANSWER_KEY_NAME = "answer_key.csv"
_HTML_NAME = "coach_rating.html"
_INSTRUCTIONS_NAME = "coach_instructions.md"

_COACH_SHEET_FIELDS = ["video_id", "n_cycles", "coach_score", "notes"]
_ANSWER_KEY_FIELDS = ["video_id", "n_cycles", "system_score", "agent_score"]

# config.yaml reference_score.scale_max_real 的键 -> kick_env.score_scales 键
_SCALE_KEY_MAP: dict[str, str] = {
    "angle_dev": "angle",
    "symmetry_dev": "symmetry",
    "timing_warp": "timing",
    "smoothness": "smoothness",
}


def _load_score_scales(config_path: str) -> dict[str, float] | None:
    """读取 ``config_path`` 的 ``reference_score.scale_max_real``，映射为
    :class:`BreaststrokeKickEnv` 的 ``score_scales`` 参数。

    加载失败（文件不存在 / 格式错误 / 未配置该键）不致命：warning 后返回
    ``None``，调用方此时让环境沿用 SwimXYZ 合成默认标定。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        scale_max_real = cfg.get("reference_score", {}).get("scale_max_real")
        if not scale_max_real:
            logger.warning(
                "%s 未配置 reference_score.scale_max_real，使用默认(SwimXYZ)标定",
                config_path,
            )
            return None
        return {
            _SCALE_KEY_MAP[k]: float(v)
            for k, v in scale_max_real.items()
            if k in _SCALE_KEY_MAP
        }
    except Exception as exc:  # noqa: BLE001 — 配置读取失败不应阻断盲评导出
        logger.warning(
            "读取 %s 的 scale_max_real 失败，使用默认(SwimXYZ)标定: %s",
            config_path,
            exc,
        )
        return None


def _resolve_model_path(model_arg: str | None, config_path: str) -> str:
    """解析 PPO 模型路径：``--model`` 显式给出则用之，否则取 config
    ``paths.model_output``；两者都拿不到时回退默认路径。
    """
    if model_arg:
        return model_arg
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("paths", {}).get("model_output", "models/ppo_kick_scorer.zip")
    except Exception as exc:  # noqa: BLE001 — 读取失败不应阻断盲评导出
        logger.warning("读取 %s 获取默认模型路径失败: %s", config_path, exc)
        return "models/ppo_kick_scorer.zip"


def _load_model(model_path: str) -> Any:
    """惰性加载 PPO 模型；文件不存在时告警并返回 ``None``（agent_score 列留空）。"""
    if not Path(model_path).is_file():
        logger.warning("模型文件不存在，agent_score 列将留空: %s", model_path)
        return None
    from stable_baselines3 import PPO  # 惰性 import：无模型时不引入该重依赖

    model = PPO.load(model_path)
    logger.info("已加载模型 %s 用于 agent_score", model_path)
    return model


def _load_representative_aligner(template_path: str) -> DTWAligner:
    """加载模板并按 train_agent 的做法切出中位长度单周期（若模板超单周期上界）。"""
    aligner = DTWAligner(template_path)
    if aligner.template.shape[0] > _MAX_CYCLE_FRAMES:
        template_cycles = aligner.segment_kick_cycles(aligner.template)
        if template_cycles:
            lengths = [c.shape[0] for c in template_cycles]
            rep = template_cycles[int(np.argsort(lengths)[len(lengths) // 2])]
            logger.warning(
                "模板 %s 含 %d 帧，疑为多周期；已就地切出中位长度单周期 (%d 帧) 用于对齐",
                template_path,
                aligner.template.shape[0],
                rep.shape[0],
            )
            aligner.template = rep
    return aligner


def score_video(
    video_id: str,
    features_dir: Path,
    aligner: DTWAligner,
    score_scales: dict[str, float] | None = None,
    model: Any = None,
) -> dict[str, Any] | None:
    """加载单个视频的蹬腿周期，构建环境并取周期参考分均值作为 system_score。

    Args:
        score_scales: 传给 :class:`BreaststrokeKickEnv` 的参考分数尺度覆盖
            （来自 config 的 ``reference_score.scale_max_real``）；``None``
            时沿用 SwimXYZ 合成默认标定。
        model: 已加载的 PPO 模型（或兼容 ``predict(obs, deterministic)`` 的
            鸭子类型）；``None`` 时不计算 agent_score（返回值中为 ``None``）。

    Returns:
        ``{"video_id", "n_cycles", "system_score", "agent_score"}``；若无可用
        周期或环境构建失败（如所有周期 DTW 都失败），返回 ``None``（调用方跳过
        并已打过 warning）。``agent_score`` 在 ``model is None`` 时为 ``None``。
    """
    pkl_path = features_dir / f"{video_id}_cycles.pkl"
    if not pkl_path.is_file():
        logger.warning("特征文件不存在，跳过 %s: %s", video_id, pkl_path)
        return None

    try:
        with open(pkl_path, "rb") as f:
            raw_cycles = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取周期失败，跳过 %s: %s", video_id, exc)
        return None

    valid_cycles: list[np.ndarray] = []
    for c in raw_cycles:
        arr = np.asarray(c)
        if arr.ndim == 2 and arr.shape[1] == 10 and arr.shape[0] >= _MIN_CYCLE_FRAMES:
            valid_cycles.append(arr)
    if not valid_cycles:
        logger.warning("视频 %s 无满足最短帧数的有效周期，跳过", video_id)
        return None

    try:
        env = BreaststrokeKickEnv(
            valid_cycles, aligner, seed=0, score_scales=score_scales
        )
    except Exception as exc:  # noqa: BLE001 — 单个视频构建失败不应影响其它视频
        logger.warning("视频 %s 构建评估环境失败，跳过: %s", video_id, exc)
        return None

    scores = env._cached_target_score
    agent_score: float | None = None
    if model is not None:
        agent_score = float(
            np.mean(
                [int(model.predict(o, deterministic=True)[0]) * 5 for o in env._cached_obs]
            )
        )
    return {
        "video_id": video_id,
        "n_cycles": len(scores),
        "system_score": float(np.mean(scores)),
        "agent_score": agent_score,
    }


_RUBRIC_HTML = """
<ul>
  <li><b>90-100</b> 技术优秀，接近竞技水平 / Excellent, near-competitive
      （对称、外翻蹬水充分、无明显瑕疵 / symmetric, full outward whip, no visible flaws）</li>
  <li><b>75-89</b> 良好，仅轻微问题 / Good, only minor issues</li>
  <li><b>60-74</b> 基本正确但有明显可纠正问题 / Basically correct but with
      clear, fixable issues</li>
  <li><b>40-59</b> 多处技术错误 / Multiple technical errors
      （明显不对称/剪式腿/时序紊乱 / visible asymmetry, scissor kick, timing disorder）</li>
  <li><b>0-39</b> 严重错误或几乎不成形 / Severe errors or barely recognizable form</li>
</ul>
"""


def _render_coach_html(rows: list[dict[str, Any]]) -> str:
    """生成自包含（内联 CSS+JS，无外部依赖）的教练盲评 HTML 页面。

    页面只内嵌 ``video_id``/``n_cycles``，不内嵌任何系统分/agent分。
    """
    videos_json = json.dumps(
        [{"video_id": r["video_id"], "n_cycles": r["n_cycles"]} for r in rows],
        ensure_ascii=False,
    )
    cards = "\n".join(
        f"""
    <div class="card">
      <h3>#{i + 1} &nbsp; {r['video_id']}</h3>
      <video controls preload="metadata" style="max-width:480px;width:100%;background:#000"
             src="clips/{r['video_id']}.mp4"
             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
      </video>
      <p class="fallback" style="display:none;color:#b00">
        视频加载失败：若无法播放，可按 video_id（{r['video_id']}）手动查找对应片段。
      </p>
      <div class="row">
        <label>评分 (0-100)：
          <input type="number" class="score-input" min="0" max="100" step="1"
                 data-video-id="{r['video_id']}">
        </label>
      </div>
      <div class="row">
        <label>备注：
          <textarea class="notes-input" data-video-id="{r['video_id']}" rows="2"
                    style="width:100%;max-width:480px"></textarea>
        </label>
      </div>
    </div>"""
        for i, r in enumerate(rows)
    )

    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>蛙泳蹬腿技术盲评</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         max-width: 720px; margin: 0 auto; padding: 16px 16px 96px; color: #222; }}
  h1 {{ font-size: 22px; }}
  .intro {{ background: #f4f6f8; border: 1px solid #dde3e8; border-radius: 8px;
            padding: 12px 16px; margin-bottom: 20px; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px;
           margin-bottom: 16px; }}
  .row {{ margin-top: 8px; }}
  input[type=number] {{ width: 80px; font-size: 16px; padding: 4px; }}
  #export-bar {{ position: fixed; bottom: 0; left: 0; right: 0; background: #fff;
                 border-top: 1px solid #ccc; padding: 12px; text-align: center; }}
  #export-btn {{ font-size: 16px; padding: 10px 24px; cursor: pointer; }}
  #export-status {{ margin-left: 12px; color: #2a7a2a; }}
</style>
</head>
<body>
  <h1>蛙泳蹬腿技术盲评 / Breaststroke Kick Blind Rating</h1>
  <div class="intro">
    <p>请对每段视频的<b>蹬腿技术正确性</b>打分（0-100），<b>而非速度或名次</b>。
       请独立打分，不要参考任何其它评分或来源。</p>
    <p>评分锚点（供参考，可按您自身标准；<b>相对排序比绝对分更重要</b>）：</p>
    {_RUBRIC_HTML}
  </div>

  <div id="cards">{cards}
  </div>

  <div id="export-bar">
    <button id="export-btn" onclick="exportCsv()">导出评分 CSV</button>
    <span id="export-status"></span>
  </div>

<script>
const VIDEOS = {videos_json};

function exportCsv() {{
  const rows = [["video_id", "n_cycles", "coach_score", "notes"]];
  for (const v of VIDEOS) {{
    const scoreEl = document.querySelector(`.score-input[data-video-id="${{CSS.escape(v.video_id)}}"]`);
    const notesEl = document.querySelector(`.notes-input[data-video-id="${{CSS.escape(v.video_id)}}"]`);
    const score = scoreEl && scoreEl.value !== "" ? scoreEl.value : "";
    let notes = notesEl ? notesEl.value : "";
    notes = notes.replace(/"/g, '""');
    rows.push([v.video_id, v.n_cycles, score, `"${{notes}}"`]);
  }}
  const csvContent = rows.map(r => r.join(",")).join("\\r\\n");
  const blob = new Blob([csvContent], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "filled_coach_sheet.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  document.getElementById("export-status").textContent = "已导出 filled_coach_sheet.csv，请回传给研究者";
}}
</script>
</body>
</html>
"""


def _render_instructions_md(n_videos: int) -> str:
    return f"""# 教练盲评须知

感谢您协助完成本研究的外部效度验证！

## 操作步骤

1. 用浏览器打开同目录下的 `coach_rating.html`（双击即可，无需网络）。
2. 页面共 {n_videos} 段视频，逐个观看后在对应输入框填写：
   - **评分（0-100）**：蹬腿技术正确性打分。
   - **备注**（可选）：任何观察或疑虑。
3. 全部填完后，点击页面底部「导出评分 CSV」按钮，浏览器会下载
   `filled_coach_sheet.csv`。
4. 将 `filled_coach_sheet.csv` 回传给研究者即可，无需回传其它文件。

## 评分说明（rubric）

{_RUBRIC_HTML.replace('<ul>', '').replace('</ul>', '').replace('<li>', '- ').replace('</li>', '')}

## 重要提醒

- 请**独立打分**，评分锚点仅供参考，可按您自身专业标准判断。
- 评的是**蹬腿技术正确性**，不是速度、名次或整体印象。
- **请勿查阅任何系统评分**——本次评估的目的正是检验系统评分与专家判断的
  一致性，若参考了系统分会使结果失去意义。
- 相对排序（谁比谁更好）比绝对分值更重要，请尽量保持评分内部一致。
"""


def export(args: argparse.Namespace) -> None:
    """导出盲评材料：coach_sheet.csv（教练用，不含系统分）+ answer_key.csv
    （留档，含系统分/agent分）+ coach_rating.html + coach_instructions.md。

    默认导出全部测试集视频；显式给出 ``--n`` 时才随机抽样 ``n`` 个。
    """
    meta_path = Path(args.dataset_meta)
    if not meta_path.is_file():
        logger.error("dataset_meta.json 不存在: %s（先运行 scripts/04）", meta_path)
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    test_ids = list(meta.get("test_video_ids", []))
    if not test_ids:
        logger.error("dataset_meta.json 中 test_video_ids 为空")
        return

    if args.n is None:
        chosen = sorted(test_ids)
        logger.info("导出全部 %d 个测试集视频", len(chosen))
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n, len(test_ids))
        chosen = sorted(rng.choice(test_ids, size=n, replace=False).tolist())
        logger.info("从 %d 个测试集视频中随机抽取 %d 个（seed=%d）", len(test_ids), n, args.seed)

    aligner = _load_representative_aligner(args.template)
    score_scales = _load_score_scales(args.config)
    model = _load_model(_resolve_model_path(args.model, args.config))

    features_dir = Path(args.features_dir)
    rows: list[dict[str, Any]] = []
    for video_id in chosen:
        result = score_video(
            video_id, features_dir, aligner, score_scales=score_scales, model=model
        )
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("所有抽中视频均评分失败，未导出任何行")
        return

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    coach_sheet_path = out_dir / _COACH_SHEET_NAME
    with open(coach_sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_COACH_SHEET_FIELDS)
        for r in rows:
            writer.writerow([r["video_id"], r["n_cycles"], "", ""])

    answer_key_path = out_dir / _ANSWER_KEY_NAME
    with open(answer_key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_ANSWER_KEY_FIELDS)
        for r in rows:
            agent_str = "" if r["agent_score"] is None else f"{r['agent_score']:.1f}"
            writer.writerow(
                [r["video_id"], r["n_cycles"], f"{r['system_score']:.1f}", agent_str]
            )

    html_path = out_dir / _HTML_NAME
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_coach_html(rows))

    instructions_path = out_dir / _INSTRUCTIONS_NAME
    with open(instructions_path, "w", encoding="utf-8") as f:
        f.write(_render_instructions_md(len(rows)))

    logger.info("已导出 %d/%d 个视频的盲评材料 -> %s", len(rows), len(chosen), out_dir)
    logger.info(
        "coach_sheet.csv / coach_rating.html / coach_instructions.md 交给教练；"
        "answer_key.csv 自己留存，勿给教练"
    )
    print(
        "\n请将 coach_rating.html + coach_instructions.md + clips/ 目录整体发给教练，"
        "教练在浏览器打开 coach_rating.html 打分并导出 filled_coach_sheet.csv 回传后，"
        "用 --compare filled_coach_sheet.csv --answer-key answer_key.csv 比对。"
    )
    print(
        "video_id 格式为 {youtube_id}_{起始秒}_{结束秒}，如 "
        "'0HW6HfdS8iA_0_10' 表示该 YouTube 视频第 0-10 秒片段。"
    )


def _report_correlation(tag: str, sys_scores: list[float], coach_scores: list[float]) -> None:
    """打印 ``tag`` 分数与教练分的 n、Spearman rho(+p)、Pearson r(+p)、MAE；
    样本不足 2 时只提示；n<10 时追加统计功效警告。
    """
    n = len(coach_scores)
    if n < 2:
        print(f"{tag}: 已填写行数 = {n}，不足以计算相关系数（需 >= 2）")
        return
    sys_arr = np.asarray(sys_scores, dtype=np.float64)
    coach_arr = np.asarray(coach_scores, dtype=np.float64)
    rho, rho_p = spearmanr(sys_arr, coach_arr)
    pearson_r, pearson_p = pearsonr(sys_arr, coach_arr)
    mae = float(np.mean(np.abs(sys_arr - coach_arr)))
    print(
        f"{tag} | n={n} | Spearman rho={rho:.3f} (p={rho_p:.4f}) | "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}) | MAE={mae:.2f} 分"
    )
    if n < 10:
        print(
            f"  [警告] 样本量 n={n} 偏小，统计功效有限，Spearman 需 rho≳0.64(n=10) "
            "才达 p<0.05，结论宜谨慎"
        )


def _is_legacy_single_csv(csv_path: Path) -> bool:
    """判断 ``csv_path`` 是否为旧版单文件格式（表头含 system_score 列）。"""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    return "system_score" in header


def _compare_legacy(csv_path: Path) -> None:
    """旧版单文件格式（同一 CSV 含 system_score/agent_score/coach_score）的比对路径。"""
    system_scores: list[float] = []
    coach_for_system: list[float] = []
    agent_scores: list[float] = []
    coach_for_agent: list[float] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coach_raw = (row.get("coach_score") or "").strip()
            if coach_raw == "":
                continue
            try:
                coach_val = float(coach_raw)
            except ValueError as exc:
                logger.warning("跳过无法解析的行 %s: %s", row, exc)
                continue

            system_raw = (row.get("system_score") or "").strip()
            if system_raw != "":
                try:
                    system_scores.append(float(system_raw))
                    coach_for_system.append(coach_val)
                except ValueError as exc:
                    logger.warning("跳过无法解析的 system_score: %s", exc)

            agent_raw = (row.get("agent_score") or "").strip()
            if agent_raw != "":
                try:
                    agent_scores.append(float(agent_raw))
                    coach_for_agent.append(coach_val)
                except ValueError as exc:
                    logger.warning("跳过无法解析的 agent_score: %s", exc)

    _report_correlation("system_score vs coach", system_scores, coach_for_system)
    _report_correlation("agent_score vs coach", agent_scores, coach_for_agent)


def _read_csv_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    """读取 CSV，返回 ``{video_id: row_dict}``。"""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["video_id"]: row for row in reader if row.get("video_id")}


def compare(args: argparse.Namespace) -> None:
    """比对教练分与系统分/蒸馏分的一致性。

    新格式：``--compare`` 指向教练回填的 coach_sheet（filled_coach_sheet.csv，
    仅含 video_id/n_cycles/coach_score/notes），与 ``--answer-key`` 指向的
    answer_key.csv（含 system_score/agent_score）按 video_id 内连接后计算。

    向后兼容：若 ``--compare`` 的 CSV 本身就含 system_score 列（旧版单文件
    格式），直接用它计算，忽略 ``--answer-key``。
    """
    csv_path = Path(args.compare)
    if not csv_path.is_file():
        logger.error("CSV 文件不存在: %s", csv_path)
        return

    if _is_legacy_single_csv(csv_path):
        logger.info("检测到旧版单文件格式（含 system_score 列），按兼容路径比对")
        _compare_legacy(csv_path)
        return

    answer_key_path = Path(args.answer_key)
    if not answer_key_path.is_file():
        logger.error(
            "答案表不存在: %s（新格式 --compare 需配合 --answer-key）", answer_key_path
        )
        return

    coach_rows = _read_csv_rows(csv_path)
    answer_rows = _read_csv_rows(answer_key_path)

    system_scores: list[float] = []
    coach_for_system: list[float] = []
    agent_scores: list[float] = []
    coach_for_agent: list[float] = []
    n_no_answer = 0
    for video_id, coach_row in coach_rows.items():
        coach_raw = (coach_row.get("coach_score") or "").strip()
        if coach_raw == "":
            continue
        try:
            coach_val = float(coach_raw)
        except ValueError as exc:
            logger.warning("跳过无法解析的教练分行 %s: %s", coach_row, exc)
            continue

        answer_row = answer_rows.get(video_id)
        if answer_row is None:
            n_no_answer += 1
            continue

        system_raw = (answer_row.get("system_score") or "").strip()
        if system_raw != "":
            try:
                system_scores.append(float(system_raw))
                coach_for_system.append(coach_val)
            except ValueError as exc:
                logger.warning("跳过无法解析的 system_score: %s", exc)

        agent_raw = (answer_row.get("agent_score") or "").strip()
        if agent_raw != "":
            try:
                agent_scores.append(float(agent_raw))
                coach_for_agent.append(coach_val)
            except ValueError as exc:
                logger.warning("跳过无法解析的 agent_score: %s", exc)

    if n_no_answer:
        logger.warning(
            "%d 个 video_id 在教练表中出现但答案表(%s)中没有，已跳过",
            n_no_answer,
            answer_key_path,
        )

    _report_correlation("system_score vs coach", system_scores, coach_for_system)
    if agent_scores:
        _report_correlation("agent_score vs coach", agent_scores, coach_for_agent)
    else:
        print("agent_score vs coach: answer_key 中该列为空，跳过")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="导出教练盲评样本，或比对已回填的教练分与系统分（reference_score 外部效度对照）"
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="随机抽取的测试集视频数；缺省(None)导出全部测试集视频",
    )
    p.add_argument("--seed", type=int, default=42, help="随机抽样种子（仅 --n 给出时生效）")
    p.add_argument(
        "--config", type=str, default=_DEFAULT_CONFIG,
        help="config.yaml 路径（读取 reference_score.scale_max_real 评分尺度）",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help=(
            "PPO 模型 .zip 路径，用于计算 agent_score 列；缺省时取 config "
            "paths.model_output。模型文件不存在时告警并留空 agent_score 列，脚本不崩。"
        ),
    )
    p.add_argument(
        "--dataset-meta", type=str, default=_DEFAULT_DATASET_META,
        help="dataset_meta.json 路径（提供 test_video_ids）",
    )
    p.add_argument(
        "--features-dir", type=str, default=_DEFAULT_FEATURES_DIR,
        help="含 *_cycles.pkl 的特征目录",
    )
    p.add_argument(
        "--template", type=str, default=_DEFAULT_TEMPLATE,
        help="冠军模板 .npy 路径（用于 DTW 对齐计算参考分）",
    )
    p.add_argument(
        "--out", type=str, default=_DEFAULT_OUT_DIR,
        help="导出目录，产出 coach_sheet.csv / answer_key.csv / coach_rating.html / "
        "coach_instructions.md",
    )
    p.add_argument(
        "--compare", type=str, default=None,
        help=(
            "教练回填的 coach_sheet CSV 路径（filled_coach_sheet.csv）；给出时进入比对"
            "模式，忽略 --n/--out。若该 CSV 本身含 system_score 列（旧版单文件格式），"
            "直接用它比对，忽略 --answer-key"
        ),
    )
    p.add_argument(
        "--answer-key", type=str, default=str(Path(_DEFAULT_OUT_DIR) / _ANSWER_KEY_NAME),
        help="answer_key.csv 路径（含 system_score/agent_score），配合新格式 --compare 使用",
    )
    return p


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.compare:
        compare(args)
    else:
        export(args)


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
