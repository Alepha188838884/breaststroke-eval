"""SwimXYZ 数据集加载器：解析 2D 关键点 ``.txt`` 标注（默认 base/SMPL 骨架）。

SwimXYZ（Fiche et al., ACM MIG 2023）的 2D 标注是 **分号(``;``) 分隔、逗号(``,``)
作小数点** 的文本文件：首行为带关节名的表头（如 ``L Calf.x;L Calf.y;L Calf.z;...``），
其后每行是一帧；数据行中**空字段表示画面外/缺失关节**。文件按目录树组织：

    {stroke}/{view}/{swimmer}/{water}/{lighting}/{speed}/{position}/{skeleton}/{coord}.txt

每个 ``{skeleton}/{coord}.txt`` 即一个序列。默认用 ``base`` 骨架（SMPL/Biped 真值，
下肢解剖正确）+ ``Side_underwater`` 视角 + ``2D_cam.txt``——数据集导出的 ``COCO`` /
``body25`` 骨架下肢关节损坏（小腿长达 1.6m），会让膝角失真，**不可用**。

本加载器**按表头列名**取所需关节（不依赖列序），返回与 ``PoseEstimator.estimate_single``
完全兼容的关节字典列表，可直接传入 ``FeatureExtractor.extract_sequence_features``。
合成真值无遮挡时置信度置 ``1.0``，缺失关节置 ``0.0``（下游据此丢帧）。``z`` 列为深度，
本项目只用 ``x,y``。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# SwimXYZ 关节名 → 项目所需关节名（仅下肢 + 肩，用于角度 / 对称性 / 角速度）。
# 采用 **base（SMPL/Biped 真值）骨架** 的命名：其下肢解剖正确（thigh≈shank≈0.45m），
# 而数据集导出的 OpenPose 风格 ``COCO`` / ``body25`` 骨架下肢关节损坏（小腿长达 1.6m、
# 踝点错位），会使膝角退化失真，故不可用。Biped 命名中 Thigh/Calf/Foot 的关节枢轴
# 分别位于髋 / 膝 / 踝；UpperArm 枢轴位于肩。mid_shoulder 由左右肩中点计算。
_SWIMXYZ_JOINT_MAP: dict[str, str] = {
    "L UpperArm": "left_shoulder",
    "R UpperArm": "right_shoulder",
    "L Thigh": "left_hip",
    "R Thigh": "right_hip",
    "L Calf": "left_knee",
    "R Calf": "right_knee",
    "L Foot": "left_ankle",
    "R Foot": "right_ankle",
}


class SwimXYZLoader:
    """加载 SwimXYZ 的 2D 关键点 ``.txt`` 标注并转换为项目关节字典格式。"""

    def __init__(
        self,
        labels_dir: str,
        stroke: str = "breaststroke",
        skeleton: str = "base",
        coord_file: str = "2D_cam.txt",
        view: str | None = "Side_underwater",
    ) -> None:
        """初始化加载器。

        Args:
            labels_dir: 解压后的标注根目录（递归扫描，如 ``data/swimxyz/Breaststroke``）。
            stroke: 泳姿名称，仅用于日志。
            skeleton: 骨架子目录名，默认 ``base``（SMPL/Biped 真值，下肢解剖正确）。
                **不要用 ``COCO`` / ``body25``**：数据集这两个 OpenPose 风格导出的下肢
                关节损坏，膝角会失真。
            coord_file: 坐标文件名，默认 ``2D_cam.txt``（相机像素坐标）。
            view: 仅纳入路径含 ``/{view}/`` 的序列；``None`` 表示全部视角。默认
                ``Side_underwater``——侧下方机位与矢状面近似垂直，2D 膝角不被前缩
                （俯视 / 正面视角会把蹬腿前缩成退化信号）。

        Raises:
            FileNotFoundError: ``labels_dir`` 不存在。
        """
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"SwimXYZ 标注目录不存在: {labels_dir}")
        self.labels_dir = labels_dir
        self.stroke = stroke
        self.skeleton = skeleton
        self.coord_file = coord_file
        self.view = view
        self._sequence_paths: list[str] = self._discover_files()
        logger.info(
            "SwimXYZLoader 初始化完成 | stroke=%s | skeleton=%s | view=%s | dir=%s | 序列=%d",
            stroke,
            skeleton,
            view,
            labels_dir,
            len(self._sequence_paths),
        )

    def _discover_files(self) -> list[str]:
        """递归查找 ``{skeleton}/{coord_file}`` 标注文件（可按 ``view`` 过滤）。"""
        out: list[str] = []
        view_tag = f"{os.sep}{self.view}{os.sep}" if self.view else None
        for root, _, files in os.walk(self.labels_dir):
            if os.path.basename(root) != self.skeleton:
                continue
            if self.coord_file not in files:
                continue
            path = os.path.join(root, self.coord_file)
            if view_tag is not None and view_tag not in path:
                continue
            out.append(path)
        out.sort()
        return out

    def list_sequences(self) -> list[str]:
        """列出所有可用的标注文件路径。"""
        return list(self._sequence_paths)

    def load_sequence(
        self, label_path: str
    ) -> list[dict[str, tuple[float, float, float]]]:
        """从单个 SwimXYZ ``.txt`` 标注加载关节序列。

        Args:
            label_path: 标注文件路径（``;`` 分隔、``,`` 作小数点、首行表头）。

        Returns:
            关节字典列表，每项 ``{joint_name: (x, y, confidence)}``。可见关节
            ``confidence=1.0``；**画面外/缺失关节**（数据行中以空字段 ``;;`` 表示）
            ``confidence=0.0``，交由 ``FeatureExtractor`` 的最小置信度门限丢弃该帧。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 表头缺少所需关节列，或文件无数据行。
        """
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"标注文件不存在: {label_path}")

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if len(lines) < 2:
            raise ValueError(f"标注文件 {label_path} 行数不足（需表头 + ≥1 帧）")

        # 按位置解析：表头与数据均**保留空字段**（SwimXYZ 用空字段表示画面外/缺失
        # 关节，过滤空字段会错位）。表头末尾的空串不计入列名映射。
        header = lines[0].split(";")
        col_index = {name: i for i, name in enumerate(header) if name != ""}

        joint_cols: dict[str, tuple[int, int]] = {}
        for src, dst in _SWIMXYZ_JOINT_MAP.items():
            xk, yk = f"{src}.x", f"{src}.y"
            if xk not in col_index or yk not in col_index:
                raise ValueError(
                    f"标注 {label_path} 表头缺少关节 {src} 的 .x/.y 列；"
                    f"请确认 skeleton='{self.skeleton}' 是否含下肢关节。"
                )
            joint_cols[dst] = (col_index[xk], col_index[yk])

        def _coord(fields: list[str], i: int) -> tuple[float, bool]:
            """返回 (值, 是否有效)；越界或空字段判为无效。"""
            if i >= len(fields):
                return 0.0, False
            s = fields[i].strip()
            if s == "":
                return 0.0, False
            try:
                return float(s.replace(",", ".")), True
            except ValueError:
                return 0.0, False

        sequence: list[dict[str, tuple[float, float, float]]] = []
        for line in lines[1:]:
            if line.strip() == "":
                continue
            fields = line.split(";")  # 保留空字段以维持列对齐
            joints: dict[str, tuple[float, float, float]] = {}
            for dst, (xi, yi) in joint_cols.items():
                x, ox = _coord(fields, xi)
                y, oy = _coord(fields, yi)
                conf = 1.0 if (ox and oy) else 0.0  # 缺失关节 → 0，下游据此丢帧
                joints[dst] = (x, y, conf)
            ls, rs = joints["left_shoulder"], joints["right_shoulder"]
            joints["mid_shoulder"] = (
                (ls[0] + rs[0]) / 2.0,
                (ls[1] + rs[1]) / 2.0,
                min(ls[2], rs[2]),
            )
            sequence.append(joints)

        if not sequence:
            raise ValueError(f"标注 {label_path} 无任何数据行")

        logger.info("加载序列 %s | 帧数=%d", label_path, len(sequence))
        return sequence

    def load_all_sequences(
        self,
    ) -> list[list[dict[str, tuple[float, float, float]]]]:
        """加载所有标注文件，跳过无法解析的文件。"""
        sequences: list[list[dict[str, tuple[float, float, float]]]] = []
        for path in self._sequence_paths:
            try:
                sequences.append(self.load_sequence(path))
            except (ValueError, FileNotFoundError) as exc:
                logger.warning("跳过无效标注 %s: %s", path, exc)
        total_frames = sum(len(s) for s in sequences)
        logger.info(
            "load_all_sequences 完成 | 序列数=%d | 总帧数=%d",
            len(sequences),
            total_frames,
        )
        return sequences
