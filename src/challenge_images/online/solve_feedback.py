"""用解题结果自动标注在线采集数据。

项目已归档 34384 张在线图，但归档目录是
``<挑战类型>/<挑战目标类别>/``——这个类别是「本轮要找什么」，
不是「这张图里有什么」。9 个格子可能只有 3 个含目标，
整张图归到同一个类别文件夹会引入大量错误标签，
因此这批数据此前完全无法用于训练。

真正的解锁点是记录挑战是否通过：

    挑战通过 ⇒ 点击的格子 = 确认正样本，未点击的格子 = 确认负样本

这是零人工成本的高质量标注，而且分布天然匹配线上真实分布。
一次通过的 3×3 挑战直接产出 9 条标注。

导出格式刻意选用多标签而非单标签文件夹：未点击的格子表示
「不含本轮目标」，而不是「属于某个其他类别」。单标签目录无法表达
这种纯负样本，多标签下它就是一个全零多热向量，BCE 天然支持。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..grid.grid_engine import GridSpec, split_grid

OUTCOME_PASSED = "passed"
OUTCOME_FAILED = "failed"
OUTCOME_UNKNOWN = "unknown"

OUTCOME_LABELS = {
    OUTCOME_PASSED: "通过",
    OUTCOME_FAILED: "未通过",
    OUTCOME_UNKNOWN: "未知",
}

FEEDBACK_FILENAME = "solve_outcomes.json"


@dataclass(frozen=True)
class SolveRecord:
    """一轮挑战的点击与结果。"""

    image_name: str
    image_sha256: str
    challenge_type: str
    target_class: str
    grid_rows: int
    grid_cols: int
    clicked_indices: list[int]
    outcome: str
    recorded_at: str = ""

    @property
    def spec(self) -> GridSpec:
        return GridSpec(self.grid_rows, self.grid_cols)

    @property
    def usable(self) -> bool:
        """只有通过的挑战才能作为真值。

        未通过时无法区分「点错了」还是「漏点了」，两种情况的标签完全不同，
        因此不能用于标注。
        """
        return self.outcome == OUTCOME_PASSED

    def as_dict(self) -> dict[str, Any]:
        return {
            "图片": self.image_name,
            "sha256": self.image_sha256,
            "挑战类型": self.challenge_type,
            "目标类别": self.target_class,
            "网格": f"{self.grid_rows}x{self.grid_cols}",
            "点击格子": sorted(self.clicked_indices),
            "结果": self.outcome,
            "结果说明": OUTCOME_LABELS.get(self.outcome, self.outcome),
            "记录时间": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SolveRecord":
        grid = str(payload.get("网格", "3x3")).lower().replace("×", "x")
        try:
            rows, cols = (int(value) for value in grid.split("x", 1))
        except (TypeError, ValueError):
            rows, cols = 3, 3
        return cls(
            image_name=str(payload.get("图片", "")),
            image_sha256=str(payload.get("sha256", "")),
            challenge_type=str(payload.get("挑战类型", "")),
            target_class=str(payload.get("目标类别", "")),
            grid_rows=rows,
            grid_cols=cols,
            clicked_indices=[int(index) for index in payload.get("点击格子", [])],
            outcome=str(payload.get("结果", OUTCOME_UNKNOWN)),
            recorded_at=str(payload.get("记录时间", "")),
        )


class SolveFeedbackStore:
    """解题反馈的追加式存储。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[SolveRecord] = []
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items = payload.get("记录", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict):
                self.records.append(SolveRecord.from_dict(item))

    def append(self, record: SolveRecord) -> SolveRecord:
        """追加一条记录并落盘。"""
        if not record.recorded_at:
            record = SolveRecord(
                **{
                    **record.__dict__,
                    "recorded_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        self.records.append(record)
        self.save()
        return record

    def save(self) -> Path:
        self.path.write_text(
            json.dumps(
                {
                    "说明": (
                        "挑战通过时，点击的格子为确认正样本、未点击的格子为确认负样本；"
                        "未通过的记录无法区分点错与漏点，不用于标注。"
                    ),
                    "记录数": len(self.records),
                    "可用于标注": sum(record.usable for record in self.records),
                    "记录": [record.as_dict() for record in self.records],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.path

    def usable_records(self) -> list[SolveRecord]:
        return [record for record in self.records if record.usable]

    def statistics(self) -> dict[str, Any]:
        by_outcome: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for record in self.records:
            by_outcome[record.outcome] = by_outcome.get(record.outcome, 0) + 1
            if record.usable:
                by_class[record.target_class] = by_class.get(record.target_class, 0) + 1
        usable = self.usable_records()
        return {
            "记录总数": len(self.records),
            "按结果": {OUTCOME_LABELS.get(k, k): v for k, v in sorted(by_outcome.items())},
            "可标注记录": len(usable),
            "可产出图块": sum(record.spec.count for record in usable),
            "按目标类别": dict(sorted(by_class.items(), key=lambda item: -item[1])),
        }


@dataclass
class LabelExportReport:
    """图块标注导出结果。"""

    output: Path
    records_used: int = 0
    positive_tiles: int = 0
    negative_tiles: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    skipped_missing_image: int = 0

    @property
    def total_tiles(self) -> int:
        return self.positive_tiles + self.negative_tiles

    def as_dict(self) -> dict[str, Any]:
        return {
            "输出目录": str(self.output),
            "使用记录数": self.records_used,
            "图块总数": self.total_tiles,
            "正样本": self.positive_tiles,
            "负样本": self.negative_tiles,
            "逐类正样本": dict(sorted(self.per_class.items(), key=lambda item: -item[1])),
            "跳过（找不到图片）": self.skipped_missing_image,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_tile_labels(
    records: Iterable[SolveRecord],
    capture_root: str | Path,
    output: str | Path,
    *,
    classes: Iterable[str] | None = None,
) -> LabelExportReport:
    """把通过的挑战切成带标签的图块。

    输出目录结构::

        output/
        ├── images/<原图名>_tile<编号>.jpg
        └── tile_labels.json      每张图块的标签列表（空列表表示纯负样本）
    """
    from PIL import Image

    capture = Path(capture_root)
    output_root = Path(output)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    report = LabelExportReport(output=output_root)
    samples: list[dict[str, Any]] = []
    seen_classes: set[str] = set(classes or [])

    for record in records:
        if not record.usable:
            continue
        source = _locate_image(capture, record)
        if source is None:
            report.skipped_missing_image += 1
            continue
        try:
            image = Image.open(source).convert("RGB")
        except OSError:
            report.skipped_missing_image += 1
            continue

        spec = record.spec
        tiles = split_grid(image, spec)
        clicked = {int(index) for index in record.clicked_indices}
        seen_classes.add(record.target_class)
        report.records_used += 1

        for index, tile in enumerate(tiles):
            name = f"{source.stem}_tile{index:02d}.jpg"
            tile.save(images_dir / name, quality=95)
            positive = index in clicked
            labels = [record.target_class] if positive else []
            if positive:
                report.positive_tiles += 1
                report.per_class[record.target_class] = (
                    report.per_class.get(record.target_class, 0) + 1
                )
            else:
                report.negative_tiles += 1
            samples.append(
                {
                    "图片": f"images/{name}",
                    "标签": labels,
                    "挑战类型": record.challenge_type,
                    "本轮目标": record.target_class,
                    "来源": "通过的挑战｜已点击" if positive else "通过的挑战｜未点击",
                }
            )

    (output_root / "tile_labels.json").write_text(
        json.dumps(
            {
                "说明": (
                    "标签来自通过的挑战：已点击格子含本轮目标，未点击格子不含。"
                    "空标签列表表示纯负样本，多标签训练下即全零多热向量。"
                ),
                "类别": sorted(seen_classes),
                "图块数": len(samples),
                "样本": samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "export_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _locate_image(capture_root: Path, record: SolveRecord) -> Path | None:
    """在归档目录中定位原图。"""
    if not record.image_name:
        return None
    direct = capture_root / record.image_name
    if direct.is_file():
        return direct
    matches = list(capture_root.rglob(record.image_name))
    return matches[0] if matches else None


def format_feedback_report(store: SolveFeedbackStore) -> str:
    """生成可直接打印的中文反馈统计。"""
    stats = store.statistics()
    lines = [
        f"记录总数: {stats['记录总数']}",
        f"可用于标注: {stats['可标注记录']} 条，预计产出 {stats['可产出图块']} 个图块",
        "",
        "按结果:",
    ]
    for name, count in stats["按结果"].items():
        lines.append(f"  {name:<8}{count}")
    if stats["按目标类别"]:
        lines.append("")
        lines.append("可标注记录的目标类别分布:")
        for name, count in stats["按目标类别"].items():
            lines.append(f"  {name:<16}{count}")
    if not stats["可标注记录"]:
        lines.append("")
        lines.append("尚无通过的挑战记录；开启在线会话并完成若干次验证后再导出。")
    return "\n".join(lines)
