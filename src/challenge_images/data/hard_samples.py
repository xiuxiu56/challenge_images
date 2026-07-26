"""把已确认的挑战格子导出为待审核困难训练素材。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import DATA_DIR, HARD_SAMPLES_DIR, M2_DATA_DIR, ROOT
from ..grid.grid_engine import GridSpec, split_grid


@dataclass(frozen=True)
class HardSample:
    """一项人工确认的困难格子。"""

    source: str
    grid_index: int
    label: str
    reason: str
    deferred: bool = False
    approved: bool = False


# Boat 暂时只保留在清单中，不创建 Boat 目录，也不导出到当前 14 类素材。
DEFAULT_HARD_SAMPLES = (
    HardSample(
        "data/challenge/dynamic/人行横道/m_014xcs_104.jpg",
        7,
        "Crosswalk",
        "车和人行横道同时出现，人行横道困难正样本",
        approved=True,
    ),
    HardSample(
        "data/challenge/dynamic/人行横道/m_014xcs_114.jpg",
        8,
        "Car",
        "只有车，用作人行横道困难负样本",
        approved=True,
    ),
    HardSample(
        "data/challenge/dynamic/车/m_0k4j_132.jpg",
        4,
        "Bridge",
        "模型高置信度误判为车，用作车的困难负样本",
        approved=True,
    ),
    HardSample(
        "data/challenge/dynamic/车/m_0k4j_135.jpg",
        1,
        "Boat",
        "当前训练集没有 Boat 类，暂缓处理",
        deferred=True,
    ),
    HardSample(
        "data/challenge/dynamic/消防栓/m_01pns0_538.jpg",
        0,
        "Hydrant",
        "远处小消防栓困难正样本",
        approved=True,
    ),
    HardSample(
        "data/challenge/dynamic/消防栓/m_01pns0_542.jpg",
        4,
        "Hydrant",
        "远处小消防栓困难正样本",
        approved=True,
    ),
)


def export_hard_samples(
    samples: tuple[HardSample, ...] = DEFAULT_HARD_SAMPLES,
    output_dir: str | Path = HARD_SAMPLES_DIR,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    """导出 3×3 格子和中文审核清单；暂缓项只记录、不导出图片。"""
    root = Path(project_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    exported = deferred = 0
    for sample in samples:
        source = Path(sample.source)
        if not source.is_absolute():
            source = root / source
        record = asdict(sample)
        record["源文件"] = str(source)
        if sample.deferred:
            deferred += 1
            record["状态"] = "暂缓：未导出图片"
            records.append(record)
            continue
        if not source.is_file():
            record["状态"] = "源图片缺失"
            records.append(record)
            continue
        tiles = split_grid(Image.open(source).convert("RGB"), GridSpec(3, 3))
        if not 0 <= sample.grid_index < len(tiles):
            record["状态"] = "格子编号越界"
            records.append(record)
            continue
        class_dir = output / sample.label
        class_dir.mkdir(parents=True, exist_ok=True)
        target = class_dir / f"{source.stem}__格子{sample.grid_index}.jpg"
        tiles[sample.grid_index].save(target, quality=95)
        exported += 1
        record["状态"] = "已审核通过" if sample.approved else "待人工审核"
        record["导出文件"] = str(target)
        records.append(record)

    manifest_path = output / "审核清单.json"
    manifest_path.write_text(
        json.dumps(
            {
                "说明": "这些图块不会自动进入训练集；审核标签和画面后再手动合并。",
                "Boat处理": "暂缓，不创建 Boat 类。",
                "记录": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "输出目录": str(output),
        "清单路径": str(manifest_path),
        "导出数量": exported,
        "暂缓数量": deferred,
    }


def export_default_hard_samples() -> dict[str, Any]:
    """按项目内置人工确认清单导出困难样本。"""
    return export_hard_samples()


def build_m2_dataset(
    base_data: str | Path = DATA_DIR,
    review_dir: str | Path = HARD_SAMPLES_DIR,
    output_dir: str | Path = M2_DATA_DIR,
) -> dict[str, Any]:
    """创建原数据集的链接副本，并把已审核困难图块加入 train。

    只接收审核清单中状态为“已审核通过”的图块。首次导出默认为“待人工审核”，
    防止标签尚未复核时自动污染训练数据。
    """
    import shutil

    source_root = Path(base_data)
    review_root = Path(review_dir)
    output_root = Path(output_dir)
    manifest_path = review_root / "审核清单.json"
    if not source_root.is_dir():
        raise FileNotFoundError(f"基础数据集不存在：{source_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"困难样本清单不存在：{manifest_path}")
    if output_root.exists():
        raise FileExistsError(f"m2 数据目录已存在，请先手动确认后换名或移走：{output_root}")

    output_root.mkdir(parents=True)
    linked = added = 0
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.is_dir():
            continue
        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            target_class_dir = output_root / split / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for source in class_dir.rglob("*"):
                if not source.is_file() or source.name.startswith("."):
                    continue
                target = target_class_dir / source.name
                # 同类出现重名时保留相对路径信息，避免覆盖。
                if target.exists():
                    target = target_class_dir / f"{source.parent.name}__{source.name}"
                target.symlink_to(source.resolve())
                linked += 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest.get("记录", []):
        if record.get("状态") != "已审核通过":
            continue
        label = str(record.get("label", "")).strip()
        source = Path(str(record.get("导出文件", "")))
        if label == "Boat" or not source.is_file():
            continue
        target_dir = output_root / "train" / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"困难样本__{source.name}"
        shutil.copy2(source, target)
        added += 1

    build_manifest = {
        "基础数据集": str(source_root),
        "困难样本目录": str(review_root),
        "基础链接数量": linked,
        "审核通过并加入数量": added,
        "Boat处理": "暂缓",
    }
    (output_root / "m2数据清单.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"输出目录": str(output_root), **build_manifest}
