"""数据集审计：只读检查，不删除原始图片。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import DATA_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def audit_dataset(data_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    """扫描坏图、尺寸、格式、哈希重复和类别分布。"""
    root = Path(data_dir)
    report: dict[str, Any] = {
        "data_dir": str(root),
        "splits": {},
        "classes": {},
        "bad_images": [],
        "duplicate_groups": 0,
        "duplicate_files": 0,
        "cross_class_duplicate_groups": 0,
        "sizes": Counter(),
        "formats": Counter(),
        "modes": Counter(),
    }
    hashes: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        split_count = 0
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            split_count += 1
            cls = path.parent.name
            report["classes"].setdefault(cls, {}).setdefault(split, 0)
            report["classes"][cls][split] += 1
            try:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                hashes[digest].append((split, cls, str(path)))
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    report["sizes"][f"{image.width}x{image.height}"] += 1
                    report["formats"][str(image.format)] += 1
                    report["modes"][str(image.mode)] += 1
            except Exception as exc:
                report["bad_images"].append(
                    {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )
        report["splits"][split] = split_count

    groups = [items for items in hashes.values() if len(items) > 1]
    cross_class = [items for items in groups if len({x[1] for x in items}) > 1]
    report["duplicate_groups"] = len(groups)
    report["duplicate_files"] = sum(len(items) for items in groups)
    report["cross_class_duplicate_groups"] = len(cross_class)
    report["sizes"] = dict(report["sizes"])
    report["formats"] = dict(report["formats"])
    report["modes"] = dict(report["modes"])
    return report


def save_audit_report(path: str | Path, data_dir: str | Path = DATA_DIR) -> Path:
    """保存 JSON 审计报告。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit_dataset(data_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def format_audit_report(report: dict[str, Any]) -> str:
    """生成菜单中展示的中文审计摘要。"""
    lines = ["======== 数据集审计（只读） ========"]
    lines.append(f"数据目录：{report['data_dir']}")
    lines.append(f"分区计数：{report['splits']}")
    lines.append(f"损坏图片：{len(report['bad_images'])}")
    lines.append(
        f"精确重复组：{report['duplicate_groups']}，涉及文件：{report['duplicate_files']}"
    )
    lines.append(f"跨类别重复组：{report['cross_class_duplicate_groups']}")
    lines.append(f"尺寸：{report['sizes']}")
    lines.append(f"格式：{report['formats']}，色彩模式：{report['modes']}")
    lines.append("说明：审计不会删除图片；像素不同的复制样本保留，由训练增强和验证结果决定取舍。")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_audit_report(audit_dataset()))
