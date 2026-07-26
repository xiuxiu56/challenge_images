"""在线采集图片统计：类型、类别和精确重复组。"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any

from .sample_manager import IMAGE_EXTS


FULL_CHALLENGE_TYPES = {"dynamic", "imageselect", "tileselect", "multicaptcha"}
ARCHIVE_KIND_LABELS = {
    "full_challenge": "完整挑战图",
    "replacement_tile": "替换单格图",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_path(root: Path, path: Path) -> tuple[str, str, str] | None:
    """从在线归档相对路径解析归档类型、挑战类型和中文类别。"""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) < 3:
        return None
    if parts[0] == "replacements":
        if len(parts) < 4:
            return None
        return "replacement_tile", parts[1], parts[2]
    if parts[0] in FULL_CHALLENGE_TYPES:
        return "full_challenge", parts[0], parts[1]
    return None


def scan_online_capture(
    root: str | Path,
    *,
    archive_kind: str = "all",
    challenge_type: str = "全部",
) -> dict[str, Any]:
    """扫描在线图片并返回 GUI 可直接展示的统计结果。

    只读取图片和路径，不写报告、不修改图片。``archive_kind`` 支持
    ``all``、``full_challenge``、``replacement_tile``。
    """
    base = Path(root)
    records: list[dict[str, str]] = []
    if not base.is_dir():
        return _build_result(records)

    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        classified = _classify_path(base, path)
        if classified is None:
            continue
        current_kind, current_type, current_category = classified
        if archive_kind != "all" and archive_kind != current_kind:
            continue
        if challenge_type != "全部" and challenge_type != current_type:
            continue
        records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "archive_kind": current_kind,
                "archive_label": ARCHIVE_KIND_LABELS[current_kind],
                "challenge_type": current_type,
                "category": current_category,
            }
        )
    return _build_result(records)


def _build_result(records: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["sha256"]].append(record)

    duplicate_groups: list[dict[str, Any]] = []
    for digest, items in groups.items():
        if len(items) < 2:
            continue
        duplicate_groups.append(
            {
                "sha256": digest,
                "count": len(items),
                "extra": len(items) - 1,
                "archive_labels": sorted({item["archive_label"] for item in items}),
                "challenge_types": sorted({item["challenge_type"] for item in items}),
                "categories": sorted({item["category"] for item in items}),
                "files": [item["path"] for item in items],
            }
        )
    duplicate_groups.sort(key=lambda item: (-item["count"], item["sha256"]))

    category_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        category_groups[
            (record["archive_label"], record["challenge_type"], record["category"])
        ].append(record)
    category_rows = []
    for (archive_label, current_type, category), items in category_groups.items():
        category_rows.append(
            {
                "archive_label": archive_label,
                "challenge_type": current_type,
                "category": category,
                "count": len(items),
                "unique": len({item["sha256"] for item in items}),
            }
        )
    category_rows.sort(
        key=lambda item: (item["archive_label"], item["challenge_type"], item["category"])
    )

    return {
        "total": len(records),
        "unique": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files": sum(item["count"] for item in duplicate_groups),
        "extra": sum(item["extra"] for item in duplicate_groups),
        "category_rows": category_rows,
        "duplicate_rows": duplicate_groups,
    }
