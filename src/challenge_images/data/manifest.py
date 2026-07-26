"""生成分类和挑战图片清单。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import CHALLENGE_DIR, DATA_DIR, REPORTS_DIR
from .sample_manager import IMAGE_EXTS, sha256_file


def build_manifest(root: str | Path, kind: str, output: str | Path) -> Path:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            continue
        relative = path.relative_to(base)
        if kind == "classification":
            row = {"path": str(path), "kind": kind, "split": relative.parts[0], "label": relative.parts[1] if len(relative.parts) > 1 else ""}
        else:
            row = {"path": str(path), "kind": kind, "challenge_type": relative.parts[0], "label": relative.parts[1] if len(relative.parts) > 1 else ""}
        row.update(
            {"sha256": sha256_file(path), "width": str(width), "height": str(height)}
        )
        rows.append(row)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination


def build_default_manifests() -> list[Path]:
    return [
        build_manifest(DATA_DIR, "classification", REPORTS_DIR / "classification_manifest.jsonl"),
        build_manifest(CHALLENGE_DIR, "challenge", REPORTS_DIR / "challenge_manifest.jsonl"),
    ]
