"""GUI 格子真实标注保存。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AnnotationStore:
    def __init__(self, path: str | Path = "annotations/grid_annotations.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get(self, image_path: str | Path) -> dict[str, Any] | None:
        return self.data.get(str(image_path))

    def set(self, image_path: str | Path, *, challenge_type: str, grid: str, target_class: str, indices: list[int]) -> None:
        self.data[str(image_path)] = {
            "挑战类型": challenge_type,
            "网格": grid,
            "目标类别": target_class,
            "真实格子": sorted(set(int(index) for index in indices)),
        }
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
