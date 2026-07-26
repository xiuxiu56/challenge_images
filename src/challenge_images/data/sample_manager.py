"""挑战大图样本枚举、SHA-256 去重与进度记录。"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..category_map import normalize_dataset_class
from ..config import REPORTS_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SampleManager:
    """按挑战类型管理样本，并默认跳过精确重复图片。"""

    def __init__(self, root: str | Path, challenge_type: str, deduplicate: bool = True, result_path: str | Path | None = None, status_filter: str = "全部") -> None:
        self.root = Path(root)
        self.challenge_type = challenge_type
        self.deduplicate = deduplicate
        self.result_path = Path(result_path) if result_path else REPORTS_DIR / "gui_results.jsonl"
        self.status_filter = status_filter
        self.samples: list[dict[str, Any]] = []
        self.position = -1
        self._load()

    def _load(self) -> None:
        seen: set[str] = set()
        directory = self.root / self.challenge_type
        if not directory.is_dir():
            return
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            digest = sha256_file(path)
            if self.deduplicate and digest in seen:
                continue
            seen.add(digest)
            raw_class = path.parent.name
            self.samples.append(
                {
                    "path": path,
                    "sha256": digest,
                    "raw_class": raw_class,
                    "target_class": normalize_dataset_class(raw_class) or raw_class,
                }
            )
        self._apply_status_filter()

    def _apply_status_filter(self) -> None:
        if self.status_filter == "全部" or not self.result_path.is_file():
            return
        statuses: dict[str, str] = {}
        try:
            for line in self.result_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                key = item.get("sha256") or item.get("path")
                if key:
                    statuses[str(key)] = str(item.get("status", ""))
        except (OSError, json.JSONDecodeError):
            return
        wanted = {"未处理": "", "成功": "success", "失败": "failed"}.get(self.status_filter)
        self.samples = [item for item in self.samples if statuses.get(item["sha256"], "") == wanted]

    def reset(self) -> None:
        self.position = -1

    def random_sample(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        self.position = random.randrange(len(self.samples))
        return self.current()

    def next_sample(self) -> dict[str, Any] | None:
        if not self.samples:
            return None
        self.position = (self.position + 1) % len(self.samples)
        return self.current()

    def current(self) -> dict[str, Any] | None:
        if 0 <= self.position < len(self.samples):
            return self.samples[self.position]
        return None

    def __len__(self) -> int:
        return len(self.samples)


def scan_duplicates(root: str | Path, challenge_type: str | None = None) -> dict[str, list[str]]:
    """扫描精确重复；指定挑战类型时只在该类型内部比较。"""
    base = Path(root)
    migrated = base / "data" / "challenge"
    if migrated.is_dir():
        base = migrated
    groups: dict[str, list[str]] = {}
    challenges = (challenge_type,) if challenge_type else ("dynamic", "imageselect", "multicaptcha")
    for challenge in challenges:
        directory = base / challenge
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                groups.setdefault(sha256_file(path), []).append(str(path))
    return {digest: paths for digest, paths in groups.items() if len(paths) > 1}


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
