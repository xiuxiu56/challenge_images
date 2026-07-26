"""生成不破坏原始数据的训练副本，可选按平方根规则平衡长尾类别。"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from ..config import BALANCED_DATA_DIR, DATA_DIR
from .dataset_info import IMG_EXTS, scan_split


def _link_or_copy(source: Path, target: Path, use_links: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if use_links:
        try:
            target.symlink_to(source.resolve())
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def prepare_dataset(
    output_dir: str | Path = BALANCED_DATA_DIR,
    data_dir: str | Path = DATA_DIR,
    balance: str = "sqrt",
    use_links: bool = True,
    max_multiplier: int = 20,
) -> Path:
    """复制 train/val 目录并对 train 做可回滚的长尾过采样。

    ``balance='none'`` 只做结构化副本；``sqrt`` 使用
    sqrt(类别最大样本数 / 当前样本数) 作为过采样倍率，避免极少类权重爆炸。
    原始文件不会被移动或删除。
    """
    source_root = Path(data_dir)
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，请换一个名称：{output_root}")
    output_root.mkdir(parents=True)
    counts = scan_split("train", source_root)
    max_count = max(counts.values(), default=1)
    manifest: dict[str, Any] = {"source": str(source_root), "balance": balance, "classes": {}}
    for split in ("train", "val", "test"):
        split_dir = source_root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files = sorted(p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if not files:
                continue
            repeats = 1
            if split == "train" and balance == "sqrt":
                repeats = max(1, min(max_multiplier, math.ceil(math.sqrt(max_count / len(files)))))
            written = 0
            for repeat in range(repeats):
                for index, source in enumerate(files):
                    suffix = f"__平衡{repeat:02d}" if repeat else ""
                    target = output_root / split / cls_dir.name / f"{index:06d}{suffix}{source.suffix.lower()}"
                    _link_or_copy(source, target, use_links=use_links)
                    written += 1
            manifest["classes"].setdefault(cls_dir.name, {})[split] = {
                "原始数量": len(files),
                "复制倍率": repeats,
                "输出数量": written,
            }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_root


if __name__ == "__main__":
    print(prepare_dataset())
