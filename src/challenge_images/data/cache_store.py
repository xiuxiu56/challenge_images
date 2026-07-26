"""推理缓存的分片存储与维护。

缓存键是 SHA-256 十六进制串，原实现直接写成 ``<root>/<key>.json``。
在线采集跑久之后 ``reports/online_prediction_cache`` 积累了 19530 个文件
全部平铺在一个目录里：macOS 的 APFS 虽然能承受，但目录遍历、Finder 打开、
备份工具扫描都会明显变慢，rm 整个目录也很慢。

改为按键的前两位十六进制分片：256 个子目录，两万个文件时每个约 76 个。
再多一级意义不大（65536 个空目录反而是负担）。

同时提供统计与清理：缓存目前没有任何淘汰机制，只增不减。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SHARD_LENGTH = 2
CACHE_SUFFIX = ".json"


def sharded_path(root: str | Path, key: str, suffix: str = CACHE_SUFFIX) -> Path:
    """返回缓存键对应的分片路径。

    键短于分片长度时退回根目录，保证任何输入都能得到合法路径。
    """
    base = Path(root)
    cleaned = str(key).strip()
    if len(cleaned) <= SHARD_LENGTH:
        return base / f"{cleaned}{suffix}"
    return base / cleaned[:SHARD_LENGTH] / f"{cleaned}{suffix}"


def iter_cache_files(root: str | Path, suffix: str = CACHE_SUFFIX) -> Iterator[Path]:
    """遍历分片与历史平铺布局下的全部缓存文件。"""
    base = Path(root)
    if not base.is_dir():
        return
    for path in base.rglob(f"*{suffix}"):
        if path.is_file():
            yield path


@dataclass
class CacheStatistics:
    """缓存规模统计。"""

    root: Path
    files: int = 0
    bytes_total: int = 0
    shards: int = 0
    flat_files: int = 0

    @property
    def megabytes(self) -> float:
        return self.bytes_total / (1024 * 1024)

    def as_dict(self) -> dict[str, object]:
        return {
            "目录": str(self.root),
            "文件数": self.files,
            "占用MB": round(self.megabytes, 1),
            "分片目录数": self.shards,
            "未分片文件数": self.flat_files,
        }


def cache_statistics(root: str | Path, suffix: str = CACHE_SUFFIX) -> CacheStatistics:
    """统计缓存文件数、占用与分片情况。"""
    base = Path(root)
    stats = CacheStatistics(root=base)
    if not base.is_dir():
        return stats
    shards: set[str] = set()
    for path in iter_cache_files(base, suffix):
        stats.files += 1
        try:
            stats.bytes_total += path.stat().st_size
        except OSError:
            continue
        if path.parent == base:
            stats.flat_files += 1
        else:
            shards.add(path.parent.name)
    stats.shards = len(shards)
    return stats


def migrate_flat_cache(root: str | Path, suffix: str = CACHE_SUFFIX) -> int:
    """把根目录下平铺的缓存文件移入分片子目录，返回迁移数量。

    迁移是幂等的：已经在分片里的文件不会被再次处理。
    目标已存在时直接删除源文件——缓存内容由键唯一决定，两者等价。
    """
    base = Path(root)
    if not base.is_dir():
        return 0
    moved = 0
    for path in list(base.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        target = sharded_path(base, path.stem, suffix)
        if target == path:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            path.unlink(missing_ok=True)
            moved += 1
            continue
        try:
            path.replace(target)
        except OSError:
            continue
        moved += 1
    return moved


def prune_cache(
    root: str | Path,
    *,
    max_age_days: float | None = None,
    max_files: int | None = None,
    suffix: str = CACHE_SUFFIX,
) -> int:
    """按age或总量清理缓存，返回删除数量。

    ``max_age_days`` 删除超期文件；``max_files`` 保留最近访问的若干个。
    两者可同时使用，先按时间再按数量。
    """
    files = list(iter_cache_files(root, suffix))
    removed = 0

    if max_age_days is not None:
        deadline = time.time() - float(max_age_days) * 86400
        remaining: list[Path] = []
        for path in files:
            try:
                if path.stat().st_mtime < deadline:
                    path.unlink()
                    removed += 1
                    continue
            except OSError:
                continue
            remaining.append(path)
        files = remaining

    if max_files is not None and len(files) > int(max_files):
        def modified(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        for path in sorted(files, key=modified)[: len(files) - int(max_files)]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue

    _remove_empty_shards(root)
    return removed


def clear_cache(root: str | Path) -> int:
    """清空整个缓存目录，返回删除的文件数。"""
    base = Path(root)
    if not base.is_dir():
        return 0
    count = sum(1 for _ in iter_cache_files(base))
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return count


def _remove_empty_shards(root: str | Path) -> None:
    """删除清理后留下的空分片目录。"""
    base = Path(root)
    if not base.is_dir():
        return
    for child in base.iterdir():
        if child.is_dir() and not any(child.iterdir()):
            try:
                child.rmdir()
            except OSError:
                continue


def format_cache_report(stats: CacheStatistics) -> str:
    """生成可直接打印的中文缓存统计。"""
    lines = [
        f"目录: {stats.root}",
        f"文件数: {stats.files}，占用: {stats.megabytes:.1f} MB",
        f"分片目录: {stats.shards}",
    ]
    if stats.flat_files:
        lines.append(
            f"未分片文件: {stats.flat_files}（运行迁移后可分散到子目录）"
        )
    return "\n".join(lines)
