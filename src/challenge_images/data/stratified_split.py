"""分层重划 train/val，保证每个类别都有足够的验证样本。

原始 ``dataset_cls_full_57k`` 的验证集按原始比例切分，结果是
Mountain 与 Tractor 各只有 3 张。用这样的验证集做早停和 best 权重选择，
等于让 Car(10193)/Bicycle(9075) 这些大类单独决定模型好坏，
稀有类的指标完全是噪声。

本模块把 train 与 val 合并成一个池，按类别重新分配：

- 每类验证样本数 = clamp(总数 × ``val_ratio``, 下限, 上限)
- 下限与上限都不会超过该类总数的 ``max_val_fraction``，
  避免稀有类把训练样本全部划走
- 按内容哈希排序后取样，同一份数据每次划分结果一致

输出使用符号链接，不复制、不移动、不删除任何原始文件。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_info import IMG_EXTS

SOURCE_SPLITS = ("train", "val", "test")


@dataclass
class ClassSplitPlan:
    """单个类别的重划分结果。"""

    name: str
    total: int
    train: int
    val: int
    duplicates_removed: int = 0
    multi_label_held_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "总数": self.total,
            "训练": self.train,
            "验证": self.val,
            "去重移除": self.duplicates_removed,
            "多标签保留在训练": self.multi_label_held_out,
        }


@dataclass
class SplitReport:
    """整体重划分报告。"""

    source: Path
    output: Path
    val_ratio: float
    classes: dict[str, ClassSplitPlan] = field(default_factory=dict)
    # 同一张图被标注为多个类别时记录 {内容哈希: [类别, ...]}。
    multi_label_conflicts: dict[str, list[str]] = field(default_factory=dict)

    @property
    def val_total(self) -> int:
        return sum(plan.val for plan in self.classes.values())

    @property
    def train_total(self) -> int:
        return sum(plan.train for plan in self.classes.values())

    @property
    def smallest_val_class(self) -> tuple[str, int]:
        """返回验证样本最少的类别，用于判断指标是否可信。"""
        if not self.classes:
            return ("", 0)
        name = min(self.classes, key=lambda key: self.classes[key].val)
        return (name, self.classes[name].val)

    def as_dict(self) -> dict[str, Any]:
        return {
            "源目录": str(self.source),
            "输出目录": str(self.output),
            "验证比例": self.val_ratio,
            "训练总数": self.train_total,
            "验证总数": self.val_total,
            "验证最少类别": dict(zip(("类别", "数量"), self.smallest_val_class)),
            "多标签冲突图片数": len(self.multi_label_conflicts),
            "类别": {name: plan.as_dict() for name, plan in sorted(self.classes.items())},
        }


def _file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_class_files(source: Path) -> dict[str, list[Path]]:
    """合并 train/val/test 下同名类别的全部图片。"""
    pool: dict[str, list[Path]] = {}
    for split in SOURCE_SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            files = [
                path
                for path in sorted(class_dir.rglob("*"))
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in IMG_EXTS
            ]
            if files:
                pool.setdefault(class_dir.name, []).extend(files)
    return pool


def _dedupe(files: list[Path]) -> tuple[dict[str, Path], int]:
    """按内容哈希去重，返回 ({哈希: 文件}, 移除数量)。

    返回按哈希排序的字典，保证同一份数据每次划分结果一致，
    且不依赖文件名或目录遍历顺序。
    """
    by_digest: dict[str, Path] = {}
    removed = 0
    for path in files:
        try:
            digest = _file_digest(path)
        except OSError:
            continue
        if digest in by_digest:
            removed += 1
            continue
        by_digest[digest] = path
    return {key: by_digest[key] for key in sorted(by_digest)}, removed


def find_multi_label_conflicts(
    class_digests: dict[str, dict[str, Path]],
) -> dict[str, list[str]]:
    """找出被标注为多个类别的同一张图片。

    单标签目录结构下，「公交车 + 人行横道」这类复合图块只能被复制进两个
    文件夹，产生互相矛盾的 one-hot 监督信号。这些图片是多标签训练的
    天然种子集，同时也不适合放进单标签验证集——无论预测哪个类都会被判错。
    """
    owners: dict[str, list[str]] = {}
    for class_name, digests in class_digests.items():
        for digest in digests:
            owners.setdefault(digest, []).append(class_name)
    return {
        digest: sorted(names)
        for digest, names in owners.items()
        if len(names) > 1
    }


def plan_val_count(
    total: int,
    *,
    val_ratio: float,
    val_min: int,
    val_max: int,
    max_val_fraction: float,
) -> int:
    """计算单个类别应划入验证集的数量。

    上限 ``max_val_fraction`` 优先于 ``val_min``：稀有类宁可验证样本偏少，
    也不能把训练样本划走大半。
    """
    if total <= 1:
        return 0
    ceiling = max(1, int(total * max_val_fraction))
    target = round(total * val_ratio)
    target = max(target, min(val_min, ceiling))
    target = min(target, val_max, ceiling)
    return max(1, int(target))


def _link_or_copy(source: Path, target: Path, use_links: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if use_links:
        try:
            target.symlink_to(source.resolve())
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def build_stratified_dataset(
    source: str | Path,
    output: str | Path,
    *,
    val_ratio: float = 0.15,
    val_min: int = 50,
    val_max: int = 300,
    max_val_fraction: float = 1 / 3,
    use_links: bool = True,
    dedupe: bool = True,
    overwrite: bool = False,
) -> SplitReport:
    """按类别分层重划 train/val，输出符号链接数据集。"""
    source_root = Path(source)
    output_root = Path(output)
    if not source_root.is_dir():
        raise FileNotFoundError(f"源数据目录不存在：{source_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请换一个名称或设置 overwrite：{output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    pool = collect_class_files(source_root)
    if not pool:
        raise ValueError(f"源目录中没有找到任何类别图片：{source_root}")

    report = SplitReport(source=source_root, output=output_root, val_ratio=val_ratio)

    # 先按类别去重，再统一检测跨类别标签冲突。
    class_digests: dict[str, dict[str, Path]] = {}
    removed_counts: dict[str, int] = {}
    for class_name, files in sorted(pool.items()):
        if dedupe:
            digests, removed = _dedupe(files)
        else:
            digests, removed = {f"{index:08d}": path for index, path in enumerate(sorted(files))}, 0
        class_digests[class_name] = digests
        removed_counts[class_name] = removed

    conflicts = find_multi_label_conflicts(class_digests) if dedupe else {}
    report.multi_label_conflicts = conflicts

    for class_name, digests in class_digests.items():
        # 多标签冲突图排除出验证集：单标签评分下无论预测哪个类都会判错。
        clean = [path for digest, path in digests.items() if digest not in conflicts]
        conflicted = [path for digest, path in digests.items() if digest in conflicts]
        total = len(digests)
        val_count = plan_val_count(
            len(clean),
            val_ratio=val_ratio,
            val_min=val_min,
            val_max=val_max,
            max_val_fraction=max_val_fraction,
        )
        val_files = clean[:val_count]
        # 冲突图仍然是真实数据，保留在训练集中。
        train_files = clean[val_count:] + conflicted
        for split, split_files in (("val", val_files), ("train", train_files)):
            for index, path in enumerate(split_files):
                target = output_root / split / class_name / f"{index:06d}{path.suffix.lower()}"
                _link_or_copy(path, target, use_links=use_links)
        report.classes[class_name] = ClassSplitPlan(
            name=class_name,
            total=total,
            train=len(train_files),
            val=len(val_files),
            duplicates_removed=removed_counts[class_name],
            multi_label_held_out=len(conflicted),
        )

    (output_root / "split_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if conflicts:
        # 导出为多标签训练的种子集：这些图确实同时包含多个目标类别。
        (output_root / "multi_label_seeds.json").write_text(
            json.dumps(
                {
                    "说明": (
                        "同一张图片被标注为多个类别。单标签目录结构无法表达，"
                        "但它们确实同时包含多个目标，是多标签训练的种子集。"
                    ),
                    "数量": len(conflicts),
                    "冲突": {digest: names for digest, names in sorted(conflicts.items())},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def format_split_report(report: SplitReport) -> str:
    """生成可直接打印的中文分层报告。"""
    lines = [
        f"源目录: {report.source}",
        f"输出目录: {report.output}",
        f"训练总数: {report.train_total}，验证总数: {report.val_total}",
    ]
    smallest_name, smallest_count = report.smallest_val_class
    lines.append(f"验证样本最少的类别: {smallest_name}（{smallest_count} 张）")
    if report.multi_label_conflicts:
        lines.append(
            f"多标签冲突图片: {len(report.multi_label_conflicts)} 张"
            "（已排除出验证集，保留在训练集）"
        )
    lines.append("")
    lines.append(f"{'类别':<16}{'总数':>8}{'训练':>8}{'验证':>8}{'去重':>8}{'多标签':>8}")
    for name, plan in sorted(report.classes.items()):
        lines.append(
            f"{name:<16}{plan.total:>8}{plan.train:>8}{plan.val:>8}"
            f"{plan.duplicates_removed:>8}{plan.multi_label_held_out:>8}"
        )
    return "\n".join(lines)
