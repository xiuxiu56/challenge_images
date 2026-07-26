"""多标签分类数据：清单格式、构建与读取。

reCAPTCHA 的图块经常同时包含多个目标类别。最典型的是「公交车 + 人行横道」：
车停在斑马线上，一个格子里两个类别都成立。单标签目录结构无法表达这种情况，
只能把同一张图复制进两个文件夹，于是同样的像素得到两个互相矛盾的 one-hot
监督信号，模型被迫在两者之间二选一。

Softmax 强制所有类别概率之和为 1，进一步放大了这个问题：Car 拿到 0.7，
Crosswalk 就只能剩 0.3，即使人行横道在画面里清清楚楚。项目此前用四视角裁剪
复核、类别对抑制阈值等一系列补丁绕开它，本模块提供的是根因修复。

清单设计有意保持轻量：

- 图片仍然放在原来的 ``<split>/<类别>/`` 目录里，不移动、不复制
- 未列入清单的图片默认只有所在文件夹这一个标签
- 只有多标签图片才写进 ``覆盖`` 表

这样 Ultralytics 的 ``check_cls_dataset`` 依旧能正常识别数据集，
类别顺序也仍然由文件夹决定，多标签只是叠加在上面的一层信息。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dataset_info import IMG_EXTS

MANIFEST_FILENAME = "multilabel.json"
MANIFEST_VERSION = 1


@dataclass
class MultiLabelManifest:
    """多标签清单：类别顺序 + 多标签覆盖表。"""

    classes: list[str]
    overrides: dict[str, list[str]] = field(default_factory=dict)

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.classes)}

    def labels_for(self, relative_path: str, folder_class: str) -> list[str]:
        """返回一张图片的全部标签；未覆盖时只有所在文件夹类别。"""
        return self.overrides.get(str(relative_path).replace("\\", "/"), [folder_class])

    def multi_hot(self, labels: Iterable[str]) -> list[float]:
        """把标签列表转成多热向量。"""
        index = self.class_to_index
        vector = [0.0] * len(self.classes)
        for label in labels:
            position = index.get(label)
            if position is not None:
                vector[position] = 1.0
        return vector

    def as_dict(self) -> dict[str, Any]:
        return {
            "版本": MANIFEST_VERSION,
            "说明": (
                "图片仍按 <split>/<类别>/ 存放；未列入「覆盖」的图片只有文件夹这一个标签。"
                "「覆盖」记录同时包含多个目标类别的图块。"
            ),
            "类别": list(self.classes),
            "多标签图片数": len(self.overrides),
            "覆盖": {key: list(value) for key, value in sorted(self.overrides.items())},
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "MultiLabelManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            classes=list(payload.get("类别", [])),
            overrides={
                str(key): list(value)
                for key, value in (payload.get("覆盖") or {}).items()
            },
        )

    @classmethod
    def load_for_dataset(cls, root: str | Path) -> "MultiLabelManifest | None":
        """读取数据集根目录下的多标签清单；不存在时返回 None。"""
        path = Path(root) / MANIFEST_FILENAME
        return cls.load(path) if path.is_file() else None


def list_classes(root: Path) -> list[str]:
    """按 train 目录确定类别顺序，与 Ultralytics 的 ImageFolder 保持一致。"""
    train_dir = root / "train"
    source = train_dir if train_dir.is_dir() else root
    return sorted(
        item.name
        for item in source.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    )


def iter_dataset_images(root: Path, splits: Iterable[str] = ("train", "val")) -> list[tuple[str, str, Path]]:
    """遍历数据集，返回 (相对路径, 文件夹类别, 绝对路径)。"""
    items: list[tuple[str, str, Path]] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.name.startswith("."):
                    continue
                if path.suffix.lower() not in IMG_EXTS:
                    continue
                relative = f"{split}/{class_dir.name}/{path.name}"
                items.append((relative, class_dir.name, path))
    return items


def _digest(path: Path, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_manifest_from_folders(
    root: str | Path,
    *,
    splits: Iterable[str] = ("train", "val"),
) -> MultiLabelManifest:
    """从单标签目录结构推导多标签清单。

    同一份图片内容出现在多个类别目录下，说明它确实同时包含这些类别。
    这些图块正是复合场景的真实样本，无需人工标注即可作为多标签起点。
    """
    dataset_root = Path(root)
    classes = list_classes(dataset_root)
    items = iter_dataset_images(dataset_root, splits)

    # 内容哈希 → 出现过的类别集合
    by_digest: dict[str, set[str]] = {}
    digests: dict[str, str] = {}
    for relative, folder_class, path in items:
        try:
            file_digest = _digest(path)
        except OSError:
            continue
        digests[relative] = file_digest
        by_digest.setdefault(file_digest, set()).add(folder_class)

    overrides: dict[str, list[str]] = {}
    for relative, folder_class, _ in items:
        digest = digests.get(relative)
        if digest is None:
            continue
        labels = by_digest.get(digest, {folder_class})
        if len(labels) > 1:
            overrides[relative] = sorted(labels)
    return MultiLabelManifest(classes=classes, overrides=overrides)


def manifest_statistics(manifest: MultiLabelManifest) -> dict[str, Any]:
    """统计多标签分布，便于判断复合场景的规模。"""
    combos: dict[str, int] = {}
    per_class: dict[str, int] = {}
    for labels in manifest.overrides.values():
        key = " + ".join(labels)
        combos[key] = combos.get(key, 0) + 1
        for label in labels:
            per_class[label] = per_class.get(label, 0) + 1
    return {
        "多标签图片数": len(manifest.overrides),
        "组合分布": dict(sorted(combos.items(), key=lambda item: -item[1])),
        "涉及类别": dict(sorted(per_class.items(), key=lambda item: -item[1])),
    }


def format_manifest_report(manifest: MultiLabelManifest) -> str:
    """生成可直接打印的中文多标签统计。"""
    stats = manifest_statistics(manifest)
    lines = [
        f"类别数: {len(manifest.classes)}",
        f"多标签图片数: {stats['多标签图片数']}",
        "",
        "标签组合分布:",
    ]
    if stats["组合分布"]:
        for combo, count in stats["组合分布"].items():
            lines.append(f"  {combo:<40} {count} 张")
    else:
        lines.append("  无（当前数据中没有检测到复合图块）")
    return "\n".join(lines)
