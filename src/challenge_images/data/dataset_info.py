"""
数据集扫描：类别数、样本分布、不均衡提示。
不依赖 ultralytics，纯 pathlib 统计。
"""

from __future__ import annotations

from pathlib import Path

from ..category_map import class_to_mid, class_to_zh
from ..config import DATA_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_classes(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in split_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def count_images(class_dir: Path) -> int:
    n = 0
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            n += 1
    return n


def scan_split(split: str, data_dir: Path = DATA_DIR) -> dict[str, int]:
    """返回 {类别名: 图片数}。"""
    split_dir = data_dir / split
    out: dict[str, int] = {}
    for cls in list_classes(split_dir):
        out[cls] = count_images(split_dir / cls)
    return out


def summarize(data_dir: Path = DATA_DIR) -> str:
    """生成可读的中文报告字符串。"""
    lines: list[str] = []
    lines.append(f"数据根目录: {data_dir}")
    lines.append(f"存在: {data_dir.is_dir()}")
    if not data_dir.is_dir():
        return "\n".join(lines)

    train = scan_split("train", data_dir)
    val = scan_split("val", data_dir)
    classes = sorted(set(train) | set(val))

    train_total = sum(train.values())
    val_total = sum(val.values())

    lines.append(f"类别数: {len(classes)}")
    lines.append(f"train 总数: {train_total}")
    lines.append(f"val   总数: {val_total}")
    lines.append("")
    lines.append(
        f"{'类别':<16} {'中文':<8} {'mid':<14} {'train':>8} {'val':>6} {'train占比':>10}"
    )
    lines.append("-" * 72)

    for cls in classes:
        t = train.get(cls, 0)
        v = val.get(cls, 0)
        pct = (t / train_total * 100) if train_total else 0.0
        zh = class_to_zh(cls)
        mid = class_to_mid(cls) or "—"
        lines.append(f"{cls:<16} {zh:<8} {mid:<14} {t:>8} {v:>6} {pct:>9.2f}%")

    # 不均衡诊断
    if train:
        counts = list(train.values())
        mx, mn = max(counts), min(counts)
        rare = [c for c, n in train.items() if n < 100]
        lines.append("")
        lines.append("不均衡诊断:")
        lines.append(f"  max/min = {mx}/{mn} = {mx / max(mn, 1):.1f}x")
        if rare:
            lines.append(f"  极少样本类(<100): {', '.join(rare)}")
        weak_val = [c for c, n in val.items() if n < 10]
        if weak_val:
            lines.append(f"  val 样本极少(<10，指标噪声大): {', '.join(weak_val)}")
        lines.append(
            "  建议: 先看整体 top1；再单独盯 rare 类召回；"
            "Tractor 这类几乎不可靠，可考虑合并到 Other 或补数据。"
        )

    return "\n".join(lines)


def class_weights(split: str = "train", data_dir: Path = DATA_DIR) -> dict[str, float]:
    """
    逆频率权重（仅供参考/后续扩展）。
    当前 ultralytics 分类训练不直接吃这个 dict，需要自定义 loss 时才用。
    """
    counts = scan_split(split, data_dir)
    total = sum(counts.values()) or 1
    n_cls = max(len(counts), 1)
    return {
        c: total / (n_cls * max(n, 1))
        for c, n in counts.items()
    }


if __name__ == "__main__":
    print(summarize())
