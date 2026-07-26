"""按类别样本数计算损失权重，缓解极端长尾。

本数据集的不均衡程度远超常规：去重后 Car 9894 张而 Tractor 只有 18 张，
相差 550 倍。不做任何处理时，模型把稀有类全部预测为负仍能取得很高的
总体准确率——这正是 top1 0.9227 与 macro-F1 0.8595 相差 7 个点的来源。

两种损失权重对应两种训练方式：

单标签（softmax + 交叉熵）
    ``class_weights`` 直接作为 ``F.cross_entropy(weight=...)``，
    按类别样本数的倒数缩放，稀有类的错误被放大。

多标签（sigmoid + BCE）
    ``pos_weight`` 是每一类「正样本相对负样本」的比例。多标签下每个类别
    都是独立的二分类，稀有类的负样本压倒性多数，不加权会让模型直接学会
    永远输出负。

两者都做了上限截断：Tractor 的原始权重高达 550，会让单个样本主导整个
批次的梯度，反而破坏训练稳定性。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset_info import IMG_EXTS

# 权重上限：超过这个倍数后收益递减，且梯度容易被极少数样本主导。
DEFAULT_MAX_WEIGHT = 20.0
# 平方根平滑：直接用倒数会让 550 倍差距原样进入损失。
SMOOTHING_SQRT = "sqrt"
SMOOTHING_LINEAR = "linear"
SMOOTHING_NONE = "none"


@dataclass(frozen=True)
class ClassBalance:
    """一个数据集的类别分布与权重。"""

    classes: list[str]
    counts: dict[str, int]
    weights: dict[str, float]
    smoothing: str
    max_weight: float

    @property
    def imbalance_ratio(self) -> float:
        """最大类与最小类的样本数之比。"""
        values = [count for count in self.counts.values() if count > 0]
        if not values:
            return 1.0
        return max(values) / min(values)

    def weight_vector(self) -> list[float]:
        """按 ``classes`` 顺序返回权重，可直接转成张量。"""
        return [self.weights.get(name, 1.0) for name in self.classes]

    def as_dict(self) -> dict[str, Any]:
        return {
            "类别数": len(self.classes),
            "不均衡倍数": round(self.imbalance_ratio, 1),
            "平滑方式": self.smoothing,
            "权重上限": self.max_weight,
            "逐类": {
                name: {"样本数": self.counts.get(name, 0), "权重": round(self.weights.get(name, 1.0), 3)}
                for name in self.classes
            },
        }


def count_images_per_class(root: str | Path, split: str = "train") -> dict[str, int]:
    """统计 ``<root>/<split>/<类别>/`` 下每个类别的图片数。"""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"数据集分割不存在：{split_dir}")
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        counts[class_dir.name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMG_EXTS
        )
    return counts


def _smooth(value: float, smoothing: str) -> float:
    if smoothing == SMOOTHING_SQRT:
        return value ** 0.5
    if smoothing == SMOOTHING_NONE:
        return 1.0
    return value


def compute_class_weights(
    counts: dict[str, int],
    *,
    classes: Iterable[str] | None = None,
    smoothing: str = SMOOTHING_SQRT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> ClassBalance:
    """按样本数倒数计算单标签交叉熵的类别权重。

    权重以样本数中位数为基准归一化，使多数类权重接近 1，
    避免整体损失量级随类别数漂移。
    """
    ordered = list(classes) if classes is not None else sorted(counts)
    positive = [counts.get(name, 0) for name in ordered if counts.get(name, 0) > 0]
    if not positive:
        return ClassBalance(ordered, dict(counts), {name: 1.0 for name in ordered}, smoothing, max_weight)

    reference = sorted(positive)[len(positive) // 2]  # 中位数
    weights: dict[str, float] = {}
    for name in ordered:
        count = counts.get(name, 0)
        if count <= 0:
            weights[name] = 1.0
            continue
        raw = reference / count
        weights[name] = min(float(max_weight), max(1.0 / max_weight, _smooth(raw, smoothing)))
    return ClassBalance(ordered, dict(counts), weights, smoothing, float(max_weight))


def compute_positive_weights(
    positive_counts: dict[str, int],
    total_samples: int,
    *,
    classes: Iterable[str] | None = None,
    smoothing: str = SMOOTHING_SQRT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> ClassBalance:
    """计算多标签 BCE 的 ``pos_weight``。

    定义为该类负样本数 / 正样本数：稀有类的正样本在损失中被放大到
    与负样本相当的量级，否则模型会学会永远输出负。
    """
    ordered = list(classes) if classes is not None else sorted(positive_counts)
    weights: dict[str, float] = {}
    for name in ordered:
        positive = positive_counts.get(name, 0)
        if positive <= 0:
            weights[name] = 1.0
            continue
        negative = max(0, int(total_samples) - positive)
        raw = negative / positive if positive else 1.0
        weights[name] = min(float(max_weight), max(1.0, _smooth(raw, smoothing)))
    return ClassBalance(ordered, dict(positive_counts), weights, smoothing, float(max_weight))


def format_balance_report(balance: ClassBalance, *, title: str = "类别权重") -> str:
    """生成可直接打印的中文权重报告。"""
    lines = [
        f"{title}（平滑={balance.smoothing}，上限={balance.max_weight:g}）",
        f"不均衡倍数: {balance.imbalance_ratio:.0f}×",
        "",
        f"{'类别':<16}{'样本数':>9}{'权重':>9}",
    ]
    for name in balance.classes:
        lines.append(
            f"{name:<16}{balance.counts.get(name, 0):>9}{balance.weights.get(name, 1.0):>9.2f}"
        )
    return "\n".join(lines)
