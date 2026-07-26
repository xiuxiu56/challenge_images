"""给单标签训练注入类别权重。

Ultralytics 的 ``v8ClassificationLoss`` 是不带权重的 ``F.cross_entropy``。
在 550 倍不均衡的数据上，模型把 Tractor（18 张）全部预测错仍只损失
总体准确率的万分之几，因此长尾类在训练中几乎没有梯度信号。

项目里原本有 ``dataset_prepare(balance="sqrt")`` 的过采样方案，但它
从未接入默认训练流程，而且复制样本会让每轮训练看到大量重复图像，
增强也无法弥补信息量的缺失。加权损失不复制任何数据，是更直接的做法。

实现用回调而非自定义训练器：``on_train_start`` 时把已加权的损失函数
装到模型上，其余训练设施完全不动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data.class_weights import (
    ClassBalance,
    compute_class_weights,
    count_images_per_class,
    format_balance_report,
)


class WeightedClassificationLoss:
    """带类别权重的交叉熵，签名与 Ultralytics 的分类损失一致。"""

    def __init__(self, weight: Any) -> None:
        self.weight = weight

    def __call__(self, preds: Any, batch: dict[str, Any]):
        import torch.nn.functional as F

        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(
            logits,
            batch["cls"],
            weight=self.weight.to(logits.device, dtype=logits.dtype),
            reduction="mean",
        )
        return loss, loss.detach()


def build_class_balance(data_dir: str | Path, class_names: list[str] | None = None) -> ClassBalance | None:
    """按训练集目录统计类别权重；目录不可用时返回 None。"""
    try:
        counts = count_images_per_class(data_dir, "train")
    except FileNotFoundError:
        return None
    if not counts:
        return None
    return compute_class_weights(counts, classes=class_names)


def attach_class_weights(model: Any, data_dir: str | Path) -> ClassBalance | None:
    """给 YOLO 模型挂上加权交叉熵，返回所用的权重分布。

    类别顺序必须与模型的 ``names`` 一致，否则权重会错位到别的类别上；
    因此在 ``on_train_start`` 时才最终确定顺序——此时模型已经建好。
    """
    balance = build_class_balance(data_dir)
    if balance is None:
        print("[类别权重] 无法统计训练集分布，本次使用无权重交叉熵。")
        return None

    def _on_train_start(trainer: Any) -> None:
        import torch

        names = getattr(trainer.model, "names", None) or {}
        ordered = [str(names[key]) for key in sorted(names)] if names else balance.classes
        aligned = compute_class_weights(
            balance.counts,
            classes=ordered,
            smoothing=balance.smoothing,
            max_weight=balance.max_weight,
        )
        weight = torch.tensor(aligned.weight_vector(), dtype=torch.float32)
        trainer.model.criterion = WeightedClassificationLoss(weight)
        print(format_balance_report(aligned, title="单标签类别权重"))

    model.add_callback("on_train_start", _on_train_start)
    return balance


def attach_domain_augment(model: Any, config: Any = None) -> None:
    """给单标签训练的训练集注入域增强。

    Ultralytics 的 ``ClassificationDataset`` 在构造时就定好了
    ``torch_transforms``，因此在 ``on_train_start``（数据加载器已建好、
    尚未开始取数据）时插入。只改训练集，验证集必须保持确定性，
    否则每轮验证结果都会抖动，指标失去可比性。
    """
    from ..data.domain_augment import DomainAugmentConfig, describe, inject_domain_augment

    settings = config or DomainAugmentConfig()

    def _on_train_start(trainer: Any) -> None:
        loader = getattr(trainer, "train_loader", None)
        dataset = getattr(loader, "dataset", None)
        transforms = getattr(dataset, "torch_transforms", None)
        if transforms is None:
            print("[域增强] 未找到训练集变换，本次跳过。")
            return
        inject_domain_augment(transforms, settings)
        print(describe(settings))

    model.add_callback("on_train_start", _on_train_start)
