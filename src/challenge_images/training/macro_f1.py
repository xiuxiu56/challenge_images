"""用 macro-F1 挑选最佳权重，替代被大类主导的 top1。

Ultralytics 的 ``ClassifyMetrics.fitness`` 是 ``(top1 + top5) / 2``，
``best.pt`` 也据此选择。本数据集极不均衡（Car 10193 / Tractor 23），
这个指标基本由大类决定：实测 top1 0.9227 而 macro-F1 只有 0.8595，
两者相差 7 个点，差距全部来自长尾类。

``on_fit_epoch_end`` 回调在 ``save_model()`` 之后才触发，此时覆盖
``trainer.fitness`` 已经来不及。因此这里不改写框架的选择逻辑，而是并行
维护一份自己的最佳权重：

1. 每轮验证结束后，直接复用验证器已经算好的 ``targets`` / ``pred``
   计算 macro-F1，不产生任何额外推理开销。
2. macro-F1 创新高时，把当轮的 ``weights/last.pt`` 复制为
   ``weights/best_macro_f1.pt``。
3. 全程记录到 ``macro_f1_history.json``，便于与 top1 曲线对照。

不依赖 sklearn，与 ``val_cls.evaluate_directory`` 的算法保持一致。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BEST_MACRO_F1_WEIGHT = "best_macro_f1.pt"
HISTORY_FILENAME = "macro_f1_history.json"


@dataclass
class ClassScore:
    """单个类别的 Precision / Recall / F1。"""

    name: str
    support: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "样本数": self.support,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


def macro_f1_from_counts(
    true_indices: list[int],
    predicted_indices: list[int],
    names: dict[int, str],
) -> tuple[float, list[ClassScore]]:
    """按逐类 TP/FP/FN 计算 macro-F1。

    只统计验证集中真实出现过的类别；没有样本的类别计入会把 F1 拉向 0，
    掩盖真实表现。
    """
    supports: dict[int, int] = {}
    true_positive: dict[int, int] = {}
    predicted: dict[int, int] = {}
    for truth, prediction in zip(true_indices, predicted_indices):
        supports[truth] = supports.get(truth, 0) + 1
        predicted[prediction] = predicted.get(prediction, 0) + 1
        if truth == prediction:
            true_positive[truth] = true_positive.get(truth, 0) + 1

    scores: list[ClassScore] = []
    for index in sorted(supports):
        tp = true_positive.get(index, 0)
        fp = predicted.get(index, 0) - tp
        fn = supports[index] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(
            ClassScore(
                name=str(names.get(index, index)),
                support=supports[index],
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    macro = sum(score.f1 for score in scores) / len(scores) if scores else 0.0
    return macro, scores


def _flatten(values: Any) -> list[int]:
    """把验证器保存的张量列表压平成 Python int 列表。"""
    flat: list[int] = []
    for item in values or []:
        tolist = getattr(item, "tolist", None)
        raw = tolist() if callable(tolist) else item
        if isinstance(raw, (list, tuple)):
            flat.extend(int(value) for value in raw)
        else:
            flat.append(int(raw))
    return flat


def _top1_predictions(values: Any) -> list[int]:
    """验证器的 ``pred`` 是每个样本的 Top-5 索引，这里只取 Top-1。"""
    flat: list[int] = []
    for item in values or []:
        tolist = getattr(item, "tolist", None)
        raw = tolist() if callable(tolist) else item
        if not isinstance(raw, (list, tuple)):
            flat.append(int(raw))
            continue
        for row in raw:
            if isinstance(row, (list, tuple)):
                if row:
                    flat.append(int(row[0]))
            else:
                flat.append(int(row))
    return flat


@dataclass
class MacroF1Tracker:
    """记录每轮 macro-F1，并维护自己的最佳权重副本。"""

    history: list[dict[str, Any]] = field(default_factory=list)
    best_value: float = -1.0
    best_epoch: int | None = None
    best_weight: Path | None = None

    def evaluate(self, trainer: Any) -> float | None:
        """从训练器读取验证结果并记录一轮 macro-F1。"""
        validator = getattr(trainer, "validator", None)
        if validator is None:
            return None
        truths = _flatten(getattr(validator, "targets", None))
        predictions = _top1_predictions(getattr(validator, "pred", None))
        if not truths or len(truths) != len(predictions):
            return None
        names = {int(k): str(v) for k, v in (getattr(validator, "names", None) or {}).items()}
        macro, scores = macro_f1_from_counts(truths, predictions, names)

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        metrics = getattr(trainer, "metrics", None) or {}
        record: dict[str, Any] = {
            "轮次": epoch,
            "macro_f1": round(macro, 6),
            "样本数": len(truths),
            "逐类": {score.name: score.as_dict() for score in scores},
        }
        for key, value in metrics.items():
            if "accuracy" in str(key):
                try:
                    record[str(key)] = round(float(value), 6)
                except (TypeError, ValueError):
                    continue
        self.history.append(record)

        if macro > self.best_value:
            self.best_value = macro
            self.best_epoch = epoch
            self.best_weight = self._promote(trainer)
        self._write_history(trainer)
        return macro

    def _promote(self, trainer: Any) -> Path | None:
        """把当轮 last.pt 复制为 macro-F1 最佳权重。"""
        weights_dir = self._weights_dir(trainer)
        if weights_dir is None:
            return None
        last = weights_dir / "last.pt"
        if not last.is_file():
            return None
        target = weights_dir / BEST_MACRO_F1_WEIGHT
        try:
            shutil.copy2(last, target)
        except OSError:
            return None
        return target

    @staticmethod
    def _weights_dir(trainer: Any) -> Path | None:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is None:
            return None
        weights_dir = Path(save_dir) / "weights"
        return weights_dir if weights_dir.is_dir() else None

    def _write_history(self, trainer: Any) -> None:
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir is None:
            return
        payload = {
            "最佳macro_f1": round(self.best_value, 6) if self.best_value >= 0 else None,
            "最佳轮次": self.best_epoch,
            "最佳权重": str(self.best_weight) if self.best_weight else None,
            "说明": (
                "Ultralytics 的 best.pt 按 (top1+top5)/2 选择，会被大类主导；"
                "本文件记录按 macro-F1 选择的权重，长尾类表现更可信。"
            ),
            "历史": self.history,
        }
        try:
            (Path(save_dir) / HISTORY_FILENAME).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def summary(self) -> str:
        """训练结束后的一行中文结论。"""
        if self.best_epoch is None:
            return "未能记录 macro-F1（验证结果不可用）"
        return (
            f"macro-F1 最佳轮次={self.best_epoch}，macro_f1={self.best_value:.4f}，"
            f"权重={self.best_weight}"
        )


def attach_macro_f1_selection(model: Any) -> MacroF1Tracker:
    """给 YOLO 模型挂上 macro-F1 跟踪回调，返回跟踪器。"""
    tracker = MacroF1Tracker()

    def _on_fit_epoch_end(trainer: Any) -> None:
        try:
            tracker.evaluate(trainer)
        except Exception as error:  # 指标跟踪绝不能中断训练
            print(f"[macro-F1] 本轮跟踪失败，训练继续：{type(error).__name__}: {error}")

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    return tracker
