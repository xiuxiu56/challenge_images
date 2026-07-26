"""
YOLO26 分类验证 / 单图预测。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ..category_map import format_predict_line, normalize_dataset_class
from ..config import (
    DEFAULT_PREDICT,
    DEFAULT_TRAIN_IMGSZ,
    DEFAULT_VAL,
    device_status,
    pick_device,
    read_training_imgsz,
    resolve_default_weight,
    resolve_model_reference,
)
from ..runtime_env import prepare_cache_dir
from ..data.dataset_info import IMG_EXTS


def resolve_inference_imgsz(weights: str | Path, requested: int | None = None) -> int:
    """确定推理分辨率：显式请求 > 模型训练分辨率 > 全局默认。

    推理分辨率与训练分辨率不一致会掉点，因此优先跟随权重自带的元数据。
    """
    if requested is not None:
        return int(requested)
    recorded = read_training_imgsz(weights)
    return int(recorded) if recorded else int(DEFAULT_TRAIN_IMGSZ)


def evaluate_directory(
    weights: str | Path,
    data: str | Path,
    imgsz: int | None = None,
    device: str | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """评估外部分类目录，目录结构要求 ``类别/图片``。

    该函数不依赖 sklearn，输出整体准确率和逐类 Precision、Recall、F1。
    ``imgsz`` 为 ``None`` 时自动跟随权重的训练分辨率。
    """
    requested_root = Path(data)
    if not requested_root.is_dir():
        raise FileNotFoundError(f"外部验证目录不存在: {requested_root}")
    imgsz = resolve_inference_imgsz(weights, imgsz)
    # Ultralytics 分类数据集根目录通常是 train/val/类别/图片。逐类报告默认
    # 评估 val，避免把 train、val 误当成类别。
    root = requested_root / "val" if (requested_root / "val").is_dir() else requested_root
    prepare_cache_dir()
    from ultralytics import YOLO

    model = YOLO(str(resolve_weights(weights)))
    names = {int(k): str(v) for k, v in (model.names or {}).items()}
    labels = sorted({normalize_dataset_class(p.name) or p.name for p in root.iterdir() if p.is_dir()})
    matrix: dict[str, dict[str, int]] = {
        truth: {pred: 0 for pred in labels} for truth in labels
    }
    total = correct = 0
    for truth in labels:
        source_dirs = [p for p in root.iterdir() if p.is_dir() and (normalize_dataset_class(p.name) or p.name) == truth]
        files = [
            p
            for source_dir in source_dirs
            for p in source_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMG_EXTS
        ]
        if not files:
            continue
        results = model.predict(
            source=[str(path) for path in files],
            imgsz=imgsz,
            device=pick_device(device),
            verbose=False,
            stream=False,
        )
        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            pred_idx = int(probs.top1)
            pred = normalize_dataset_class(names.get(pred_idx, str(pred_idx))) or names.get(pred_idx, str(pred_idx))
            if pred not in matrix[truth]:
                matrix[truth][pred] = 0
            matrix[truth][pred] += 1
            total += 1
            correct += int(pred == truth)
    per_class: dict[str, dict[str, float | int]] = {}
    for cls in labels:
        tp = matrix[cls].get(cls, 0)
        fp = sum(matrix[other].get(cls, 0) for other in labels if other != cls)
        fn = sum(matrix[cls].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"样本数": sum(matrix[cls].values()), "precision": precision, "recall": recall, "f1": f1}
    valid = [x for x in per_class.values() if x["样本数"]]
    report: dict[str, Any] = {
        "weights": str(weights),
        "data": str(root),
        "requested_data": str(requested_root),
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(float(x["f1"]) for x in valid) / len(valid) if valid else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
    print(f"外部验证总数：{total}，准确率：{report['accuracy']:.4f}，宏平均 F1（macro_f1）：{report['macro_f1']:.4f}")
    for cls, item in per_class.items():
        print(f"  类别={cls}，样本数（n）={item['样本数']}，精确率（precision）={item['precision']:.4f}，召回率（recall）={item['recall']:.4f}，F1（f1）={item['f1']:.4f}")
    if report_path:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"外部验证报告已保存：{output}")
    return report


def resolve_weights(weights: str | Path | None, run_name: str = "exp") -> Path:
    if weights is None:
        w = resolve_default_weight(run_name)
    else:
        w = Path(resolve_model_reference(weights))
    if w is None or not w.is_file():
        raise FileNotFoundError(f"权重不存在: {w}")
    return w


def val_cls(
    weights: str | Path | None = None,
    data: str | Path | None = None,
    run_name: str = "exp",
    **extra: Any,
):
    """
    在 val 集上评估，打印 top1/top5 等指标。
    返回 ultralytics Metrics 对象。
    """
    w = resolve_weights(weights, run_name)
    cfg: dict[str, Any] = {**DEFAULT_VAL}
    if data is not None:
        cfg["data"] = str(data)
    cfg.update(extra)
    cfg["device"] = pick_device(str(cfg.get("device", "")))
    # 未显式指定时跟随权重的训练分辨率，避免训练/验证尺寸错配。
    cfg["imgsz"] = resolve_inference_imgsz(w, extra.get("imgsz"))

    print("=" * 60)
    print("开始验证")
    print(f"  设备状态: {device_status()}")
    print(f"  权重: {w}")
    print(f"  数据: {cfg['data']}")
    print(f"  输入尺寸（imgsz）/批次大小（batch）/验证设备（device）: {cfg.get('imgsz')}/{cfg.get('batch')}/{cfg.get('device')}")
    print("=" * 60)

    prepare_cache_dir()
    from ultralytics import YOLO

    model = YOLO(str(w))
    metrics = model.val(**cfg)
    # 分类指标常见字段
    top1 = getattr(metrics, "top1", None)
    top5 = getattr(metrics, "top5", None)
    if top1 is not None:
        print(f"最高准确率（top1）: {top1}")
    if top5 is not None:
        print(f"前五准确率（top5）: {top5}")
    return metrics


def predict_cls(
    source: str | Path,
    weights: str | Path | None = None,
    run_name: str = "exp",
    **extra: Any,
):
    """
    对单张图 / 目录做分类推理。
    返回 results 列表。
    """
    w = resolve_weights(weights, run_name)
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"预测源不存在: {src}")

    cfg: dict[str, Any] = {**DEFAULT_PREDICT}
    cfg.update(extra)
    cfg["device"] = pick_device(str(cfg.get("device", "")))
    cfg["imgsz"] = resolve_inference_imgsz(w, extra.get("imgsz"))

    print(f"预测图片（source）: {src}  模型权重（weights）: {w}")
    print(f"  设备状态: {device_status()}  → 预测设备（device）={cfg['device']}")
    prepare_cache_dir()
    from ultralytics import YOLO

    model = YOLO(str(w))
    results = model.predict(source=str(src), **cfg)

    for r in results:
        # 分类结果在 r.probs
        probs = getattr(r, "probs", None)
        if probs is None:
            print(f"  {getattr(r, 'path', src)}: 无分类结果")
            continue
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        names = getattr(r, "names", None) or {}
        label = names.get(top1_idx, str(top1_idx))
        # 带上中文 + mid，方便对接 recaptcha 词表
        source_path = getattr(r, "path", src)
        print(format_predict_line(Path(str(source_path)).name, str(label), top1_conf))

    return results


if __name__ == "__main__":
    # 默认验证 exp 的 best.pt
    val_cls()
