"""
YOLO26 图像分类训练。
入口函数 train_cls(...)，供 main 菜单或其它脚本调用。
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from ..config import (
    DEFAULT_TRAIN,
    RECOMMENDED_MODEL,
    SMOKE_TRAIN_OVERRIDES,
    device_status,
    pick_device,
    next_available_run_name,
    resolve_model_reference,
    TRAINED_MODELS_DIR,
)
from ..runtime_env import prepare_cache_dir
from .experiment import save_training_meta
from .class_balance import attach_class_weights, attach_domain_augment
from .macro_f1 import BEST_MACRO_F1_WEIGHT, HISTORY_FILENAME, attach_macro_f1_selection


def train_cls(
    model: str = RECOMMENDED_MODEL,
    data: str | Path | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    name: str | None = None,
    smoke: bool = False,
    resume: bool = False,
    balance_classes: bool = True,
    domain_augment: bool = True,
    **extra: Any,
) -> Path:
    """
    训练分类模型，返回 best.pt 路径（若存在）。

    smoke=True 时用短 epoch 冒烟，确认环境/数据管线。
    resume=True 时从 last.pt 续训（需 name 对应已有 run）。
    """
    cfg: dict[str, Any] = {**DEFAULT_TRAIN}
    if smoke:
        cfg.update(SMOKE_TRAIN_OVERRIDES)

    if data is not None:
        cfg["data"] = str(data)
    if epochs is not None:
        cfg["epochs"] = epochs
    if imgsz is not None:
        cfg["imgsz"] = imgsz
    if batch is not None:
        cfg["batch"] = batch
    # 统一走 pick_device：Mac 上优先 MPS，不可用则回退 cpu
    cfg["device"] = pick_device(device if device is not None else cfg.get("device"))
    if name is not None:
        cfg["name"] = name
    cfg.update(extra)
    # extra 里若又写了 device，再规范化一次
    if "device" in cfg:
        cfg["device"] = pick_device(str(cfg["device"]))
    if not resume:
        requested_name = str(cfg["name"])
        cfg["name"] = next_available_run_name(
            requested_name,
            project_dir=cfg["project"],
            trained_dir=TRAINED_MODELS_DIR,
        )
        cfg["exist_ok"] = False
        if cfg["name"] != requested_name:
            print(
                f"检测到同名训练或已导出模型：{requested_name}，"
                f"本次自动使用：{cfg['name']}"
            )

    data_path = Path(cfg["data"])
    if not data_path.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {data_path}")

    print("=" * 60)
    prepare_cache_dir()
    from ultralytics import YOLO

    print("开始训练 YOLO26 图像分类")
    print(f"  设备状态: {device_status()}")
    print(f"  模型:   {model}")
    print(f"  数据:   {cfg['data']}")
    print(f"  训练轮数（epochs）: {cfg['epochs']}  输入尺寸（imgsz）: {cfg['imgsz']}  批次大小（batch）: {cfg['batch']}")
    print(f"  训练设备（device）: {cfg['device']}  混合精度（amp）: {cfg.get('amp')}  运行名称（name）: {cfg['name']}")
    print(
        f"  正则化: dropout={cfg.get('dropout')}  weight_decay={cfg.get('weight_decay')}  "
        f"erasing={cfg.get('erasing')}"
    )
    print(
        f"  图像增强: scale={cfg.get('scale')}  fliplr={cfg.get('fliplr')}  "
        f"自动增强（auto_augment）={cfg.get('auto_augment')}"
    )
    print(
        f"  学习率: lr0={cfg.get('lr0')}  lrf={cfg.get('lrf')}  "
        f"cos_lr={cfg.get('cos_lr')}  warmup_epochs={cfg.get('warmup_epochs')}"
    )
    print(
        f"  运行策略: momentum={cfg.get('momentum')}  nbs={cfg.get('nbs')}  "
        f"deterministic={cfg.get('deterministic')}  cache={cfg.get('cache')}  "
        f"fraction={cfg.get('fraction')}  freeze={cfg.get('freeze')}"
    )
    print(f"  冒烟训练（smoke）:   {smoke}")
    print(f"  类别权重: {'启用（缓解 550 倍长尾）' if balance_classes else '关闭'}")
    print(f"  域增强: {'启用（JPEG/降采样/模糊/亮度对比度）' if domain_augment else '关闭'}")
    print("=" * 60)

    if resume:
        last = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"找不到可续训权重: {last}")
        yolo = YOLO(str(last))
        # 长尾数据集上 top1 由大类主导，额外按 macro-F1 维护一份最佳权重。
        tracker = attach_macro_f1_selection(yolo)
        if balance_classes:
            attach_class_weights(yolo, data_path)
        if domain_augment:
            attach_domain_augment(yolo)
        results = yolo.train(resume=True)
    else:
        yolo = YOLO(resolve_model_reference(model))
        tracker = attach_macro_f1_selection(yolo)
        if balance_classes:
            attach_class_weights(yolo, data_path)
        if domain_augment:
            attach_domain_augment(yolo)
        results = yolo.train(**cfg)
    print(f"[macro-F1] {tracker.summary()}")

    # 解析 best 路径
    best = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if not best.is_file():
        # ultralytics 有时把 save_dir 放在 results 上
        save_dir = getattr(results, "save_dir", None)
        if save_dir:
            cand = Path(save_dir) / "weights" / "best.pt"
            if cand.is_file():
                best = cand

    print(f"训练结束。best 权重: {best if best.is_file() else '未找到 best.pt'}")
    if best.is_file():
        run_dir = best.parent.parent
        # 以 Ultralytics 实际保存目录为准，避免并发启动时名称自动递增后导出串位。
        cfg["name"] = run_dir.name
        names = {int(k): str(v) for k, v in (getattr(yolo, "names", {}) or {}).items()}
        meta_path = save_training_meta(run_dir, model=str(resolve_model_reference(model)), config=cfg, data_dir=data_path, class_names=names)
        export_dir = TRAINED_MODELS_DIR / str(cfg["name"])
        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, export_dir / "best.pt")
        last = best.parent / "last.pt"
        if last.is_file():
            shutil.copy2(last, export_dir / "last.pt")
        shutil.copy2(meta_path, export_dir / "model_meta.json")
        print(f"训练元数据已保存：{run_dir / 'training_meta.json'}")
        print(f"最佳模型已同步：{export_dir / 'best.pt'}")
        # macro-F1 最佳权重与 best.pt 往往不是同一轮；长尾类以前者为准。
        macro_best = best.parent / BEST_MACRO_F1_WEIGHT
        if macro_best.is_file():
            shutil.copy2(macro_best, export_dir / BEST_MACRO_F1_WEIGHT)
            print(f"macro-F1 最佳模型已同步：{export_dir / BEST_MACRO_F1_WEIGHT}")
        history = run_dir / HISTORY_FILENAME
        if history.is_file():
            shutil.copy2(history, export_dir / HISTORY_FILENAME)
    return best


def train_recommended(smoke: bool = False) -> Path:
    """一键：推荐模型 + 默认超参。"""
    model = RECOMMENDED_MODEL if not smoke else "yolo26n-cls.pt"
    return train_cls(model=model, smoke=smoke)


if __name__ == "__main__":
    train_recommended(smoke=False)
