"""YOLO26 实例分割训练入口。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
from typing import Any

from ..config import (
    DEFAULT_SEGMENTATION_TRAIN,
    RECOMMENDED_SEGMENTATION_MODEL,
    SEGMENTATION_TRAINED_DIR,
    device_status,
    next_available_run_name,
    pick_device,
    resolve_segmentation_model_reference,
)
from ..runtime_env import prepare_cache_dir


def train_seg(
    model: str = RECOMMENDED_SEGMENTATION_MODEL,
    data: str | Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> Path:
    """训练分割模型并同步 best/last 和分割元数据。"""
    cfg = dict(DEFAULT_SEGMENTATION_TRAIN)
    if data is not None:
        cfg["data"] = str(data)
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    if batch is not None:
        cfg["batch"] = int(batch)
    if imgsz is not None:
        cfg["imgsz"] = int(imgsz)
    if device is not None:
        cfg["device"] = pick_device(device)
    else:
        cfg["device"] = pick_device(str(cfg.get("device", "cpu")))
    if name is not None:
        cfg["name"] = name
    cfg.update(extra)
    cfg["device"] = pick_device(str(cfg.get("device", "cpu")))

    data_yaml = Path(str(cfg["data"]))
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"分割数据配置不存在：{data_yaml}。请先准备 images/、labels/ 和 data.yaml。"
        )
    requested_name = str(cfg["name"])
    cfg["name"] = next_available_run_name(
        requested_name,
        project_dir=cfg["project"],
        trained_dir=SEGMENTATION_TRAINED_DIR,
    )
    cfg["exist_ok"] = False

    print("=" * 64)
    print("开始训练 YOLO26 实例分割模型")
    print(f"设备状态：{device_status()}")
    print(f"模型（model）：{model}")
    print(f"数据配置（data）：{data_yaml}")
    print(
        f"训练轮数（epochs）：{cfg['epochs']}  批次大小（batch）：{cfg['batch']}  "
        f"输入尺寸（imgsz）：{cfg['imgsz']}"
    )
    print(f"设备（device）：{cfg['device']}  运行名称（name）：{cfg['name']}")
    print(
        f"优化器：{cfg.get('optimizer')}  初始学习率：{cfg.get('lr0')}  "
        f"余弦学习率：{cfg.get('cos_lr')}  早停：{cfg.get('patience')}"
    )
    print("=" * 64)

    prepare_cache_dir()
    from ultralytics import YOLO

    yolo = YOLO(resolve_segmentation_model_reference(model))
    task = str(getattr(yolo, "task", "") or "")
    if task and task != "segment":
        raise ValueError(f"所选权重任务为 {task}，请使用 -seg.pt 分割权重")
    results = yolo.train(**cfg)
    save_dir = Path(str(getattr(results, "save_dir", Path(cfg["project"]) / cfg["name"])))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        best = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    print(f"训练结束，best 权重：{best if best.is_file() else '未找到'}")
    if not best.is_file():
        return best

    run_dir = best.parent.parent
    names = getattr(yolo, "names", {}) or {}
    if isinstance(names, list):
        names = dict(enumerate(names))
    meta = {
        "创建时间": datetime.now().isoformat(timespec="seconds"),
        "任务": "segment",
        "模型": str(resolve_segmentation_model_reference(model)),
        "数据配置": str(data_yaml),
        "训练参数": cfg,
        "类别顺序": {str(key): str(value) for key, value in names.items()},
        "Python版本": platform.python_version(),
    }
    meta_path = run_dir / "segmentation_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    export_dir = SEGMENTATION_TRAINED_DIR / run_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_dir / "best.pt")
    last = best.parent / "last.pt"
    if last.is_file():
        shutil.copy2(last, export_dir / "last.pt")
    shutil.copy2(meta_path, export_dir / "segmentation_meta.json")
    print(f"分割元数据已保存：{meta_path}")
    print(f"分割模型已同步：{export_dir / 'best.pt'}")
    return best
