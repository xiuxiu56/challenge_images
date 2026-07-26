"""多标签分类训练：sigmoid + BCE 替代 softmax + 交叉熵。

单标签 softmax 是本项目大量补丁的根因。一个格子里同时有车和人行横道时，
softmax 强制两者概率之和不超过 1，目标类别被主类别压制；项目为此加了
四视角裁剪复核、类别对抑制阈值、多视角白名单等一系列绕行逻辑。

改成每类独立的 sigmoid + BCE 之后，Car 0.9 与 Crosswalk 0.85 可以并存，
判定退化成一句 ``probability[target] >= threshold``。

实现上复用 Ultralytics 的整套训练设施（增强、EMA、调度器、MPS 支持），
只替换三个扩展点：

1. ``build_dataset``  → 读取多标签清单，额外产出多热向量
2. ``init_criterion`` → 换成 ``BCEWithLogitsLoss``
3. ``get_validator``  → 换成按阈值统计 P/R/F1 的多标签验证器

图片仍然放在原来的 ``<split>/<类别>/`` 目录，因此 ``check_cls_dataset``
和类别顺序都不受影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_TRAIN,
    RECOMMENDED_MODEL,
    TRAINED_MODELS_DIR,
    device_status,
    next_available_run_name,
    pick_device,
    resolve_model_reference,
)
from ..data.multilabel import MANIFEST_FILENAME, MultiLabelManifest
from ..runtime_env import prepare_cache_dir

# 判定阈值：多标签下每类独立比较，不再依赖 Top-K 排名。
DEFAULT_POSITIVE_THRESHOLD = 0.5


def _build_dataset_class() -> type:
    """延迟构造多标签数据集类，避免顶层导入 torch/ultralytics。"""
    import torch
    from ultralytics.data.dataset import ClassificationDataset

    class MultiLabelClassificationDataset(ClassificationDataset):  # type: ignore[misc]
        """在单标签数据集基础上附加多热标签。

        继承而非重写，保证增强管线、缓存策略与官方实现完全一致；
        ``__getitem__`` 只在原有返回值上补一个 ``multi_cls`` 字段。
        """

        def __init__(self, root, args, augment=False, prefix="", manifest=None):
            super().__init__(root=root, args=args, augment=augment, prefix=prefix)
            self.manifest: MultiLabelManifest | None = manifest
            self._num_classes = (
                len(manifest.classes)
                if manifest is not None
                else max((int(sample[1]) for sample in self.samples), default=0) + 1
            )
            self._vectors = self._build_vectors()

        def _build_vectors(self) -> list[list[float]]:
            """预先算好每个样本的多热向量，训练时零开销。"""
            if self.manifest is None:
                return []
            vectors: list[list[float]] = []
            for sample in self.samples:
                path = Path(sample[0])
                folder_class = path.parent.name
                relative = f"{path.parent.parent.name}/{folder_class}/{path.name}"
                labels = self.manifest.labels_for(relative, folder_class)
                vectors.append(self.manifest.multi_hot(labels))
            return vectors

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = super().__getitem__(index)
            if self._vectors:
                item["multi_cls"] = torch.tensor(self._vectors[index], dtype=torch.float32)
            else:
                # 没有清单时退化为单标签 one-hot，训练目标与单标签完全一致。
                vector = torch.zeros(self._num_classes, dtype=torch.float32)
                vector[int(item["cls"])] = 1.0
                item["multi_cls"] = vector
            return item

    return MultiLabelClassificationDataset


def _build_trainer_class() -> type:
    """延迟构造训练器类。"""
    import torch
    from torch import nn
    from ultralytics.models import yolo
    from ultralytics.models.yolo.classify import ClassificationTrainer

    dataset_class = _build_dataset_class()

    class MultiLabelLoss:
        """每类独立的二元交叉熵。

        ``pos_weight`` 用于缓解长尾：稀有类的正样本在损失中被放大，
        避免模型把所有稀有类直接预测为负。
        """

        def __init__(self, pos_weight: "torch.Tensor | None" = None) -> None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        def __call__(self, preds: Any, batch: dict[str, Any]):
            # 训练期分类头返回 logits；推理期返回 (softmax, logits) 元组。
            logits = preds[1] if isinstance(preds, (list, tuple)) else preds
            target = batch["multi_cls"].to(logits.device, dtype=logits.dtype)
            loss = self.criterion(logits, target)
            return loss, loss.detach()

    class MultiLabelValidator(yolo.classify.ClassificationValidator):  # type: ignore[misc]
        """按固定阈值统计多标签 P/R/F1。

        单标签验证器的 top1/top5 在多标签下没有意义：一个格子有两个正确
        答案时，top1 最多只能命中一个。
        """

        threshold: float = DEFAULT_POSITIVE_THRESHOLD

        def update_metrics(self, preds: Any, batch: dict[str, Any]) -> None:
            logits = preds[1] if isinstance(preds, (list, tuple)) else preds
            probabilities = torch.sigmoid(logits.detach().float())
            self._probabilities.append(probabilities.cpu())
            self._targets.append(batch["multi_cls"].detach().float().cpu())
            # 同时维护父类的 top1 统计，保持训练日志可读。
            super().update_metrics(preds, batch)

        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            self._probabilities: list[Any] = []
            self._targets: list[Any] = []

        def multi_label_scores(self) -> dict[str, float]:
            """返回 micro / macro 维度的多标签指标。"""
            if not self._probabilities:
                return {}
            probabilities = torch.cat(self._probabilities)
            targets = torch.cat(self._targets)
            predicted = (probabilities >= self.threshold).float()
            true_positive = (predicted * targets).sum(dim=0)
            predicted_positive = predicted.sum(dim=0)
            actual_positive = targets.sum(dim=0)

            precision = true_positive / predicted_positive.clamp(min=1)
            recall = true_positive / actual_positive.clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
            present = actual_positive > 0
            return {
                "multilabel/precision": float(precision[present].mean()) if present.any() else 0.0,
                "multilabel/recall": float(recall[present].mean()) if present.any() else 0.0,
                "multilabel/macro_f1": float(f1[present].mean()) if present.any() else 0.0,
            }

    class MultiLabelTrainer(ClassificationTrainer):  # type: ignore[misc]
        """多标签分类训练器。"""

        def __init__(self, cfg=None, overrides=None, _callbacks=None):
            super().__init__(cfg, overrides, _callbacks)
            self.manifest: MultiLabelManifest | None = None
            self._load_manifest()

        def _load_manifest(self) -> None:
            root = Path(str(self.args.data))
            manifest_path = root / MANIFEST_FILENAME
            if manifest_path.is_file():
                self.manifest = MultiLabelManifest.load(manifest_path)

        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            return dataset_class(
                root=img_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
                manifest=self.manifest,
            )

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            batch = super().preprocess_batch(batch)
            if "multi_cls" in batch:
                batch["multi_cls"] = batch["multi_cls"].to(self.device)
            return batch

        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            # 用二元交叉熵替换官方的 softmax 交叉熵。
            model.init_criterion = lambda: MultiLabelLoss()  # type: ignore[assignment]
            model.criterion = None
            return model

        def get_validator(self):
            self.loss_names = ["bce_loss"]
            from copy import copy

            return MultiLabelValidator(
                self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
            )

    return MultiLabelTrainer


def train_multilabel(
    model: str = RECOMMENDED_MODEL,
    data: str | Path | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> Path:
    """训练多标签分类模型，返回 best.pt 路径。

    ``data`` 目录需要同时具备单标签文件夹结构和 ``multilabel.json`` 清单；
    缺少清单时每张图片退化为单标签，训练仍可进行。
    """
    cfg: dict[str, Any] = {**DEFAULT_TRAIN}
    if data is not None:
        cfg["data"] = str(data)
    if epochs is not None:
        cfg["epochs"] = epochs
    if imgsz is not None:
        cfg["imgsz"] = imgsz
    if batch is not None:
        cfg["batch"] = batch
    cfg["device"] = pick_device(device if device is not None else cfg.get("device"))
    cfg["name"] = name or "recaptcha_multilabel_160"
    cfg.update(extra)
    cfg["name"] = next_available_run_name(str(cfg["name"]), project_dir=cfg["project"])
    cfg["exist_ok"] = False
    cfg["model"] = resolve_model_reference(model)

    data_root = Path(cfg["data"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在: {data_root}")
    manifest_path = data_root / MANIFEST_FILENAME
    manifest = MultiLabelManifest.load(manifest_path) if manifest_path.is_file() else None

    prepare_cache_dir()
    print("=" * 60)
    print("开始训练多标签分类模型（sigmoid + BCE）")
    print(f"  设备状态: {device_status()}")
    print(f"  模型: {cfg['model']}")
    print(f"  数据: {data_root}")
    if manifest is None:
        print(f"  多标签清单: 未找到 {MANIFEST_FILENAME}，本次按单标签训练")
    else:
        print(f"  多标签清单: {manifest_path}（{len(manifest.overrides)} 张多标签图片）")
    print(f"  轮数/批次/尺寸: {cfg['epochs']}/{cfg['batch']}/{cfg['imgsz']}")
    print(f"  判定阈值: {DEFAULT_POSITIVE_THRESHOLD}（每类独立，不再依赖 Top-K）")
    print("=" * 60)

    trainer_class = _build_trainer_class()
    trainer = trainer_class(overrides=cfg)
    trainer.train()

    best = Path(trainer.save_dir) / "weights" / "best.pt"
    if best.is_file():
        export_dir = TRAINED_MODELS_DIR / Path(trainer.save_dir).name
        export_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(best, export_dir / "best.pt")
        if manifest is not None:
            manifest.save(export_dir / MANIFEST_FILENAME)
        print(f"多标签模型已同步：{export_dir / 'best.pt'}")
    return best
