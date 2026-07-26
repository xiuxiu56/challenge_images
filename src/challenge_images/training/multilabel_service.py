"""多标签分类推理服务。

接口与 ``ModelService`` 对齐，使识别引擎可以直接替换，无需改动上层逻辑。

与单标签的本质差别在判定方式：

单标签
    Softmax 让所有类别共享同一份概率预算，复合格子里目标被主类别压制，
    因此需要 Top-K 排名、类别对抑制阈值和局部裁剪复核才能把目标捞回来。

多标签
    每类独立 sigmoid，判定退化成 ``probability[target] >= threshold``。
    ``Car 0.92`` 与 ``Crosswalk 0.88`` 可以同时成立，不需要任何补丁。

Ultralytics 的分类头在推理期返回 ``(softmax, logits)`` 元组，原始 logits
可直接取用，因此不需要修改模型结构，只在本服务里改用 sigmoid。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..category_map import class_to_zh, normalize_dataset_class
from ..config import REPORTS_DIR, pick_device, read_training_imgsz, resolve_model_reference
from ..data.cache_store import sharded_path
from ..data.multilabel import MANIFEST_FILENAME, MultiLabelManifest
from ..grid.grid_engine import GridSpec, split_grid
from ..runtime_env import prepare_cache_dir
from .model_service import TilePrediction

DEFAULT_POSITIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class MultiLabelTile:
    """单个格子的多标签结果。"""

    index: int
    probabilities: dict[str, float]

    def top(self, count: int = 5) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda item: -item[1])[:count]


class MultiLabelModelService:
    """加载多标签权重并对网格逐格推理。"""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.model: Any | None = None
        self.weights: Path | None = None
        self.model_hash: str | None = None
        self.device = "cpu"
        self.training_imgsz: int | None = None
        self.names: dict[int, str] = {}
        self.manifest: MultiLabelManifest | None = None
        self.cache_dir = Path(cache_dir) if cache_dir else REPORTS_DIR / "multilabel_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self, weights: str | Path, device: str | None = None) -> dict[str, Any]:
        prepare_cache_dir()
        from ultralytics import YOLO

        reference = resolve_model_reference(weights)
        path = Path(reference)
        if not path.is_file():
            raise FileNotFoundError(f"多标签权重不存在：{path}")
        self.model = YOLO(reference)
        self.weights = path
        self.model_hash = self._file_hash(path)
        self.device = pick_device(device)
        self.training_imgsz = read_training_imgsz(path)
        self.names = {int(k): str(v) for k, v in (self.model.names or {}).items()}
        manifest_path = path.parent / MANIFEST_FILENAME
        self.manifest = (
            MultiLabelManifest.load(manifest_path) if manifest_path.is_file() else None
        )
        return {
            "weights": str(self.weights),
            "device": self.device,
            "classes": self.names,
            "model_hash": self.model_hash,
            "training_imgsz": self.training_imgsz,
        }

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def supports_multiview_target(target_class: str | None) -> bool:
        """多标签模型不需要局部裁剪复核。

        四视角裁剪是为了把被 softmax 压制的目标捞回来；每类独立打分之后
        这个补丁失去存在意义，因此恒定返回 False。
        """
        del target_class
        return False

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _forward(self, tiles: list, imgsz: int):
        """对切好的图块批量前向，返回每类独立概率矩阵。"""
        import torch
        from ultralytics.data.augment import classify_transforms

        transform = classify_transforms(size=imgsz)
        batch = torch.stack([transform(tile) for tile in tiles]).to(self.device)
        torch_model = self.model.model.to(self.device).eval()  # type: ignore[union-attr]
        with torch.no_grad():
            output = torch_model(batch)
        # 推理期分类头返回 (softmax, logits)；多标签只用 logits。
        logits = output[1] if isinstance(output, (list, tuple)) else output
        return torch.sigmoid(logits.float()).cpu()

    def predict_grid(
        self,
        image,
        spec: GridSpec,
        threshold: float = DEFAULT_POSITIVE_THRESHOLD,
        target_class: str | None = None,
        imgsz: int | None = None,
        top_k: int = 3,
        selected_only: bool = True,
        image_key: str | None = None,
    ) -> list[TilePrediction]:
        """逐格推理并转换为与单标签一致的 ``TilePrediction``。"""
        if self.model is None:
            raise RuntimeError("请先加载多标签模型")
        size = int(imgsz or self.training_imgsz or 160)

        cache_path: Path | None = None
        if image_key:
            key = hashlib.sha256(
                f"multilabel-v1|{self.model_hash}|{image_key}|{spec.rows}x{spec.columns}|{size}".encode()
            ).hexdigest()
            cache_path = sharded_path(self.cache_dir, key)
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                tiles = [
                    MultiLabelTile(index=int(item["index"]), probabilities=item["probabilities"])
                    for item in cached
                ]
                predictions = [self._to_prediction(tile, top_k) for tile in tiles]
                return (
                    self.select_target(predictions, threshold, target_class, top_k)
                    if selected_only
                    else predictions
                )

        crops = split_grid(image, spec)
        probabilities = self._forward(crops, size)
        tiles = [
            MultiLabelTile(
                index=index,
                probabilities={
                    str(self.names.get(position, position)): float(value)
                    for position, value in enumerate(row.tolist())
                },
            )
            for index, row in enumerate(probabilities)
        ]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    [{"index": tile.index, "probabilities": tile.probabilities} for tile in tiles],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        predictions = [self._to_prediction(tile, top_k) for tile in tiles]
        return (
            self.select_target(predictions, threshold, target_class, top_k)
            if selected_only
            else predictions
        )

    def _to_prediction(self, tile: MultiLabelTile, top_k: int) -> TilePrediction:
        """把多标签概率整理成统一的格子结果结构。

        ``candidates`` 仍按概率降序排列，使既有的报告与融合代码无需改动；
        区别是这些概率互相独立，可以同时很高。
        """
        ranked = tile.top(max(5, top_k))
        candidates = [
            {
                "rank": rank,
                "label": label,
                "dataset_class": normalize_dataset_class(label),
                "zh": class_to_zh(label),
                "confidence": confidence,
            }
            for rank, (label, confidence) in enumerate(ranked, start=1)
        ]
        top_label, top_confidence = ranked[0] if ranked else ("", 0.0)
        return TilePrediction(
            index=tile.index,
            label=top_label,
            dataset_class=normalize_dataset_class(top_label),
            zh=class_to_zh(top_label),
            confidence=top_confidence,
            candidates=candidates,
        )

    @staticmethod
    def select_target(
        items: list[TilePrediction],
        threshold: float,
        target_class: str | None,
        top_k: int = 3,
        multiview_threshold: float | None = None,
        top1_threshold: float | None = None,
    ) -> list[TilePrediction]:
        """按目标类别的独立概率筛选格子。

        多标签下不存在「被主类别压制」的情况，因此忽略 ``top_k``、
        ``multiview_threshold`` 与 ``top1_threshold``，只做一次阈值比较。
        这三个参数保留仅为与单标签服务保持调用签名一致。
        """
        del top_k, multiview_threshold, top1_threshold
        wanted = normalize_dataset_class(target_class) if target_class else None
        if wanted is None:
            return [item for item in items if item.confidence >= threshold]

        from dataclasses import replace

        selected: list[TilePrediction] = []
        for item in items:
            match = next(
                (
                    candidate
                    for candidate in item.candidates
                    if candidate.get("dataset_class") == wanted
                ),
                None,
            )
            if match is None:
                continue
            probability = float(match.get("confidence", 0.0))
            if probability < float(threshold):
                continue
            selected.append(
                replace(
                    item,
                    target_rank=int(match.get("rank", 0)) or None,
                    target_confidence=probability,
                    target_label=str(match.get("label", wanted)),
                    target_dataset_class=wanted,
                    evidence_view="多标签独立概率",
                )
            )
        return selected

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
