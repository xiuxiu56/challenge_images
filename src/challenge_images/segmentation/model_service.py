"""YOLO 分割模型加载、整图推理和目标 mask 网格转换。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
from PIL import Image

from ..config import pick_device, resolve_segmentation_model_reference
from ..grid.grid_engine import GridSpec
from ..runtime_env import prepare_cache_dir
from ..thresholds import THRESHOLDS
from .category_map import segmentation_category_key
from .mask_grid import (
    CellMaskEvidence,
    mask_bottom_extensions,
    mask_grid_evidence,
    normalize_mask,
    render_mask_overlay,
)


# 来自 challenge_images.thresholds，可通过 config/thresholds.yaml 覆盖。
SEGMENTATION_RECOVERY_CONFIDENCE = THRESHOLDS.instance_validation.recovery_confidence


@dataclass
class MaskInstancePrediction:
    """一个目标实例的类别、置信度和格子覆盖结果。"""

    class_id: int
    label: str
    category_key: str | None
    confidence: float
    selected_indices: list[int]
    cells: list[CellMaskEvidence] = field(default_factory=list)
    bottom_extension_indices: list[int] = field(default_factory=list)


@dataclass
class SegmentationPrediction:
    """一次完整图片分割推理的目标类别结果。"""

    target_class: str
    target_key: str | None
    supported: bool
    selected_indices: list[int]
    instances: list[MaskInstancePrediction]
    cell_scores: dict[int, float]
    preview: Image.Image
    message: str = ""


class SegmentationModelService:
    """独立加载 YOLO segmentation 权重，并把 mask 转换为网格索引。"""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.weights: Path | None = None
        self.model_hash: str | None = None
        self.device = "cpu"
        self.class_names: dict[int, str] = {}

    def load(self, weights: str | Path, device: str | None = None) -> dict[str, Any]:
        prepare_cache_dir()
        from ultralytics import YOLO

        reference = resolve_segmentation_model_reference(weights)
        path = Path(reference)
        bare_name = not path.is_absolute() and path.parent == Path(".") and path.suffix == ".pt"
        if not path.is_file() and not bare_name:
            raise FileNotFoundError(f"分割模型权重不存在：{path}")
        model = YOLO(reference)
        task = str(getattr(model, "task", "") or "")
        if task and task != "segment":
            raise ValueError(f"所选权重任务为 {task}，请加载 YOLO segmentation 权重")
        self.model = model
        loaded_path = Path(str(getattr(model, "ckpt_path", reference)))
        self.weights = loaded_path if loaded_path.is_file() else path
        self.model_hash = (
            self._file_hash(loaded_path)
            if loaded_path.is_file()
            else hashlib.sha256(reference.encode()).hexdigest()
        )
        self.device = pick_device(device)
        names = getattr(model, "names", {}) or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.class_names = {int(key): str(value) for key, value in names.items()}
        return {
            "weights": str(self.weights),
            "device": self.device,
            "classes": self.class_names,
            "model_hash": self.model_hash,
        }

    def supports_target(self, target_class: str) -> bool:
        """检查当前分割权重是否包含目标类别。"""
        target_key = segmentation_category_key(target_class)
        return target_key is not None and any(
            segmentation_category_key(label) == target_key
            for label in self.class_names.values()
        )

    def predict(
        self,
        image: Image.Image,
        spec: GridSpec,
        target_class: str,
        *,
        imgsz: int = 640,
        confidence: float = 0.25,
        min_overlap_pixels: int = 20,
        min_cell_ratio: float = 0.002,
        min_mask_ratio: float = 0.10,
    ) -> SegmentationPrediction:
        """对完整挑战图片执行目标实例分割并返回命中格子。"""
        if self.model is None:
            raise RuntimeError("请先加载分割模型")
        target_key = segmentation_category_key(target_class)
        if target_key is None or not self.supports_target(target_class):
            preview = render_mask_overlay(image, [], spec, [])
            return SegmentationPrediction(
                target_class=target_class,
                target_key=target_key,
                supported=False,
                selected_indices=[],
                instances=[],
                cell_scores={},
                preview=preview,
                message=f"当前分割模型不包含目标类别：{target_class}，本次由分类模型兜底",
            )

        requested_confidence = float(confidence)
        instances, target_masks, cell_scores, selected_indices = (
            self._predict_target_instances(
                image,
                spec,
                target_key,
                imgsz=int(imgsz),
                confidence=requested_confidence,
                min_overlap_pixels=min_overlap_pixels,
                min_cell_ratio=min_cell_ratio,
                min_mask_ratio=min_mask_ratio,
            )
        )
        recovery_used = False
        if (
            not instances
            and requested_confidence > SEGMENTATION_RECOVERY_CONFIDENCE
        ):
            instances, target_masks, cell_scores, selected_indices = (
                self._predict_target_instances(
                    image,
                    spec,
                    target_key,
                    imgsz=int(imgsz),
                    confidence=SEGMENTATION_RECOVERY_CONFIDENCE,
                    min_overlap_pixels=min_overlap_pixels,
                    min_cell_ratio=min_cell_ratio,
                    min_mask_ratio=min_mask_ratio,
                )
            )
            recovery_used = bool(instances)

        preview = render_mask_overlay(image, target_masks, spec, selected_indices)
        if recovery_used:
            message = (
                f"常规阈值 {requested_confidence:.3f} 未发现目标；"
                f"已用 {SEGMENTATION_RECOVERY_CONFIDENCE:.3f} 找回 "
                f"{len(instances)} 个候选 mask，交由分类复核"
            )
        elif instances:
            message = f"发现 {len(instances)} 个目标 mask"
        else:
            message = "模型有该类别，但本图未发现目标 mask"
        return SegmentationPrediction(
            target_class=target_class,
            target_key=target_key,
            supported=True,
            selected_indices=sorted(selected_indices),
            instances=instances,
            cell_scores=cell_scores,
            preview=preview,
            message=message,
        )

    def _predict_target_instances(
        self,
        image: Image.Image,
        spec: GridSpec,
        target_key: str,
        *,
        imgsz: int,
        confidence: float,
        min_overlap_pixels: int,
        min_cell_ratio: float,
        min_mask_ratio: float,
    ) -> tuple[
        list[MaskInstancePrediction],
        list[np.ndarray],
        dict[int, float],
        set[int],
    ]:
        """执行一次分割并提取目标类别实例。"""
        results = self.model.predict(
            source=image,
            imgsz=int(imgsz),
            conf=float(confidence),
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0] if results else None
        masks_obj = getattr(result, "masks", None) if result is not None else None
        boxes = getattr(result, "boxes", None) if result is not None else None
        if masks_obj is None or boxes is None:
            return [], [], {}, set()

        raw_masks = masks_obj.data.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        instances: list[MaskInstancePrediction] = []
        target_masks: list[np.ndarray] = []
        cell_scores: dict[int, float] = {}
        selected_indices: set[int] = set()
        for raw_mask, class_id, score in zip(raw_masks, class_ids, confidences):
            label = self.class_names.get(int(class_id), str(class_id))
            category_key = segmentation_category_key(label)
            if category_key != target_key:
                continue
            mask = normalize_mask(raw_mask, image.size)
            evidence = mask_grid_evidence(
                mask,
                spec,
                min_overlap_pixels=min_overlap_pixels,
                min_cell_ratio=min_cell_ratio,
                min_mask_ratio=min_mask_ratio,
            )
            selected = [item.index for item in evidence if item.selected]
            for item in evidence:
                if item.overlap_pixels > 0:
                    cell_scores[item.index] = max(
                        cell_scores.get(item.index, 0.0),
                        item.cell_ratio,
                    )
                if item.selected:
                    selected_indices.add(item.index)
            target_masks.append(mask)
            instances.append(
                MaskInstancePrediction(
                    class_id=int(class_id),
                    label=label,
                    category_key=category_key,
                    confidence=float(score),
                    selected_indices=selected,
                    cells=evidence,
                    bottom_extension_indices=mask_bottom_extensions(mask, spec),
                )
            )
        return instances, target_masks, cell_scores, selected_indices

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def loaded(self) -> bool:
        return self.model is not None
