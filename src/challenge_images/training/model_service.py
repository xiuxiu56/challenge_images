"""YOLO26 分类模型加载与网格批量识别。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import hashlib
import json
from typing import Any

from ..category_map import class_to_zh, normalize_dataset_class
from ..config import REPORTS_DIR, pick_device, read_training_imgsz, resolve_model_reference
from ..grid.grid_engine import GridSpec, split_grid
from ..runtime_env import prepare_cache_dir
from ..thresholds import THRESHOLDS


# 以下均来自 challenge_images.thresholds，可通过 config/thresholds.yaml 覆盖。
#
# 局部裁剪适合寻找位于格子下方的人行横道；对 Bus 等依赖完整轮廓的类别，
# 裁剪容易把烟囱、桥和车辆局部误判为目标，因此默认只为 Crosswalk 启用。
MULTIVIEW_TARGET_CLASSES = set(THRESHOLDS.classification.multiview_targets)
# 只有完整格子被道路相关大目标压制时才允许局部复核。自行车、摩托车等
# Top-1 场景曾产生明显误选，因此不使用局部裁剪覆盖它们。
MULTIVIEW_SUPPRESSOR_CLASSES = set(THRESHOLDS.classification.multiview_suppressors)
# Car 图块经常把道路纹理压成第二候选 Crosswalk。该组合使用更严格的完整
# 格子候选阈值，较弱证据交给局部复核，避免纯 Car 被直接命中。
CANDIDATE_SUPPRESSOR_THRESHOLDS = {
    tuple(key.split("|", 1)): value
    for key, value in THRESHOLDS.classification.candidate_suppressors.items()
}


@dataclass
class TilePrediction:
    index: int
    label: str
    dataset_class: str | None
    zh: str
    confidence: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    target_rank: int | None = None
    target_confidence: float | None = None
    target_label: str | None = None
    target_dataset_class: str | None = None
    evidence_view: str | None = None


class ModelService:
    """模型只加载一次，后续样本重复使用同一个实例。"""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.model: Any | None = None
        self.weights: Path | None = None
        self.model_hash: str | None = None
        self.device = "cpu"
        # 该权重训练时使用的 imgsz；推理侧据此对齐，None 表示缺少元数据。
        self.training_imgsz: int | None = None
        self.cache_dir = Path(cache_dir) if cache_dir else REPORTS_DIR / "prediction_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, weights: str | Path, device: str | None = None) -> dict[str, Any]:
        prepare_cache_dir()
        from ultralytics import YOLO

        reference = resolve_model_reference(weights)
        path = Path(reference)
        is_bare_model_name = not path.is_absolute() and path.parent == Path(".") and path.suffix == ".pt"
        if not path.is_file() and not is_bare_model_name:
            raise FileNotFoundError(f"模型权重不存在：{path}")
        self.model = YOLO(reference)
        loaded_path = Path(str(getattr(self.model, "ckpt_path", reference)))
        self.weights = loaded_path if loaded_path.is_file() else path
        self.model_hash = self._file_hash(loaded_path) if loaded_path.is_file() else hashlib.sha256(reference.encode()).hexdigest()
        self.device = pick_device(device)
        self.training_imgsz = read_training_imgsz(self.weights) if self.weights else None
        names = {int(k): str(v) for k, v in (self.model.names or {}).items()}
        return {
            "weights": str(self.weights),
            "device": self.device,
            "classes": names,
            "model_hash": self.model_hash,
            "training_imgsz": self.training_imgsz,
        }

    def predict_grid(
        self,
        image,
        spec: GridSpec,
        threshold: float = 0.5,
        target_class: str | None = None,
        imgsz: int = 224,
        top_k: int = 3,
        selected_only: bool = True,
        image_key: str | None = None,
    ) -> list[TilePrediction]:
        if self.model is None:
            raise RuntimeError("请先加载模型")
        cache_key = None
        if image_key:
            cache_key = hashlib.sha256(f"topk-v2|{self.model_hash}|{image_key}|{spec.rows}x{spec.columns}|{imgsz}".encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.is_file():
                items = [TilePrediction(**item) for item in json.loads(cache_path.read_text(encoding="utf-8"))]
                return self.select_target(items, threshold, target_class, top_k) if selected_only else items
        tiles = split_grid(image, spec)
        results = self.model.predict(
            source=tiles,
            imgsz=imgsz,
            batch=min(16, len(tiles)),
            device=self.device,
            verbose=False,
            stream=False,
        )
        predictions: list[TilePrediction] = []
        for index, result in enumerate(results):
            probs = getattr(result, "probs", None)
            if probs is None:
                continue
            top_index = int(probs.top1)
            confidence = float(probs.top1conf)
            label = str((result.names or {}).get(top_index, top_index))
            dataset_class = normalize_dataset_class(label)
            candidates: list[dict[str, Any]] = []
            for rank, (candidate_index, candidate_confidence) in enumerate(
                zip(probs.top5, probs.top5conf),
                start=1,
            ):
                candidate_label = str((result.names or {}).get(int(candidate_index), candidate_index))
                candidates.append(
                    {
                        "rank": rank,
                        "label": candidate_label,
                        "dataset_class": normalize_dataset_class(candidate_label),
                        "zh": class_to_zh(candidate_label),
                        "confidence": float(candidate_confidence),
                    }
                )
            predictions.append(
                TilePrediction(
                    index=index,
                    label=label,
                    dataset_class=dataset_class,
                    zh=class_to_zh(label),
                    confidence=confidence,
                    candidates=candidates,
                )
            )
        if cache_key:
            (self.cache_dir / f"{cache_key}.json").write_text(json.dumps([item.__dict__ for item in predictions], ensure_ascii=False), encoding="utf-8")
        return self.select_target(predictions, threshold, target_class, top_k) if selected_only else predictions

    def predict_grid_multiview(
        self,
        image,
        spec: GridSpec,
        target_class: str,
        imgsz: int = 224,
        image_key: str | None = None,
    ) -> list[TilePrediction]:
        """用完整格子和局部裁剪共同寻找目标类别证据。

        分类模型每次只输出一个标签。对于“车 + 人行横道”这类复合格子，完整
        格子的 Softmax 概率容易被车压制，因此额外检查格子下部和中央区域。最终
        保留完整格子的 Top-5，同时记录目标类别证据最强的视角。
        """
        if self.model is None:
            raise RuntimeError("请先加载模型")
        wanted = normalize_dataset_class(target_class)
        if wanted is None:
            return self.predict_grid(
                image, spec, imgsz=imgsz, selected_only=False, image_key=image_key
            )

        cache_path: Path | None = None
        if image_key:
            cache_key = hashlib.sha256(
                f"multiview-v1|{self.model_hash}|{image_key}|{spec.rows}x{spec.columns}|{imgsz}|{wanted}".encode()
            ).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.is_file():
                return [
                    TilePrediction(**item)
                    for item in json.loads(cache_path.read_text(encoding="utf-8"))
                ]

        tiles = split_grid(image, spec)
        view_names = ("完整格子", "下部 80%", "下部 65%", "中央 80%")
        sources = []
        for tile in tiles:
            width, height = tile.size
            sources.extend(
                [
                    tile,
                    tile.crop((0, round(height * 0.20), width, height)),
                    tile.crop((0, round(height * 0.35), width, height)),
                    tile.crop(
                        (
                            round(width * 0.10),
                            round(height * 0.10),
                            round(width * 0.90),
                            round(height * 0.90),
                        )
                    ),
                ]
            )

        results = self.model.predict(
            source=sources,
            imgsz=imgsz,
            batch=min(32, len(sources)),
            device=self.device,
            verbose=False,
            stream=False,
        )
        predictions: list[TilePrediction] = []
        view_count = len(view_names)
        for tile_index in range(len(tiles)):
            tile_results = results[tile_index * view_count : (tile_index + 1) * view_count]
            if not tile_results:
                continue
            base = self._prediction_from_result(tile_index, tile_results[0])
            best_evidence: tuple[float, int, str, str] | None = None
            for view_name, result in zip(view_names, tile_results):
                probs = getattr(result, "probs", None)
                if probs is None:
                    continue
                names = result.names or {}
                values = [float(value) for value in probs.data]
                target_index = next(
                    (
                        index
                        for index, label in names.items()
                        if normalize_dataset_class(str(label)) == wanted
                    ),
                    None,
                )
                if target_index is None or int(target_index) >= len(values):
                    continue
                probability = values[int(target_index)]
                rank = 1 + sum(value > probability for value in values)
                label = str(names.get(int(target_index), wanted))
                if best_evidence is None or probability > best_evidence[0]:
                    best_evidence = (probability, rank, label, view_name)
            if best_evidence is not None:
                probability, rank, label, view_name = best_evidence
                base = replace(
                    base,
                    target_rank=rank,
                    target_confidence=probability,
                    target_label=label,
                    target_dataset_class=wanted,
                    evidence_view=view_name,
                )
            predictions.append(base)

        if cache_path is not None:
            cache_path.write_text(
                json.dumps([item.__dict__ for item in predictions], ensure_ascii=False),
                encoding="utf-8",
            )
        return predictions

    @staticmethod
    def supports_multiview_target(target_class: str | None) -> bool:
        """返回目标类别是否适合使用局部裁剪复核。"""
        wanted = normalize_dataset_class(target_class) if target_class else None
        return wanted in MULTIVIEW_TARGET_CLASSES

    @staticmethod
    def _prediction_from_result(index: int, result: Any) -> TilePrediction:
        """把一项 Ultralytics 分类结果转换为统一的格子结果。"""
        probs = getattr(result, "probs", None)
        if probs is None:
            raise ValueError(f"格子 {index} 没有分类结果")
        top_index = int(probs.top1)
        confidence = float(probs.top1conf)
        names = result.names or {}
        label = str(names.get(top_index, top_index))
        candidates: list[dict[str, Any]] = []
        for rank, (candidate_index, candidate_confidence) in enumerate(
            zip(probs.top5, probs.top5conf), start=1
        ):
            candidate_label = str(names.get(int(candidate_index), candidate_index))
            candidates.append(
                {
                    "rank": rank,
                    "label": candidate_label,
                    "dataset_class": normalize_dataset_class(candidate_label),
                    "zh": class_to_zh(candidate_label),
                    "confidence": float(candidate_confidence),
                }
            )
        return TilePrediction(
            index=index,
            label=label,
            dataset_class=normalize_dataset_class(label),
            zh=class_to_zh(label),
            confidence=confidence,
            candidates=candidates,
        )

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def select_target(
        items: list[TilePrediction],
        threshold: float,
        target_class: str | None,
        top_k: int = 3,
        multiview_threshold: float | None = None,
        top1_threshold: float | None = None,
    ) -> list[TilePrediction]:
        """按目标类别的 Top-K 排名和独立概率阈值筛选格子。

        ``threshold`` 用于完整格子的 Top-2/Top-3 候选；完整格子 Top-1 可用
        更严格的 ``top1_threshold``。只有适合局部复核的类别才读取裁剪证据。
        """
        wanted = normalize_dataset_class(target_class) if target_class else None
        if wanted is None:
            return [item for item in items if item.confidence >= threshold]
        selected: list[TilePrediction] = []
        limit = max(1, min(int(top_k), 5))
        direct_threshold = threshold if top1_threshold is None else float(top1_threshold)
        for item in items:
            # 第一层先判断完整格子。Top-1 使用较严格阈值；Top-2/Top-3
            # 使用候选阈值，以保留复合图像中被主类别压制的目标证据。
            base_candidate = next(
                (
                    candidate
                    for candidate in item.candidates[:limit]
                    if candidate.get("dataset_class") == wanted
                ),
                None,
            )
            if base_candidate is not None:
                base_probability = float(base_candidate.get("confidence", 0.0))
                base_rank = int(base_candidate.get("rank", 0))
                required = direct_threshold if base_rank == 1 else threshold
                if base_rank > 1:
                    required = max(
                        required,
                        CANDIDATE_SUPPRESSOR_THRESHOLDS.get(
                            (wanted, str(item.dataset_class)),
                            required,
                        ),
                    )
                if base_probability >= required:
                    selected.append(
                        replace(
                            item,
                            target_rank=base_rank,
                            target_confidence=base_probability,
                            target_label=str(base_candidate.get("label", wanted)),
                            target_dataset_class=wanted,
                            evidence_view="完整格子",
                        )
                    )
                    continue

            # 第二层仅允许特定类别使用局部裁剪救回。Bus 等类别即使局部视角
            # 给出高分，也仍以完整格子为准。
            if (
                wanted in MULTIVIEW_TARGET_CLASSES
                and multiview_threshold is not None
                and item.target_dataset_class == wanted
                and item.target_rank is not None
                and item.evidence_view not in (None, "完整格子")
                and item.target_rank <= limit
                and item.dataset_class in MULTIVIEW_SUPPRESSOR_CLASSES
                and float(item.target_confidence or 0.0) >= float(multiview_threshold)
            ):
                selected.append(item)
                continue

            # 兼容未保存 Top-K 的旧缓存记录。
            if not item.candidates and item.dataset_class == wanted and item.confidence >= direct_threshold:
                selected.append(
                    replace(
                        item,
                        target_rank=1,
                        target_confidence=item.confidence,
                        target_label=item.label,
                        target_dataset_class=wanted,
                        evidence_view="完整格子",
                    )
                )
                continue

            # 没有命中后继续处理下一格。
            continue
        return selected

    @property
    def loaded(self) -> bool:
        return self.model is not None
