"""分类与分割共用的统一识别执行引擎。"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from ..grid.grid_engine import GridSpec
from ..segmentation.model_service import SegmentationModelService, SegmentationPrediction
from ..segmentation.result_fusion import FusionResult, format_fusion_report, fuse_predictions
from ..training.model_service import ModelService, TilePrediction
from .continuous_grid import refine_continuous_grid
from .policy import (
    ENGINE_MODE_LABELS,
    RecognitionParameters,
    RecognitionRoute,
    resolve_recognition_route,
)


@dataclass(frozen=True)
class RecognitionResult:
    """所有功能页共用的一次识别结果。"""

    target_class: str
    challenge_type: str
    spec: GridSpec
    route: RecognitionRoute
    parameters: RecognitionParameters
    indices: list[int]
    selected_predictions: list[TilePrediction]
    all_predictions: list[TilePrediction]
    preview: Image.Image
    segmentation: SegmentationPrediction | None = None
    fusion: FusionResult | None = None


class RecognitionEngine:
    """把逐格分类、受控多视角和整图分割收敛为一个入口。"""

    def __init__(
        self,
        classification: ModelService,
        segmentation: SegmentationModelService,
    ) -> None:
        self.classification = classification
        self.segmentation = segmentation

    def recognize(
        self,
        image: Image.Image,
        *,
        challenge_type: str,
        spec: GridSpec,
        target_class: str,
        requested_mode: str,
        parameters: RecognitionParameters,
        image_key: str | None = None,
    ) -> RecognitionResult:
        """执行策略解析、分类推理，并在需要时运行分割融合。"""
        if not self.classification.loaded:
            raise RuntimeError("请先加载分类模型")
        segmentation_supported = (
            self.segmentation.loaded and self.segmentation.supports_target(target_class)
        )
        route = resolve_recognition_route(
            requested_mode,
            challenge_type=challenge_type,
            spec=spec,
            target_class=target_class,
            segmentation_loaded=self.segmentation.loaded,
            segmentation_supported=segmentation_supported,
            allow_multiview=parameters.allow_multiview,
        )
        use_multiview = (
            route.use_multiview
            and self.classification.supports_multiview_target(target_class)
        )
        classification_imgsz = (
            parameters.fusion_classification_imgsz
            if route.actual_mode == "fusion"
            else parameters.classification_imgsz
        )
        if use_multiview:
            all_predictions = self.classification.predict_grid_multiview(
                image,
                spec,
                target_class,
                imgsz=classification_imgsz,
                image_key=image_key,
            )
        else:
            all_predictions = self.classification.predict_grid(
                image,
                spec,
                threshold=0.0,
                target_class=None,
                imgsz=classification_imgsz,
                top_k=parameters.classification_top_k,
                selected_only=False,
                image_key=image_key,
            )
        selected_classification = self.classification.select_target(
            all_predictions,
            threshold=parameters.classification_candidate,
            target_class=target_class,
            top_k=parameters.classification_top_k,
            multiview_threshold=(
                parameters.classification_local if use_multiview else None
            ),
            top1_threshold=parameters.classification_top1,
        )

        segmentation: SegmentationPrediction | None = None
        fusion: FusionResult | None = None
        if route.actual_mode == "fusion":
            segmentation = self.segmentation.predict(
                image,
                spec,
                target_class,
                imgsz=parameters.segmentation_imgsz,
                confidence=parameters.segmentation_confidence,
                min_cell_ratio=parameters.segmentation_min_cell_ratio,
                min_mask_ratio=parameters.segmentation_min_mask_ratio,
            )
            fusion = fuse_predictions(
                all_predictions,
                selected_classification,
                segmentation,
                target_class=target_class,
                grid_count=spec.count,
                mode=parameters.fusion_mode,
                instance_classification_threshold=(
                    parameters.instance_classification_threshold
                ),
                instance_confidence_threshold=parameters.instance_confidence_threshold,
            )
            indices = list(fusion.selected_indices)
            preview = segmentation.preview
        else:
            indices = sorted(item.index for item in selected_classification)
            # 把每格对目标类别的概率一并交给形态学收敛：4×4 下孤立单格
            # 没有连通性背书，需要更强的自身证据才保留。
            scores = {
                item.index: float(item.target_confidence or 0.0)
                for item in selected_classification
            }
            indices = refine_continuous_grid(target_class, spec, indices, scores=scores)
            preview = image

        by_index = {item.index: item for item in all_predictions}
        selected_predictions = [by_index[index] for index in indices if index in by_index]
        return RecognitionResult(
            target_class=target_class,
            challenge_type=challenge_type,
            spec=spec,
            route=route,
            parameters=parameters,
            indices=indices,
            selected_predictions=selected_predictions,
            all_predictions=all_predictions,
            preview=preview,
            segmentation=segmentation,
            fusion=fusion,
        )


def format_recognition_report(result: RecognitionResult) -> str:
    """生成三个功能页共用的中文识别明细。"""
    params = result.parameters
    selected = []
    selected_set = set(result.indices)
    for item in result.all_predictions:
        if item.index not in selected_set:
            continue
        selected.append(
            f"格子{item.index}: Top-1={item.label} {item.confidence:.4f}; "
            f"目标={item.target_label or result.target_class}, "
            f"排名={item.target_rank or '未进入候选'}, "
            f"概率={float(item.target_confidence or 0):.4f}, "
            f"证据视角={item.evidence_view or '完整格子'}"
        )
    all_details = []
    for item in result.all_predictions:
        candidates = ", ".join(
            f"#{candidate['rank']} {candidate['label']} {candidate['confidence']:.4f}"
            for candidate in item.candidates[: params.classification_top_k]
        )
        extra = ""
        if item.evidence_view:
            extra = (
                f" | 多视角目标={item.target_label}, 排名={item.target_rank}, "
                f"概率={float(item.target_confidence or 0):.4f}, 视角={item.evidence_view}"
            )
        all_details.append(f"格子{item.index}: {candidates}{extra}")

    lines = [
        f"目标类别: {result.target_class}",
        f"挑战类型: {result.challenge_type}",
        f"网格: {result.spec.text}",
        f"请求引擎: {ENGINE_MODE_LABELS.get(result.route.requested_mode, result.route.requested_mode)}",
        f"实际方案: {result.route.label}",
        f"选择原因: {result.route.reason}",
        (
            "分类参数: "
            f"imgsz={params.fusion_classification_imgsz if result.route.actual_mode == 'fusion' else params.classification_imgsz}, "
            f"Top-K={params.classification_top_k}, "
            f"Top-1阈值={params.classification_top1:.3f}, "
            f"候选阈值={params.classification_candidate:.3f}, "
            f"局部阈值={params.classification_local:.3f}"
        ),
    ]
    if result.route.actual_mode == "fusion":
        lines.append(
            "分割参数: "
            f"imgsz={params.segmentation_imgsz}, "
            f"置信度={params.segmentation_confidence:.3f}, "
            f"格子覆盖率={params.segmentation_min_cell_ratio:.4f}, "
            f"mask占比={params.segmentation_min_mask_ratio:.3f}"
        )
    lines.extend(
        [
            f"识别到的格子: {result.indices}",
            "",
            "命中详情:",
            "\n".join(selected) if selected else "无",
            "",
            "全部格子 Top-K:",
            "\n".join(all_details),
        ]
    )
    if result.fusion is not None and result.segmentation is not None:
        lines.extend(
            [
                "",
                "融合诊断:",
                format_fusion_report(
                    result.fusion,
                    result.segmentation,
                    target_class=result.target_class,
                    spec=result.spec,
                ).rstrip(),
            ]
        )
    return "\n".join(lines) + "\n"
