"""统一识别路由和参数预设。

策略只决定“使用哪条识别链路”和“使用什么参数”，不直接加载模型，
因此 GUI、离线验证和在线识别能够共享同一套规则。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..category_map import normalize_dataset_class
from ..config import DEFAULT_SEGMENTATION_IMGSZ, DEFAULT_TRAIN_IMGSZ
from ..grid.grid_engine import GridSpec
from ..training.inference_policy import profile_for


ENGINE_MODE_LABELS = {
    "smart": "智能推荐",
    "classifier": "逐格分类",
    "fusion": "分类 + 整图分割",
}

PARAMETER_PRESET_LABELS = {
    "balanced": "平衡（推荐）",
    "precision": "精度优先",
    "recall": "召回优先",
    "custom": "自定义",
}


@dataclass(frozen=True)
class RecognitionParameters:
    """分类、分割和融合共用的一组推理参数。"""

    classification_imgsz: int = DEFAULT_TRAIN_IMGSZ
    classification_top1: float = 0.80
    classification_candidate: float = 0.25
    classification_local: float = 0.80
    classification_top_k: int = 3
    allow_multiview: bool = True
    fusion_classification_imgsz: int = DEFAULT_TRAIN_IMGSZ
    segmentation_imgsz: int = DEFAULT_SEGMENTATION_IMGSZ
    segmentation_confidence: float = 0.25
    segmentation_min_cell_ratio: float = 0.002
    segmentation_min_mask_ratio: float = 0.10
    instance_classification_threshold: float = 0.80
    instance_confidence_threshold: float = 0.60
    fusion_mode: str = "balanced"


@dataclass(frozen=True)
class RecognitionRoute:
    """一次识别最终采用的实际链路。"""

    requested_mode: str
    actual_mode: str
    use_multiview: bool
    reason: str

    @property
    def label(self) -> str:
        return ENGINE_MODE_LABELS.get(self.actual_mode, self.actual_mode)


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def parameters_for(
    preset: str,
    target_class: str | None,
    model_imgsz: int | None = None,
    challenge_type: str | None = None,
) -> RecognitionParameters:
    """按「类别 × 挑战类型」基础配置生成平衡、精度或召回预设。

    ``model_imgsz`` 为已加载权重的训练分辨率。推理分辨率必须与训练一致，
    因此它优先于类别配置里的 ``imgsz``；缺少模型元数据时才回退类别默认值。

    ``challenge_type`` 决定 3×3 独立图片还是 4×4 连续照片：后者目标横跨
    多格，边缘格需要从 Top-2/Top-3 召回，不能沿用 3×3 的严格阈值。
    """
    profile = profile_for(target_class, challenge_type)
    classification_imgsz = int(model_imgsz) if model_imgsz else profile.imgsz
    base = RecognitionParameters(
        classification_imgsz=classification_imgsz,
        classification_top1=profile.top1_threshold,
        classification_candidate=profile.candidate_threshold,
        classification_local=profile.local_threshold,
        classification_top_k=profile.top_k,
        allow_multiview=profile.allow_multiview,
        # 融合链路同样按整格送入分类模型，分辨率不再单独降级。
        fusion_classification_imgsz=classification_imgsz,
    )
    if preset in {"balanced", "custom"}:
        return base
    if preset == "precision":
        return replace(
            base,
            classification_top1=_bounded(base.classification_top1 + 0.10, 0.0, 0.98),
            classification_candidate=_bounded(
                base.classification_candidate + 0.10, 0.0, 1.0
            ),
            classification_local=_bounded(base.classification_local + 0.10, 0.0, 0.98),
            segmentation_confidence=0.35,
            segmentation_min_mask_ratio=0.15,
            instance_classification_threshold=0.85,
            instance_confidence_threshold=0.70,
            fusion_mode="consensus",
        )
    if preset == "recall":
        return replace(
            base,
            classification_top1=_bounded(base.classification_top1 - 0.15, 0.40, 1.0),
            classification_candidate=_bounded(
                base.classification_candidate - 0.10, 0.10, 1.0
            ),
            classification_local=_bounded(base.classification_local - 0.15, 0.45, 1.0),
            classification_top_k=max(3, base.classification_top_k),
            segmentation_confidence=0.15,
            segmentation_min_mask_ratio=0.05,
            instance_classification_threshold=0.60,
            instance_confidence_threshold=0.45,
            fusion_mode="balanced",
        )
    raise ValueError(f"未知参数方案：{preset}")


def resolve_recognition_route(
    requested_mode: str,
    *,
    challenge_type: str,
    spec: GridSpec,
    target_class: str,
    segmentation_loaded: bool,
    segmentation_supported: bool,
    allow_multiview: bool = True,
) -> RecognitionRoute:
    """根据题型、网格和模型覆盖情况选择实际识别链路。

    3×3 独立图和动态题优先逐格分类；4×4 连续照片在分割模型覆盖
    目标类别时使用整图融合。人行横道继续使用经过约束的局部多视角，
    避免通用分割模型缺少该类别时扩大误点。
    """
    if requested_mode not in ENGINE_MODE_LABELS:
        raise ValueError(f"未知识别引擎：{requested_mode}")
    challenge = str(challenge_type or "").strip().lower()
    wanted = normalize_dataset_class(target_class)
    multiview = bool(allow_multiview and wanted == "Crosswalk")

    if requested_mode == "classifier":
        return RecognitionRoute(
            requested_mode=requested_mode,
            actual_mode="classifier",
            use_multiview=multiview,
            reason="用户选择逐格分类",
        )

    if requested_mode == "fusion":
        if not segmentation_loaded:
            return RecognitionRoute(
                requested_mode=requested_mode,
                actual_mode="classifier",
                use_multiview=multiview,
                reason="分割模型未加载，已回退逐格分类",
            )
        if not segmentation_supported:
            return RecognitionRoute(
                requested_mode=requested_mode,
                actual_mode="classifier",
                use_multiview=multiview,
                reason="分割模型未覆盖目标类别，已回退逐格分类",
            )
        return RecognitionRoute(
            requested_mode=requested_mode,
            actual_mode="fusion",
            use_multiview=multiview,
            reason="用户选择分类与整图分割融合",
        )

    if spec.count == 9 or challenge in {"dynamic", "imageselect", "tileselect"}:
        return RecognitionRoute(
            requested_mode=requested_mode,
            actual_mode="classifier",
            use_multiview=multiview,
            reason="3×3 独立图或动态题采用逐格分类",
        )
    if wanted == "Crosswalk":
        return RecognitionRoute(
            requested_mode=requested_mode,
            actual_mode="classifier",
            use_multiview=multiview,
            reason="人行横道采用受控逐格多视角",
        )
    if segmentation_loaded and segmentation_supported:
        return RecognitionRoute(
            requested_mode=requested_mode,
            actual_mode="fusion",
            use_multiview=False,
            reason="4×4 连续照片且分割模型覆盖目标类别",
        )
    fallback = "分割模型未加载" if not segmentation_loaded else "分割模型未覆盖目标类别"
    return RecognitionRoute(
        requested_mode=requested_mode,
        actual_mode="classifier",
        use_multiview=multiview,
        reason=f"4×4 连续照片的{fallback}，已回退逐格分类",
    )
