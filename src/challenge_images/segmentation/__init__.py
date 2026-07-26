"""YOLO 分割推理、mask 网格证据和分类融合。"""

from .model_service import MaskInstancePrediction, SegmentationModelService, SegmentationPrediction
from .result_fusion import FusionDecision, FusionResult, fuse_predictions

__all__ = [
    "FusionDecision",
    "FusionResult",
    "MaskInstancePrediction",
    "SegmentationModelService",
    "SegmentationPrediction",
    "fuse_predictions",
]
