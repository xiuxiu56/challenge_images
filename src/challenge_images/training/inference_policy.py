"""按目标类别配置网格推理阈值。"""

from __future__ import annotations

from dataclasses import dataclass

from ..category_map import normalize_dataset_class


@dataclass(frozen=True)
class InferenceProfile:
    """单个目标类别的推理参数。"""

    name: str
    top1_threshold: float
    candidate_threshold: float
    local_threshold: float
    top_k: int
    imgsz: int
    allow_multiview: bool = False


DEFAULT_PROFILE = InferenceProfile(
    name="通用类别",
    top1_threshold=0.80,
    candidate_threshold=0.25,
    local_threshold=0.80,
    top_k=3,
    imgsz=320,
)


CATEGORY_PROFILES = {
    "Crosswalk": InferenceProfile(
        name="人行横道复合场景",
        top1_threshold=0.60,
        candidate_threshold=0.30,
        local_threshold=0.65,
        top_k=3,
        # 现有 m1 模型在 224 下对 Bus+Crosswalk 复合格子更稳定；m2 仍按 320 训练。
        imgsz=224,
        allow_multiview=True,
    ),
    "Hydrant": InferenceProfile(
        name="消防栓小目标",
        top1_threshold=0.50,
        candidate_threshold=0.20,
        local_threshold=0.80,
        top_k=3,
        imgsz=320,
    ),
    "Car": InferenceProfile(
        name="车辆严格 Top-1",
        top1_threshold=0.85,
        candidate_threshold=1.0,
        local_threshold=1.0,
        top_k=1,
        imgsz=320,
    ),
    "Bus": InferenceProfile(
        name="公共汽车严格 Top-1",
        top1_threshold=0.80,
        candidate_threshold=1.0,
        local_threshold=1.0,
        top_k=1,
        imgsz=320,
    ),
}


def profile_for(target_class: str | None) -> InferenceProfile:
    """取得目标类别对应配置；未知类别使用通用配置。"""
    wanted = normalize_dataset_class(target_class) if target_class else None
    return CATEGORY_PROFILES.get(wanted, DEFAULT_PROFILE)
