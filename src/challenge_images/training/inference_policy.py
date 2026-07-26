"""按「目标类别 × 挑战类型」配置网格推理阈值。

同一个类别在两种题型下的图像性质完全不同，必须使用不同参数：

``dynamic`` / ``imageselect`` / ``tileselect``（3×3，大图 300×300，每格 100×100）
    每个格子是一张**独立照片**，目标通常是照片主体并占据整格。
    此时 Top-1 就是目标，可以使用严格阈值换取精度。

``multicaptcha``（4×4，大图 450×450，每格 112×112）
    整张连续照片被切成 16 块，一个目标往往横跨多格。边缘格只包含目标的
    一小部分（车尾、车轮、半条斑马线），整格分类的 Top-1 常常是背景类，
    目标掉到 Top-2/Top-3。此时沿用 3×3 的严格阈值必然漏掉全部边缘格。

因此参数按两层组合：
1. 类别基线：已用真实样本调校过的 Car/Bus/Crosswalk/Hydrant 保持原值，
   其余类别使用通用基线。
2. 挑战类型调整：仅 4×4 连续照片需要放宽候选证据，按类别分组给出。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..category_map import normalize_dataset_class
from ..config import DEFAULT_TRAIN_IMGSZ


@dataclass(frozen=True)
class InferenceProfile:
    """单个「类别 × 挑战类型」组合的推理参数。

    ``imgsz`` 仅在权重缺少 ``model_meta.json`` 时作为兜底值使用。
    正常情况下推理分辨率由 ``config.read_training_imgsz`` 从权重元数据读取，
    以保证与训练分辨率一致。
    """

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
    imgsz=DEFAULT_TRAIN_IMGSZ,
)


# ---------- 第一层：类别基线（面向 3×3 独立图片） ----------
# 这四类的数值来自真实样本调校，不要凭直觉改动；调整前先跑回归评测。
CATEGORY_PROFILES = {
    "Crosswalk": InferenceProfile(
        name="人行横道复合场景",
        top1_threshold=0.60,
        candidate_threshold=0.30,
        local_threshold=0.65,
        top_k=3,
        imgsz=DEFAULT_TRAIN_IMGSZ,
        allow_multiview=True,
    ),
    "Hydrant": InferenceProfile(
        name="消防栓小目标",
        top1_threshold=0.50,
        candidate_threshold=0.20,
        local_threshold=0.80,
        top_k=3,
        imgsz=DEFAULT_TRAIN_IMGSZ,
    ),
    "Car": InferenceProfile(
        name="车辆严格 Top-1",
        top1_threshold=0.85,
        candidate_threshold=1.0,
        local_threshold=1.0,
        top_k=1,
        imgsz=DEFAULT_TRAIN_IMGSZ,
    ),
    "Bus": InferenceProfile(
        name="公共汽车严格 Top-1",
        top1_threshold=0.80,
        candidate_threshold=1.0,
        local_threshold=1.0,
        top_k=1,
        imgsz=DEFAULT_TRAIN_IMGSZ,
    ),
}


# ---------- 第二层：挑战类型 ----------
# 3×3 家族共用类别基线；只有 4×4 连续照片需要单独放宽。
GRID3_CHALLENGES = frozenset({"dynamic", "imageselect", "tileselect"})
GRID4_CHALLENGES = frozenset({"multicaptcha"})


@dataclass(frozen=True)
class ContinuousAdjustment:
    """4×4 连续照片相对类别基线的调整。

    ``candidate_ceiling`` 是候选阈值的上限而非绝对值：Car/Bus 基线为 1.0
    （只认 Top-1），单靠增量无法降到可召回的水平，必须给出上限；而同组的
    Motorcycle/Tractor 基线本就是 0.25，若直接改写成绝对值反而会收紧。
    取 ``min(基线, 上限)`` 保证这一层只会放宽、不会收紧。
    """

    name: str
    top1_delta: float = 0.0
    candidate_ceiling: float | None = None
    candidate_delta: float = 0.0
    top_k_floor: int = 3


# 类别分组：决定 4×4 下如何放宽。
CONTINUOUS_GROUPS = {
    # 大型载具：3×3 整车在格内，4×4 横跨多格，边缘格只剩车尾/车轮。
    "Car": "large_vehicle",
    "Bus": "large_vehicle",
    "Motorcycle": "large_vehicle",
    "Tractor": "large_vehicle",
    "Boat": "large_vehicle",
    # 中型目标：4×4 下通常只跨 2～4 格，按通用幅度放宽即可。
    "Bicycle": "medium_object",
    # 小目标：本身在整图中占比就低，切成 16 格后更小。
    "Hydrant": "small_object",
    "Traffic Light": "small_object",
    "Chimney": "small_object",
    "Parking meter": "small_object",
    # 跨格纹理/结构：天然横贯画面，边缘格证据弱但仍属目标。
    "Crosswalk": "spanning",
    "Stair": "spanning",
    "Bridge": "spanning",
    # 背景场景类：占据大片画面，几乎每格都有部分证据，需防止全选。
    "Mountain": "scene",
    "Palm": "scene",
}

CONTINUOUS_ADJUSTMENTS = {
    "large_vehicle": ContinuousAdjustment(
        name="4×4 大型载具跨格",
        # 主体格仍要求高置信，但边缘格必须允许从 Top-2/Top-3 召回，
        # 否则 Car/Bus 的 candidate=1.0 会漏掉全部边缘格。
        candidate_ceiling=0.35,
        top_k_floor=3,
    ),
    "medium_object": ContinuousAdjustment(
        name="4×4 中型目标",
        candidate_delta=-0.05,
        top_k_floor=3,
    ),
    "small_object": ContinuousAdjustment(
        name="4×4 小目标",
        top1_delta=-0.10,
        candidate_delta=-0.05,
        top_k_floor=3,
    ),
    "spanning": ContinuousAdjustment(
        name="4×4 跨格纹理",
        top1_delta=-0.05,
        candidate_delta=-0.05,
        top_k_floor=3,
    ),
    "scene": ContinuousAdjustment(
        name="4×4 背景场景",
        # 场景类几乎每格都有弱证据，放宽会导致整屏全选，因此只提 top_k。
        top_k_floor=3,
    ),
}

DEFAULT_CONTINUOUS_ADJUSTMENT = ContinuousAdjustment(
    name="4×4 通用",
    candidate_delta=-0.05,
    top_k_floor=3,
)


def challenge_family(challenge_type: str | None) -> str:
    """把挑战类型归入 ``grid3`` / ``grid4``；未知类型按 3×3 处理。"""
    normalized = str(challenge_type or "").strip().lower()
    if normalized in GRID4_CHALLENGES:
        return "grid4"
    return "grid3"


def _bounded(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def profile_for(
    target_class: str | None,
    challenge_type: str | None = None,
) -> InferenceProfile:
    """取得「类别 × 挑战类型」对应配置。

    ``challenge_type`` 省略或属于 3×3 家族时返回类别基线；
    ``multicaptcha`` 会按类别分组放宽候选证据，以召回跨格的边缘格。
    """
    wanted = normalize_dataset_class(target_class) if target_class else None
    base = CATEGORY_PROFILES.get(wanted, DEFAULT_PROFILE) if wanted else DEFAULT_PROFILE
    if challenge_family(challenge_type) != "grid4":
        return base

    group = CONTINUOUS_GROUPS.get(str(wanted or ""), "")
    adjustment = CONTINUOUS_ADJUSTMENTS.get(group, DEFAULT_CONTINUOUS_ADJUSTMENT)
    candidate = _bounded(base.candidate_threshold + adjustment.candidate_delta, 0.05)
    if adjustment.candidate_ceiling is not None:
        candidate = min(candidate, float(adjustment.candidate_ceiling))
    return replace(
        base,
        name=f"{base.name}｜{adjustment.name}",
        top1_threshold=_bounded(base.top1_threshold + adjustment.top1_delta, 0.30),
        candidate_threshold=candidate,
        top_k=max(base.top_k, adjustment.top_k_floor),
    )
