"""分类 Top-K 与分割 mask 格子证据融合。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

from ..category_map import normalize_dataset_class
from ..thresholds import THRESHOLDS
from ..training.model_service import TilePrediction
from .model_service import (
    SEGMENTATION_RECOVERY_CONFIDENCE,
    MaskInstancePrediction,
    SegmentationPrediction,
)
from ..grid.grid_engine import GridSpec


FUSION_MODE_LABELS = {
    "balanced": "平衡融合",
    "union": "并集（召回优先）",
    "consensus": "双证据（精度优先）",
}

# 以下阈值全部来自 challenge_images.thresholds，可通过 config/thresholds.yaml 覆盖。
# 保留模块级名称是为了不改动既有调用方与测试。
_MOTORCYCLE = THRESHOLDS.motorcycle
_BUS = THRESHOLDS.bus
_TRAFFIC_LIGHT = THRESHOLDS.traffic_light
_MASK = THRESHOLDS.mask_evidence

STRONG_CLASSIFICATION_RESCUE_THRESHOLD = THRESHOLDS.instance_validation.strong_classification_rescue
MOTORCYCLE_FRINGE_MIN_CELL_RATIO = _MOTORCYCLE.fringe_min_cell_ratio
MOTORCYCLE_FRINGE_MIN_MASK_RATIO = _MOTORCYCLE.fringe_min_mask_ratio
MOTORCYCLE_HORIZONTAL_EXPANSION_CELL_RATIO = _MOTORCYCLE.horizontal_expansion_cell_ratio
MOTORCYCLE_HORIZONTAL_EXPANSION_MASK_RATIO = _MOTORCYCLE.horizontal_expansion_mask_ratio
MOTORCYCLE_BOX_FILL_CELL_RATIO = _MOTORCYCLE.box_fill_cell_ratio
MOTORCYCLE_BOX_FILL_MASK_RATIO = _MOTORCYCLE.box_fill_mask_ratio
BUS_MAJOR_CELL_RATIO = _BUS.major_cell_ratio
BUS_THIN_CELL_RATIO = _BUS.thin_cell_ratio
TRAFFIC_LIGHT_FRINGE_CLASSIFICATION_SCORE = _TRAFFIC_LIGHT.fringe_classification_score
TRAFFIC_LIGHT_FRINGE_MASK_RATIO = _TRAFFIC_LIGHT.fringe_mask_ratio
TRAFFIC_LIGHT_FRINGE_CELL_RATIO = _TRAFFIC_LIGHT.fringe_cell_ratio
DOMINANT_MOTORCYCLE_MIN_AREA_RATIO = _MOTORCYCLE.dominant_min_area_ratio
MIN_OVERLAP_PIXELS = _MASK.min_overlap_pixels
RAW_INDEX_MIN_CELL_RATIO = _MASK.min_cell_ratio


@dataclass(frozen=True)
class FusionDecision:
    index: int
    selected: bool
    classification_hit: bool
    segmentation_hit: bool
    classification_score: float
    classification_rank: int | None
    mask_cell_ratio: float
    validated_segmentation_hit: bool
    reason: str


@dataclass(frozen=True)
class FusionResult:
    mode: str
    selected_indices: list[int]
    classification_indices: list[int]
    segmentation_indices: list[int]
    validated_segmentation_indices: list[int]
    validated_instance_count: int
    rejected_instance_count: int
    instance_classification_threshold: float
    instance_confidence_threshold: float
    decisions: list[FusionDecision]


@dataclass
class _MaskValidationEvidence:
    """一个 mask 实例在融合前的复核证据。"""

    instance: MaskInstancePrediction
    indices: set[int]
    raw_indices: set[int]
    maximum_score: float
    area: int
    accepted: bool


def _target_candidate(item: TilePrediction, target_class: str) -> tuple[float, int | None]:
    wanted = normalize_dataset_class(target_class)
    for candidate in item.candidates:
        if candidate.get("dataset_class") == wanted:
            return float(candidate.get("confidence", 0.0)), int(candidate.get("rank", 0)) or None
    if item.dataset_class == wanted:
        return float(item.confidence), 1
    return 0.0, None


def _instance_indices(
    instance: MaskInstancePrediction,
    target_class: str,
    grid_count: int,
) -> set[int]:
    """返回实例的有效格子；摩托车恢复少量跨格边缘。"""
    indices = {int(index) for index in instance.selected_indices}
    columns = isqrt(grid_count)
    if (
        normalize_dataset_class(target_class) != "Motorcycle"
        or not indices
        or columns * columns != grid_count
    ):
        return indices
    selected_rows = {index // columns for index in indices}
    selected_columns = {index % columns for index in indices}
    minimum_row, maximum_row = min(selected_rows), max(selected_rows)
    minimum_column, maximum_column = min(selected_columns), max(selected_columns)
    linear_instance_extension = len(selected_rows) == 1 or len(selected_columns) == 1
    for cell in instance.cells:
        index = int(cell.index)
        row, column = divmod(index, columns)
        within_instance_box = (
            minimum_row <= row <= maximum_row
            and minimum_column <= column <= maximum_column
        )
        if (
            int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS
            and float(cell.cell_ratio) >= MOTORCYCLE_FRINGE_MIN_CELL_RATIO
            and float(cell.mask_ratio) >= MOTORCYCLE_FRINGE_MIN_MASK_RATIO
            and (within_instance_box or linear_instance_extension)
        ):
            indices.add(index)

    if linear_instance_extension:
        return indices

    # 二维摩托车实例常先漏掉右侧车轮，再导致上方同列格子落在初始
    # 包围框之外。先恢复同一行的明显横向延伸，再用真实 mask 证据
    # 填充扩展后的包围框；不会向 m_04_sv_1116 这类实例底部盲目扩张。
    for cell in instance.cells:
        index = int(cell.index)
        row, column = divmod(index, columns)
        horizontally_adjacent = any(
            existing // columns == row
            and abs(existing % columns - column) == 1
            for existing in indices
        )
        if (
            minimum_row <= row <= maximum_row
            and horizontally_adjacent
            and int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS
            and float(cell.cell_ratio) >= MOTORCYCLE_HORIZONTAL_EXPANSION_CELL_RATIO
            and float(cell.mask_ratio) >= MOTORCYCLE_HORIZONTAL_EXPANSION_MASK_RATIO
        ):
            indices.add(index)

    rows = {index // columns for index in indices}
    cell_columns = {index % columns for index in indices}
    minimum_row, maximum_row = min(rows), max(rows)
    minimum_column, maximum_column = min(cell_columns), max(cell_columns)
    for cell in instance.cells:
        index = int(cell.index)
        row, column = divmod(index, columns)
        if (
            minimum_row <= row <= maximum_row
            and minimum_column <= column <= maximum_column
            and int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS
            and float(cell.cell_ratio) >= MOTORCYCLE_BOX_FILL_CELL_RATIO
            and float(cell.mask_ratio) >= MOTORCYCLE_BOX_FILL_MASK_RATIO
        ):
            indices.add(index)
    return indices


def _bus_instance_indices(
    record: _MaskValidationEvidence,
    grid_count: int,
) -> set[int]:
    """按大型公共汽车 mask 的格子占用和边界连通性恢复跨格部分。"""
    columns = isqrt(grid_count)
    if columns * columns != grid_count:
        return set(record.indices)
    by_cell = {int(cell.index): cell for cell in record.instance.cells}
    original = {int(index) for index in record.indices}
    if not original:
        return set()
    row_counts: dict[int, int] = {}
    for index in original:
        row = index // columns
        row_counts[row] = row_counts.get(row, 0) + 1
    primary_row = min(
        row for row, count in row_counts.items() if count == max(row_counts.values())
    )
    major = {
        int(cell.index)
        for cell in record.instance.cells
        if int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS
        and float(cell.cell_ratio) >= BUS_MAJOR_CELL_RATIO
        and int(cell.index) // columns >= primary_row
    }
    indices = original | major
    if not major:
        return indices
    row_widths: dict[int, int] = {}
    for index in indices:
        row = index // columns
        row_widths[row] = row_widths.get(row, 0) + 1
    thin_extension_rows = {
        row for row, width in row_widths.items() if width == 1
    }

    # 小面积边缘仅在与已接纳格子共享水平 mask 边界时加入。
    # 只扩展当前仅有一个主体格的窄行；已经连续覆盖多格的车身行不再
    # 吸收尾部碎片。这样能恢复 483 的格子8，同时排除 486 的格子11。
    changed = True
    while changed:
        changed = False
        for index, cell in by_cell.items():
            if (
                index in indices
                or index // columns < primary_row
                or index // columns not in thin_extension_rows
                or int(cell.overlap_pixels) < MIN_OVERLAP_PIXELS
                or float(cell.cell_ratio) < BUS_THIN_CELL_RATIO
            ):
                continue
            column = index % columns
            connected = False
            if column > 0 and index - 1 in indices:
                left = by_cell.get(index - 1)
                connected = bool(
                    left and left.touches_right and cell.touches_left
                )
            if not connected and column + 1 < columns and index + 1 in indices:
                right = by_cell.get(index + 1)
                connected = bool(
                    right and right.touches_left and cell.touches_right
                )
            if connected:
                indices.add(index)
                changed = True

    # mask 底部若停在格线前几像素，仅沿实际底边所在列补下一格。
    for index in record.instance.bottom_extension_indices:
        above = int(index) - columns
        if 0 <= int(index) < grid_count and above in indices:
            indices.add(int(index))
    return indices


def _expand_accepted_instance(
    record: _MaskValidationEvidence,
    by_index: dict[int, TilePrediction],
    target_class: str,
    grid_count: int,
) -> None:
    """实例通过复核后，按类别恢复可靠的跨格边缘。"""
    wanted = normalize_dataset_class(target_class)
    if wanted == "Bus":
        record.indices = _bus_instance_indices(record, grid_count)
        return
    if wanted != "Traffic Light":
        return
    columns = isqrt(grid_count)
    if columns * columns != grid_count:
        return
    for cell in record.instance.cells:
        index = int(cell.index)
        score = (
            _target_candidate(by_index[index], target_class)[0]
            if index in by_index
            else 0.0
        )
        vertically_connected = any(
            abs(index - existing) == columns
            and index % columns == existing % columns
            for existing in record.indices
        )
        if (
            vertically_connected
            and int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS
            and float(cell.cell_ratio) >= TRAFFIC_LIGHT_FRINGE_CELL_RATIO
            and float(cell.mask_ratio) >= TRAFFIC_LIGHT_FRINGE_MASK_RATIO
            and score >= TRAFFIC_LIGHT_FRINGE_CLASSIFICATION_SCORE
        ):
            record.indices.add(index)


def _instance_raw_indices(instance: MaskInstancePrediction) -> set[int]:
    """返回去掉 mask 占比限制后的原始覆盖格，用于关联同一目标。"""
    raw = {
        int(cell.index)
        for cell in instance.cells
        if int(cell.overlap_pixels) >= MIN_OVERLAP_PIXELS and float(cell.cell_ratio) >= RAW_INDEX_MIN_CELL_RATIO
    }
    return raw or {int(index) for index in instance.selected_indices}


def _instance_area(instance: MaskInstancePrediction) -> int:
    """返回实例在所有格子中的 mask 像素数。"""
    return sum(int(cell.overlap_pixels) for cell in instance.cells)


def _validate_mask_instances(
    segmentation: SegmentationPrediction,
    by_index: dict[int, TilePrediction],
    target_class: str,
    *,
    classification_threshold: float,
    confidence_threshold: float,
    grid_count: int,
) -> tuple[set[int], int, int]:
    """用分类、实例关联和相对尺寸复核 mask，兼顾漏检与误检。"""
    records: list[_MaskValidationEvidence] = []
    for instance in segmentation.instances:
        indices = _instance_indices(instance, target_class, grid_count)
        if not indices:
            continue
        maximum_score = max(
            (
                _target_candidate(by_index[index], target_class)[0]
                for index in indices
                if index in by_index
            ),
            default=0.0,
        )
        records.append(
            _MaskValidationEvidence(
                instance=instance,
                indices=indices,
                raw_indices=_instance_raw_indices(instance),
                maximum_score=maximum_score,
                area=_instance_area(instance),
                accepted=(
                    float(instance.confidence) >= float(confidence_threshold)
                    and maximum_score >= float(classification_threshold)
                ),
            )
        )

    # 摩托车场景中，近处主体可能与远处小目标同时被高置信检出。
    # 当小实例只占一格且面积明显小于主实例时，优先保留主体。
    accepted = [record for record in records if record.accepted]
    if normalize_dataset_class(target_class) == "Motorcycle" and len(accepted) > 1:
        largest_area = max(record.area for record in accepted)
        if largest_area > 0:
            for record in accepted:
                if (
                    record.area > 0
                    and len(record.indices) == 1
                    and record.area / largest_area < DOMINANT_MOTORCYCLE_MIN_AREA_RATIO
                ):
                    record.accepted = False

    accepted = [record for record in records if record.accepted]
    rescue_threshold = max(
        float(classification_threshold),
        STRONG_CLASSIFICATION_RESCUE_THRESHOLD,
    )
    if not accepted:
        # 常规分割完全漏检时，仅接受有强分类锚点的低置信候选实例。
        for record in records:
            if (
                float(record.instance.confidence) >= SEGMENTATION_RECOVERY_CONFIDENCE
                and record.maximum_score >= rescue_threshold
            ):
                record.accepted = True
    else:
        # 同一窄长目标有时会被模型分成两个实例。低置信半边需要
        # 与已通过实例共享原始覆盖格，且自身存在强分类证据。
        accepted_raw = set().union(*(record.raw_indices for record in accepted))
        for record in records:
            if record.accepted:
                continue
            if (
                float(record.instance.confidence) >= SEGMENTATION_RECOVERY_CONFIDENCE
                and record.maximum_score >= rescue_threshold
                and bool(record.indices & accepted_raw)
            ):
                record.accepted = True

    for record in records:
        if record.accepted:
            _expand_accepted_instance(
                record,
                by_index,
                target_class,
                grid_count,
            )

    validated = set().union(
        *(record.indices for record in records if record.accepted)
    ) if records else set()
    accepted_count = sum(record.accepted for record in records)
    rejected_count = len(records) - accepted_count
    return validated, accepted_count, rejected_count


def fuse_predictions(
    all_classification: list[TilePrediction],
    selected_classification: list[TilePrediction],
    segmentation: SegmentationPrediction,
    *,
    target_class: str,
    grid_count: int,
    mode: str = "balanced",
    weak_classification_threshold: float = THRESHOLDS.weak_evidence.weak_classification_threshold,
    strong_mask_cell_ratio: float = THRESHOLDS.weak_evidence.strong_mask_cell_ratio,
    instance_classification_threshold: float = THRESHOLDS.instance_validation.classification_threshold,
    instance_confidence_threshold: float = THRESHOLDS.instance_validation.confidence_threshold,
) -> FusionResult:
    """按所选策略融合分类格子和分割 mask 格子。"""
    if mode not in FUSION_MODE_LABELS:
        raise ValueError(f"未知融合策略：{mode}")
    selected_class = {item.index for item in selected_classification}
    selected_mask = set(segmentation.selected_indices)
    by_index = {item.index: item for item in all_classification}
    if segmentation.instances:
        validated_mask, accepted_instances, rejected_instances = _validate_mask_instances(
            segmentation,
            by_index,
            target_class,
            classification_threshold=instance_classification_threshold,
            confidence_threshold=instance_confidence_threshold,
            grid_count=grid_count,
        )
    else:
        # 兼容旧缓存和不带实例明细的调用方。
        validated_mask = set(selected_mask)
        accepted_instances = 0
        rejected_instances = 0
    decisions: list[FusionDecision] = []
    final: list[int] = []
    for index in range(grid_count):
        score, rank = _target_candidate(by_index[index], target_class) if index in by_index else (0.0, None)
        class_hit = index in selected_class
        mask_hit = index in selected_mask
        validated_mask_hit = index in validated_mask
        mask_ratio = float(segmentation.cell_scores.get(index, 0.0))
        if not segmentation.supported:
            selected = class_hit
            reason = "分割模型未覆盖该类别，使用分类结果"
        elif mode == "union":
            selected = class_hit or mask_hit
            reason = "分类或分割任一命中" if selected else "两路均未命中"
        elif mode == "consensus":
            selected = class_hit and validated_mask_hit
            reason = "分类与分割共同命中" if selected else "缺少双路共同证据"
        else:
            if segmentation.instances:
                # 平衡模式以“通过分类复核的完整 mask 实例”为最终证据。
                # 一个实例只要有一个格子给出强分类证据，其他格子可由 mask 补齐。
                selected = validated_mask_hit
                if selected and class_hit:
                    reason = "分类与已复核 mask 共同命中"
                elif selected:
                    reason = "同一 mask 实例已通过分类复核"
                elif mask_hit:
                    reason = "mask 实例未通过分类或置信度复核"
                elif class_hit:
                    reason = "分类命中但缺少有效 mask 证据"
                else:
                    reason = "融合证据不足"
            else:
                # 分割未产生 mask 时由分类兜底；旧格式仍保留原有弱证据融合。
                mask_rescue = mask_hit and (
                    score >= float(weak_classification_threshold)
                    or mask_ratio >= float(strong_mask_cell_ratio)
                )
                selected = class_hit or mask_rescue
                if class_hit and mask_hit:
                    reason = "分类与分割共同命中"
                elif class_hit:
                    reason = "分割未产生可复核实例，使用分类结果"
                elif mask_rescue:
                    reason = "分割 mask 与分类弱候选融合命中"
                else:
                    reason = "融合证据不足"
        if selected:
            final.append(index)
        decisions.append(
            FusionDecision(
                index=index,
                selected=selected,
                classification_hit=class_hit,
                segmentation_hit=mask_hit,
                classification_score=score,
                classification_rank=rank,
                mask_cell_ratio=mask_ratio,
                validated_segmentation_hit=validated_mask_hit,
                reason=reason,
            )
        )
    return FusionResult(
        mode=mode,
        selected_indices=final,
        classification_indices=sorted(selected_class),
        segmentation_indices=sorted(selected_mask),
        validated_segmentation_indices=sorted(validated_mask),
        validated_instance_count=accepted_instances,
        rejected_instance_count=rejected_instances,
        instance_classification_threshold=float(instance_classification_threshold),
        instance_confidence_threshold=float(instance_confidence_threshold),
        decisions=decisions,
    )


def format_fusion_report(
    fusion: FusionResult,
    segmentation: SegmentationPrediction,
    *,
    target_class: str,
    spec: GridSpec,
) -> str:
    """生成 main.py 和 GUI 可直接显示、复制的中文融合报告。"""
    details = []
    for decision in fusion.decisions:
        rank = decision.classification_rank if decision.classification_rank is not None else "未进入候选"
        details.append(
            f"格子{decision.index}: 最终={'命中' if decision.selected else '忽略'}; "
            f"分类命中={decision.classification_hit}, 目标概率={decision.classification_score:.4f}, "
            f"排名={rank}; mask命中={decision.segmentation_hit}, "
            f"有效mask={decision.validated_segmentation_hit}, "
            f"格子覆盖率={decision.mask_cell_ratio:.4f}; 原因={decision.reason}"
        )
    instances = []
    for number, instance in enumerate(segmentation.instances, start=1):
        instances.append(
            f"mask{number}: 类别={instance.label}, 置信度={instance.confidence:.4f}, "
            f"覆盖格子={instance.selected_indices}"
        )
    mode_label = FUSION_MODE_LABELS.get(fusion.mode, fusion.mode)
    return (
        f"目标类别: {target_class}\n"
        f"网格: {spec.text}\n"
        f"融合策略: {mode_label}\n"
        f"分类格子: {fusion.classification_indices}\n"
        f"分割原始格子: {fusion.segmentation_indices}\n"
        f"复核后mask格子: {fusion.validated_segmentation_indices}\n"
        f"融合格子: {fusion.selected_indices}\n"
        f"mask实例复核: 通过 {fusion.validated_instance_count} / "
        f"拒绝 {fusion.rejected_instance_count}; "
        f"分类阈值={fusion.instance_classification_threshold:.3f}, "
        f"实例置信度={fusion.instance_confidence_threshold:.3f}\n"
        f"分割类别覆盖: {'是' if segmentation.supported else '否'}\n"
        f"分割说明: {segmentation.message}\n\n"
        f"目标 mask:\n{chr(10).join(instances) if instances else '无'}\n\n"
        f"逐格融合详情:\n{chr(10).join(details)}\n"
    )
