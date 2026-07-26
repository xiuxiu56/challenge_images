import numpy as np
from PIL import Image

from challenge_images.grid.grid_engine import GridSpec
from challenge_images.segmentation.category_map import segmentation_category_key
from challenge_images.segmentation.mask_grid import (
    CellMaskEvidence,
    mask_bottom_extensions,
    mask_grid_evidence,
    render_mask_overlay,
)
from challenge_images.segmentation.model_service import (
    SEGMENTATION_RECOVERY_CONFIDENCE,
    MaskInstancePrediction,
    SegmentationModelService,
    SegmentationPrediction,
)
from challenge_images.segmentation.result_fusion import fuse_predictions
from challenge_images.training.model_service import TilePrediction


def _classification(index: int, target_score: float) -> TilePrediction:
    return TilePrediction(
        index=index,
        label="Car",
        dataset_class="Car",
        zh="车",
        confidence=1.0 - target_score,
        candidates=[
            {
                "rank": 1,
                "label": "Car",
                "dataset_class": "Car",
                "zh": "车",
                "confidence": 1.0 - target_score,
            },
            {
                "rank": 2,
                "label": "Crosswalk",
                "dataset_class": "Crosswalk",
                "zh": "人行横道",
                "confidence": target_score,
            },
        ],
    )


def _classification_for(
    index: int,
    target_class: str,
    target_score: float,
) -> TilePrediction:
    return TilePrediction(
        index=index,
        label=target_class if target_score >= 0.5 else "Car",
        dataset_class=target_class if target_score >= 0.5 else "Car",
        zh=target_class,
        confidence=max(target_score, 1.0 - target_score),
        candidates=[
            {
                "rank": 1,
                "label": target_class,
                "dataset_class": target_class,
                "zh": target_class,
                "confidence": target_score,
            }
        ],
    )


def _segmentation(indices: list[int], scores: dict[int, float], supported: bool = True):
    return SegmentationPrediction(
        target_class="Crosswalk",
        target_key="Crosswalk",
        supported=supported,
        selected_indices=indices,
        instances=[],
        cell_scores=scores,
        preview=Image.new("RGB", (90, 90)),
    )


def _segmentation_with_instances(
    instances: list[MaskInstancePrediction],
) -> SegmentationPrediction:
    indices = sorted(
        {index for instance in instances for index in instance.selected_indices}
    )
    return SegmentationPrediction(
        target_class="Crosswalk",
        target_key="Crosswalk",
        supported=True,
        selected_indices=indices,
        instances=instances,
        cell_scores={index: 0.20 for index in indices},
        preview=Image.new("RGB", (90, 90)),
    )


def _mask_instance(
    confidence: float,
    indices: list[int],
    *,
    cells: list[CellMaskEvidence] | None = None,
    label: str = "crosswalk",
    category_key: str = "Crosswalk",
    bottom_extensions: list[int] | None = None,
) -> MaskInstancePrediction:
    return MaskInstancePrediction(
        class_id=0,
        label=label,
        category_key=category_key,
        confidence=confidence,
        selected_indices=indices,
        cells=cells or [],
        bottom_extension_indices=bottom_extensions or [],
    )


def _cell(
    index: int,
    cell_ratio: float,
    mask_ratio: float,
    overlap_pixels: int,
    selected: bool,
    *,
    touches_left: bool = False,
    touches_right: bool = False,
    touches_top: bool = False,
    touches_bottom: bool = False,
) -> CellMaskEvidence:
    return CellMaskEvidence(
        index=index,
        overlap_pixels=overlap_pixels,
        cell_ratio=cell_ratio,
        mask_ratio=mask_ratio,
        selected=selected,
        touches_left=touches_left,
        touches_right=touches_right,
        touches_top=touches_top,
        touches_bottom=touches_bottom,
    )


def test_segmentation_category_map_supports_coco_and_custom_labels():
    assert segmentation_category_key("fire hydrant") == "Hydrant"
    assert segmentation_category_key("school bus") == "Bus"
    assert segmentation_category_key("Crosswalk") == "Crosswalk"
    assert segmentation_category_key("boat") == "Boat"


def test_mask_is_converted_to_three_by_three_cells():
    mask = np.zeros((90, 90), dtype=np.float32)
    mask[30:60, 30:60] = 1.0

    evidence = mask_grid_evidence(mask, GridSpec(3, 3))

    assert [item.index for item in evidence if item.selected] == [4]
    assert evidence[4].cell_ratio == 1.0


def test_mask_can_cover_multiple_four_by_four_cells():
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[20:60, 20:60] = 1.0

    evidence = mask_grid_evidence(mask, GridSpec(4, 4))

    assert [item.index for item in evidence if item.selected] == [5, 6, 9, 10]


def test_mask_fringe_requires_cell_and_instance_share_thresholds():
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[0:20, 0:20] = 1.0
    mask[0:3, 20:30] = 1.0

    evidence = mask_grid_evidence(
        mask,
        GridSpec(4, 4),
        min_cell_ratio=0.002,
        min_mask_ratio=0.10,
    )

    assert [item.index for item in evidence if item.selected] == [0]


def test_mask_bottom_extension_only_uses_the_real_bottom_edge():
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[40:58, 20:40] = 1.0

    extensions = mask_bottom_extensions(mask, GridSpec(4, 4))

    assert extensions == [13]


def test_balanced_fusion_uses_mask_to_rescue_weak_topk_candidate():
    all_items = [_classification(index, 0.01) for index in range(9)]
    all_items[4] = _classification(4, 0.20)

    fused = fuse_predictions(
        all_items,
        [],
        _segmentation([4], {4: 0.02}),
        target_class="Crosswalk",
        grid_count=9,
        mode="balanced",
        weak_classification_threshold=0.10,
        strong_mask_cell_ratio=0.01,
    )

    assert fused.selected_indices == [4]
    assert fused.decisions[4].reason == "分割 mask 与分类弱候选融合命中"


def test_balanced_fusion_rejects_unverified_instances_and_class_only_hits():
    scores = {0: 0.02, 3: 0.002, 4: 0.99, 7: 0.99, 13: 0.85}
    all_items = [_classification(index, scores.get(index, 0.0)) for index in range(16)]
    segmentation = _segmentation_with_instances(
        [
            _mask_instance(0.90, [0, 4]),
            _mask_instance(0.85, [3]),
            _mask_instance(0.54, [7]),
        ]
    )

    fused = fuse_predictions(
        all_items,
        [all_items[4], all_items[7], all_items[13]],
        segmentation,
        target_class="Crosswalk",
        grid_count=16,
        mode="balanced",
        instance_classification_threshold=0.80,
        instance_confidence_threshold=0.60,
    )

    assert fused.selected_indices == [0, 4]
    assert fused.validated_segmentation_indices == [0, 4]
    assert fused.validated_instance_count == 1
    assert fused.rejected_instance_count == 2
    assert fused.decisions[13].reason == "分类命中但缺少有效 mask 证据"


def test_balanced_fusion_rejects_a_whole_false_cross_grid_instance():
    scores = {4: 0.016, 5: 0.383, 6: 0.143, 8: 0.418, 9: 0.979, 10: 0.370, 12: 0.016}
    all_items = [_classification(index, scores.get(index, 0.0)) for index in range(16)]
    segmentation = _segmentation_with_instances(
        [
            _mask_instance(0.93, [5, 6, 9, 10]),
            _mask_instance(0.66, [4, 8, 12]),
        ]
    )

    fused = fuse_predictions(
        all_items,
        [all_items[8], all_items[9], all_items[10]],
        segmentation,
        target_class="Crosswalk",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [5, 6, 9, 10]
    assert fused.validated_instance_count == 1
    assert fused.rejected_instance_count == 1


def test_motorcycle_fusion_recovers_mask_fringe_and_rejects_small_background_instance():
    scores = {0: 0.99, 6: 0.68, 9: 0.99, 10: 0.99, 13: 0.04}
    all_items = [
        _classification_for(index, "Motorcycle", scores.get(index, 0.0))
        for index in range(16)
    ]
    main = _mask_instance(
        0.92,
        [9, 10],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[
            _cell(6, 0.044, 0.059, 557, False),
            _cell(9, 0.244, 0.330, 3118, True),
            _cell(10, 0.432, 0.583, 5511, True),
            _cell(13, 0.020, 0.026, 247, False),
        ],
    )
    background = _mask_instance(
        0.80,
        [0],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[_cell(0, 0.25, 1.0, 1000, True)],
    )
    segmentation = _segmentation_with_instances([main, background])

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (0, 6, 9, 10)],
        segmentation,
        target_class="Motorcycle",
        grid_count=16,
        mode="balanced",
        instance_classification_threshold=0.60,
        instance_confidence_threshold=0.60,
    )

    assert fused.selected_indices == [6, 9, 10, 13]
    assert fused.validated_instance_count == 1
    assert fused.rejected_instance_count == 1


def test_motorcycle_fusion_does_not_expand_outside_a_two_dimensional_main_instance():
    scores = {0: 0.99, 4: 0.99, 5: 0.98, 9: 0.96, 13: 0.37}
    all_items = [
        _classification_for(index, "Motorcycle", scores.get(index, 0.0))
        for index in range(16)
    ]
    main = _mask_instance(
        0.84,
        [4, 5, 8, 9],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[
            _cell(4, 0.16, 0.12, 2066, True),
            _cell(5, 0.24, 0.17, 3069, True),
            _cell(8, 0.19, 0.14, 2394, True),
            _cell(9, 0.67, 0.48, 8499, True),
            _cell(13, 0.14, 0.097, 1722, False),
        ],
    )
    background = _mask_instance(
        0.80,
        [0],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[_cell(0, 0.25, 1.0, 3154, True)],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (0, 4, 5, 9, 13)],
        _segmentation_with_instances([main, background]),
        target_class="Motorcycle",
        grid_count=16,
        mode="balanced",
        instance_classification_threshold=0.60,
        instance_confidence_threshold=0.60,
    )

    assert fused.selected_indices == [4, 5, 8, 9]
    assert fused.validated_instance_count == 1
    assert fused.rejected_instance_count == 1


def test_motorcycle_fusion_expands_horizontally_then_fills_the_instance_box():
    scores = {5: 0.93, 8: 0.95, 9: 1.0}
    all_items = [
        _classification_for(index, "Motorcycle", scores.get(index, 0.0))
        for index in range(16)
    ]
    instance = _mask_instance(
        0.91,
        [5, 8, 9],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[
            _cell(4, 0.0874, 0.04, 1100, False),
            _cell(5, 0.3186, 0.25, 4000, True),
            _cell(6, 0.0072, 0.006, 91, False),
            _cell(8, 0.2354, 0.19, 3000, True),
            _cell(9, 0.3194, 0.25, 4050, True),
            _cell(10, 0.0919, 0.07, 1160, False),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (5, 8, 9)],
        _segmentation_with_instances([instance]),
        target_class="Motorcycle",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [4, 5, 6, 8, 9, 10]


def test_motorcycle_fusion_recovers_right_edge_and_internal_mask_cells():
    scores = {5: 1.0, 11: 0.84, 13: 0.31}
    all_items = [
        _classification_for(index, "Motorcycle", scores.get(index, 0.0))
        for index in range(16)
    ]
    instance = _mask_instance(
        0.80,
        [5, 9, 10, 14],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[
            _cell(5, 0.2292, 0.18, 2900, True),
            _cell(9, 0.3535, 0.28, 4500, True),
            _cell(10, 0.5610, 0.36, 7100, True),
            _cell(11, 0.0674, 0.04, 850, False),
            _cell(13, 0.0259, 0.015, 330, False),
            _cell(14, 0.5409, 0.35, 6850, True),
            _cell(15, 0.1359, 0.08, 1720, False),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (5, 11, 13)],
        _segmentation_with_instances([instance]),
        target_class="Motorcycle",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [5, 9, 10, 11, 13, 14, 15]


def test_low_confidence_motorcycle_candidate_is_rescued_by_strong_classification():
    scores = {9: 0.997, 14: 0.38}
    all_items = [
        _classification_for(index, "Motorcycle", scores.get(index, 0.0))
        for index in range(16)
    ]
    recovered = _mask_instance(
        0.082,
        [9, 10, 13, 14, 15],
        label="motorcycle",
        category_key="Motorcycle",
        cells=[
            _cell(9, 0.25, 0.15, 3200, True),
            _cell(10, 0.25, 0.14, 3100, True),
            _cell(11, 0.16, 0.09, 2000, False),
            _cell(13, 0.24, 0.14, 3000, True),
            _cell(14, 0.23, 0.13, 2800, True),
            _cell(15, 0.59, 0.34, 7300, True),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[9]],
        _segmentation_with_instances([recovered]),
        target_class="Motorcycle",
        grid_count=16,
        mode="balanced",
        instance_classification_threshold=0.60,
        instance_confidence_threshold=0.60,
    )

    assert fused.selected_indices == [9, 10, 11, 13, 14, 15]
    assert fused.validated_instance_count == 1


def test_linked_low_confidence_half_mask_is_merged_into_validated_traffic_light():
    scores = {5: 1.0, 8: 0.998, 9: 1.0}
    all_items = [
        _classification_for(index, "Traffic Light", scores.get(index, 0.0))
        for index in range(16)
    ]
    primary = _mask_instance(
        0.80,
        [5, 9],
        label="traffic light",
        category_key="Traffic Light",
        cells=[
            _cell(4, 0.03, 0.05, 380, False),
            _cell(5, 0.41, 0.64, 5200, True),
            _cell(8, 0.06, 0.09, 720, False),
            _cell(9, 0.13, 0.21, 1700, True),
        ],
    )
    linked_half = _mask_instance(
        0.32,
        [4, 8],
        label="traffic light",
        category_key="Traffic Light",
        cells=[
            _cell(4, 0.005, 0.42, 61, True),
            _cell(8, 0.007, 0.59, 86, True),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (5, 8, 9)],
        _segmentation_with_instances([primary, linked_half]),
        target_class="Traffic Light",
        grid_count=16,
        mode="balanced",
        instance_classification_threshold=0.60,
        instance_confidence_threshold=0.60,
    )

    assert fused.selected_indices == [4, 5, 8, 9]
    assert fused.validated_instance_count == 2


def test_traffic_light_fusion_recovers_a_classified_thin_mask_edge():
    scores = {5: 1.0, 6: 0.3174, 9: 1.0, 10: 1.0}
    all_items = [
        _classification_for(index, "Traffic Light", scores.get(index, 0.0))
        for index in range(16)
    ]
    left = _mask_instance(
        0.61,
        [5, 9],
        label="traffic light",
        category_key="Traffic Light",
        cells=[
            _cell(5, 0.1554, 0.48, 1970, True),
            _cell(9, 0.1324, 0.41, 1680, True),
        ],
    )
    right = _mask_instance(
        0.82,
        [10],
        label="traffic light",
        category_key="Traffic Light",
        cells=[
            _cell(6, 0.0045, 0.0299, 57, False),
            _cell(10, 0.1475, 0.9701, 1870, True),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (5, 6, 9, 10)],
        _segmentation_with_instances([left, right]),
        target_class="Traffic Light",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [5, 6, 9, 10]


def test_traffic_light_fusion_keeps_known_false_cells_excluded():
    scores = {2: 1.0, 3: 0.0025, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.99, 13: 0.85}
    all_items = [
        _classification_for(index, "Traffic Light", scores.get(index, 0.0))
        for index in range(16)
    ]
    instances = [
        _mask_instance(
            0.89,
            [1, 5],
            label="traffic light",
            category_key="Traffic Light",
            cells=[
                _cell(1, 0.0247, 0.1920, 310, True),
                _cell(4, 0.0119, 0.0920, 150, False),
                _cell(5, 0.0903, 0.7074, 1140, True),
            ],
        ),
        _mask_instance(0.88, [2, 6], label="traffic light", category_key="Traffic Light"),
        _mask_instance(0.88, [0, 4], label="traffic light", category_key="Traffic Light"),
        _mask_instance(0.85, [3], label="traffic light", category_key="Traffic Light"),
        _mask_instance(0.75, [2], label="traffic light", category_key="Traffic Light"),
        _mask_instance(0.54, [7], label="traffic light", category_key="Traffic Light"),
    ]

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (2, 4, 5, 6, 7, 13)],
        _segmentation_with_instances(instances),
        target_class="Traffic Light",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [0, 1, 2, 4, 5, 6]


def test_bus_fusion_recovers_connected_lower_fringe_without_roof_leakage():
    scores = {4: 0.86, 5: 0.99, 6: 0.93}
    all_items = [
        _classification_for(index, "Bus", scores.get(index, 0.0))
        for index in range(16)
    ]
    instance = _mask_instance(
        0.93,
        [5, 6],
        label="bus",
        category_key="Bus",
        cells=[
            _cell(1, 0.20, 0.04, 2500, False),
            _cell(2, 0.18, 0.04, 2300, False),
            _cell(4, 0.0811, 0.06, 1030, False),
            _cell(5, 0.9801, 0.45, 12400, True),
            _cell(6, 0.4886, 0.22, 6200, True),
            _cell(8, 0.0045, 0.002, 57, False, touches_right=True),
            _cell(9, 0.0586, 0.03, 740, False, touches_left=True),
            _cell(10, 0.0046, 0.002, 58, False),
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (4, 5, 6)],
        _segmentation_with_instances([instance]),
        target_class="Bus",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [4, 5, 6, 8, 9]


def test_bus_fusion_recovers_bottom_grid_line_extension():
    scores = {5: 1.0, 6: 0.94, 9: 1.0}
    all_items = [
        _classification_for(index, "Bus", scores.get(index, 0.0))
        for index in range(16)
    ]
    instance = _mask_instance(
        0.92,
        [5, 6, 9, 10],
        label="bus",
        category_key="Bus",
        cells=[
            _cell(5, 0.36, 0.18, 4600, True),
            _cell(6, 0.57, 0.28, 7200, True),
            _cell(9, 0.56, 0.28, 7100, True),
            _cell(10, 0.44, 0.22, 5600, True),
        ],
        bottom_extensions=[13],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (5, 6, 9, 14)],
        _segmentation_with_instances([instance]),
        target_class="Bus",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [5, 6, 9, 10, 13]


def test_bus_fusion_expands_from_the_primary_row_downward():
    scores = {0: 0.99, 6: 0.89}
    all_items = [
        _classification_for(index, "Bus", scores.get(index, 0.0))
        for index in range(16)
    ]
    ratios = {
        0: 0.8794,
        1: 0.4138,
        2: 0.4678,
        4: 0.9412,
        5: 0.8413,
        6: 0.9324,
        7: 0.1482,
        8: 0.9370,
        9: 0.5979,
        10: 0.3353,
        11: 0.0171,
        12: 0.0932,
    }
    instance = _mask_instance(
        0.89,
        [0, 4, 5, 6, 8],
        label="bus",
        category_key="Bus",
        cells=[
            _cell(index, ratio, 0.05, max(20, int(ratio * 12000)), index in {0, 4, 5, 6, 8})
            for index, ratio in ratios.items()
        ],
    )

    fused = fuse_predictions(
        all_items,
        [all_items[index] for index in (0, 6)],
        _segmentation_with_instances([instance]),
        target_class="Bus",
        grid_count=16,
        mode="balanced",
    )

    assert fused.selected_indices == [0, 4, 5, 6, 7, 8, 9, 10, 12]


def test_segmentation_retries_at_recovery_confidence_when_normal_threshold_misses(monkeypatch):
    service = SegmentationModelService()
    service.model = object()
    service.class_names = {3: "motorcycle"}
    calls: list[float] = []
    recovered = _mask_instance(
        0.08,
        [9],
        label="motorcycle",
        category_key="Motorcycle",
    )

    def fake_predict(*_args, confidence: float, **_kwargs):
        calls.append(confidence)
        if confidence > SEGMENTATION_RECOVERY_CONFIDENCE:
            return [], [], {}, set()
        return [recovered], [], {9: 0.20}, {9}

    monkeypatch.setattr(service, "_predict_target_instances", fake_predict)
    result = service.predict(
        Image.new("RGB", (80, 80)),
        GridSpec(4, 4),
        "Motorcycle",
        confidence=0.25,
    )

    assert calls == [0.25, SEGMENTATION_RECOVERY_CONFIDENCE]
    assert result.selected_indices == [9]
    assert "找回 1 个候选 mask" in result.message


def test_fusion_falls_back_to_classification_for_unsupported_target():
    all_items = [_classification(index, 0.01) for index in range(9)]
    selected = [all_items[2]]

    fused = fuse_predictions(
        all_items,
        selected,
        _segmentation([], {}, supported=False),
        target_class="Crosswalk",
        grid_count=9,
    )

    assert fused.selected_indices == [2]


def test_mask_overlay_preserves_image_size():
    image = Image.new("RGB", (101, 99), "white")
    mask = np.ones((30, 30), dtype=np.float32)

    preview = render_mask_overlay(image, [mask], GridSpec(3, 3), [0, 1])

    assert preview.size == image.size
