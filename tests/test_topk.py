from challenge_images.training.model_service import ModelService, TilePrediction


def _mixed_prediction() -> TilePrediction:
    return TilePrediction(
        index=3,
        label="Car",
        dataset_class="Car",
        zh="车",
        confidence=0.52,
        candidates=[
            {"rank": 1, "label": "Car", "dataset_class": "Car", "zh": "车", "confidence": 0.52},
            {"rank": 2, "label": "Crosswalk", "dataset_class": "Crosswalk", "zh": "人行横道", "confidence": 0.35},
            {"rank": 3, "label": "Bridge", "dataset_class": "Bridge", "zh": "桥", "confidence": 0.07},
        ],
    )


def test_target_in_top3_above_threshold_is_selected():
    item = _mixed_prediction()
    item.candidates[1]["confidence"] = 0.45
    selected = ModelService.select_target([item], 0.25, "Crosswalk", 3, 0.65)
    assert [item.index for item in selected] == [3]
    assert selected[0].target_rank == 2
    assert selected[0].target_confidence == 0.45


def test_target_outside_top1_is_not_selected_when_k_is_one():
    assert ModelService.select_target([_mixed_prediction()], 0.25, "Crosswalk", 1) == []


def test_target_below_threshold_is_not_selected():
    assert ModelService.select_target([_mixed_prediction()], 0.40, "Crosswalk", 3) == []


def test_multiview_target_evidence_can_select_mixed_tile():
    item = _mixed_prediction()
    item.target_rank = 2
    item.target_confidence = 0.18
    item.target_label = "Crosswalk"
    item.target_dataset_class = "Crosswalk"
    item.evidence_view = "下部 65%"

    selected = ModelService.select_target([item], 0.40, "Crosswalk", 3, 0.15)

    assert [prediction.index for prediction in selected] == [3]
    assert selected[0].evidence_view == "下部 65%"


def test_multiview_evidence_still_respects_top_k():
    item = _mixed_prediction()
    item.target_rank = 4
    item.target_confidence = 0.80
    item.target_label = "Crosswalk"
    item.target_dataset_class = "Crosswalk"
    item.evidence_view = "中央 80%"

    assert ModelService.select_target([item], 0.40, "Crosswalk", 3, 0.50) == []


def test_local_evidence_uses_stricter_multiview_threshold():
    item = _mixed_prediction()
    item.target_rank = 1
    item.target_confidence = 0.41
    item.target_label = "Crosswalk"
    item.target_dataset_class = "Crosswalk"
    item.evidence_view = "下部 80%"

    assert ModelService.select_target([item], 0.40, "Crosswalk", 3, 0.50) == []

    item.target_confidence = 0.83
    assert [x.index for x in ModelService.select_target([item], 0.40, "Crosswalk", 3, 0.50)] == [3]


def test_bus_does_not_use_local_crop_evidence():
    item = _mixed_prediction()
    item.candidates = [
        {"rank": 1, "label": "Car", "dataset_class": "Car", "zh": "车", "confidence": 0.99},
        {"rank": 2, "label": "Bus", "dataset_class": "Bus", "zh": "公共汽车", "confidence": 0.003},
    ]
    item.target_rank = 1
    item.target_confidence = 0.98
    item.target_label = "Bus"
    item.target_dataset_class = "Bus"
    item.evidence_view = "下部 65%"

    assert ModelService.select_target([item], 0.25, "Bus", 3, 0.80, 0.80) == []


def test_low_confidence_top1_respects_direct_threshold():
    item = _mixed_prediction()
    item.label = "Bus"
    item.dataset_class = "Bus"
    item.confidence = 0.5883
    item.candidates = [
        {"rank": 1, "label": "Bus", "dataset_class": "Bus", "zh": "公共汽车", "confidence": 0.5883},
        {"rank": 2, "label": "Car", "dataset_class": "Car", "zh": "车", "confidence": 0.3145},
    ]

    assert ModelService.select_target([item], 0.25, "Bus", 3, None, 0.80) == []


def test_crosswalk_local_crop_does_not_override_bicycle():
    item = _mixed_prediction()
    item.label = "Bicycle"
    item.dataset_class = "Bicycle"
    item.confidence = 0.814
    item.candidates = [
        {"rank": 1, "label": "Bicycle", "dataset_class": "Bicycle", "zh": "自行车", "confidence": 0.814},
        {"rank": 2, "label": "Crosswalk", "dataset_class": "Crosswalk", "zh": "人行横道", "confidence": 0.183},
    ]
    item.target_rank = 1
    item.target_confidence = 0.843
    item.target_label = "Crosswalk"
    item.target_dataset_class = "Crosswalk"
    item.evidence_view = "下部 65%"

    assert ModelService.select_target([item], 0.30, "Crosswalk", 3, 0.65, 0.60) == []


def test_crosswalk_candidate_from_car_uses_stricter_threshold():
    item = _mixed_prediction()
    item.confidence = 0.615
    item.candidates[0]["confidence"] = 0.615
    item.candidates[1]["confidence"] = 0.375

    assert ModelService.select_target([item], 0.30, "Crosswalk", 3, 0.65, 0.60) == []

    item.candidates[1]["confidence"] = 0.453
    assert [x.index for x in ModelService.select_target([item], 0.30, "Crosswalk", 3, 0.65, 0.60)] == [3]
