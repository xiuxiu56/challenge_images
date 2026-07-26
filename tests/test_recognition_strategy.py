"""统一识别路由、参数预设和点击计划单测。"""

from challenge_images.grid.grid_engine import GridSpec
from challenge_images.recognition.click_plan import ClickSettings, build_click_plan
from challenge_images.recognition.policy import parameters_for, resolve_recognition_route


def test_smart_route_uses_classifier_for_three_by_three_dynamic():
    route = resolve_recognition_route(
        "smart",
        challenge_type="dynamic",
        spec=GridSpec(3, 3),
        target_class="Crosswalk",
        segmentation_loaded=True,
        segmentation_supported=True,
    )

    assert route.actual_mode == "classifier"
    assert route.use_multiview is True
    assert "3×3" in route.reason


def test_smart_route_uses_fusion_for_supported_four_by_four_target():
    route = resolve_recognition_route(
        "smart",
        challenge_type="multicaptcha",
        spec=GridSpec(4, 4),
        target_class="Motorcycle",
        segmentation_loaded=True,
        segmentation_supported=True,
    )

    assert route.actual_mode == "fusion"
    assert route.use_multiview is False


def test_fusion_route_falls_back_when_segmentation_has_no_target():
    route = resolve_recognition_route(
        "fusion",
        challenge_type="multicaptcha",
        spec=GridSpec(4, 4),
        target_class="Crosswalk",
        segmentation_loaded=True,
        segmentation_supported=False,
    )

    assert route.actual_mode == "classifier"
    assert "未覆盖目标类别" in route.reason


def test_parameter_presets_have_clear_precision_recall_order():
    precision = parameters_for("precision", "Motorcycle")
    balanced = parameters_for("balanced", "Motorcycle")
    recall = parameters_for("recall", "Motorcycle")

    assert precision.classification_top1 > balanced.classification_top1
    assert recall.classification_top1 < balanced.classification_top1
    assert precision.instance_confidence_threshold > balanced.instance_confidence_threshold
    assert recall.instance_confidence_threshold < balanced.instance_confidence_threshold


def test_click_plan_distinguishes_static_dynamic_and_continuous_grids():
    static = build_click_plan("imageselect", GridSpec(3, 3), [5, 2])
    dynamic = build_click_plan("dynamic", GridSpec(3, 3), [5, 2])
    continuous = build_click_plan("multicaptcha", GridSpec(4, 4), [10, 4])

    assert static.strategy == "static_batch"
    assert static.indices == [2, 5]
    assert static.watch_after_ms == 0
    assert dynamic.strategy == "dynamic_sequential"
    assert dynamic.watch_after_ms == 8_000
    assert continuous.strategy == "continuous_batch"


def test_click_plan_passes_adjustable_timing_and_blocks_near_all_selection():
    settings = ClickSettings(
        delay_ms=350,
        dynamic_wait_ms=12_000,
        auto_verify=True,
        maximum_selected_ratio=0.80,
    )
    dynamic = build_click_plan("dynamic", GridSpec(3, 3), [1, 4], settings)
    blocked = build_click_plan("multicaptcha", GridSpec(4, 4), list(range(15)), settings)

    assert dynamic.delay_ms == 350
    assert dynamic.watch_after_ms == 12_000
    assert dynamic.click_verify is True
    assert blocked.blocked is True
    assert blocked.strategy == "blocked"


def test_parameters_follow_model_training_resolution():
    """已加载权重时推理分辨率跟随其训练分辨率，避免尺寸错配掉点。"""
    from challenge_images.config import DEFAULT_TRAIN_IMGSZ

    fallback = parameters_for("balanced", "Crosswalk")
    assert fallback.classification_imgsz == DEFAULT_TRAIN_IMGSZ

    matched = parameters_for("balanced", "Crosswalk", model_imgsz=320)
    assert matched.classification_imgsz == 320
    # 融合链路同样按整格送入分类模型，不再单独降级到 224。
    assert matched.fusion_classification_imgsz == 320
