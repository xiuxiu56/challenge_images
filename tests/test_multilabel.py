import json

from challenge_images.data.multilabel import (
    MANIFEST_FILENAME,
    MultiLabelManifest,
    build_manifest_from_folders,
    format_manifest_report,
    iter_dataset_images,
    list_classes,
    manifest_statistics,
)


def _write(root, split, class_name, name, content):
    directory = root / split / class_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)


def test_manifest_defaults_to_folder_label():
    manifest = MultiLabelManifest(classes=["Bus", "Car", "Crosswalk"])
    assert manifest.labels_for("train/Car/0001.jpg", "Car") == ["Car"]


def test_manifest_override_wins():
    manifest = MultiLabelManifest(
        classes=["Bus", "Car", "Crosswalk"],
        overrides={"train/Bus/0001.jpg": ["Bus", "Crosswalk"]},
    )
    assert manifest.labels_for("train/Bus/0001.jpg", "Bus") == ["Bus", "Crosswalk"]


def test_multi_hot_encodes_all_labels():
    manifest = MultiLabelManifest(classes=["Bus", "Car", "Crosswalk"])
    assert manifest.multi_hot(["Bus", "Crosswalk"]) == [1.0, 0.0, 1.0]
    # 未知类别被忽略而不是报错，便于清单与模型类别不同步时降级。
    assert manifest.multi_hot(["Bus", "Boat"]) == [1.0, 0.0, 0.0]


def test_multi_hot_is_not_normalised():
    """多标签的核心：多个类别可以同时为 1，不共享概率预算。"""
    manifest = MultiLabelManifest(classes=["Bus", "Car", "Crosswalk"])
    vector = manifest.multi_hot(["Bus", "Car", "Crosswalk"])
    assert sum(vector) == 3.0


def test_build_manifest_detects_composite_tiles(tmp_path):
    """同一张图出现在两个类别目录 = 复合图块，应记为多标签。"""
    composite = b"bus-parked-on-crosswalk"
    _write(tmp_path, "train", "Bus", "a.jpg", composite)
    _write(tmp_path, "train", "Crosswalk", "b.jpg", composite)
    _write(tmp_path, "train", "Bus", "plain.jpg", b"just-a-bus")
    _write(tmp_path, "train", "Car", "car.jpg", b"just-a-car")

    manifest = build_manifest_from_folders(tmp_path)
    assert manifest.classes == ["Bus", "Car", "Crosswalk"]
    # 两个副本都被标成同一组标签。
    assert manifest.overrides["train/Bus/a.jpg"] == ["Bus", "Crosswalk"]
    assert manifest.overrides["train/Crosswalk/b.jpg"] == ["Bus", "Crosswalk"]
    # 单标签图片不写进覆盖表，保持清单轻量。
    assert "train/Bus/plain.jpg" not in manifest.overrides
    assert "train/Car/car.jpg" not in manifest.overrides


def test_build_manifest_handles_three_way_overlap(tmp_path):
    payload = b"car-crosswalk-trafficlight"
    for class_name in ("Car", "Crosswalk", "Traffic Light"):
        _write(tmp_path, "train", class_name, "x.jpg", payload)

    manifest = build_manifest_from_folders(tmp_path)
    labels = manifest.overrides["train/Car/x.jpg"]
    assert labels == ["Car", "Crosswalk", "Traffic Light"]
    assert sum(manifest.multi_hot(labels)) == 3.0


def test_manifest_roundtrip(tmp_path):
    manifest = MultiLabelManifest(
        classes=["Bus", "Crosswalk"],
        overrides={"train/Bus/a.jpg": ["Bus", "Crosswalk"]},
    )
    path = manifest.save(tmp_path / MANIFEST_FILENAME)
    loaded = MultiLabelManifest.load(path)
    assert loaded.classes == manifest.classes
    assert loaded.overrides == manifest.overrides

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["多标签图片数"] == 1


def test_load_for_dataset_returns_none_when_absent(tmp_path):
    assert MultiLabelManifest.load_for_dataset(tmp_path) is None
    MultiLabelManifest(classes=["Bus"]).save(tmp_path / MANIFEST_FILENAME)
    assert MultiLabelManifest.load_for_dataset(tmp_path) is not None


def test_statistics_group_by_combination(tmp_path):
    manifest = MultiLabelManifest(
        classes=["Bus", "Car", "Crosswalk"],
        overrides={
            "train/Bus/a.jpg": ["Bus", "Crosswalk"],
            "train/Crosswalk/b.jpg": ["Bus", "Crosswalk"],
            "train/Car/c.jpg": ["Car", "Crosswalk"],
        },
    )
    stats = manifest_statistics(manifest)
    assert stats["多标签图片数"] == 3
    assert stats["组合分布"]["Bus + Crosswalk"] == 2
    assert stats["涉及类别"]["Crosswalk"] == 3
    assert "Bus + Crosswalk" in format_manifest_report(manifest)


def test_iter_dataset_images_reports_relative_paths(tmp_path):
    _write(tmp_path, "train", "Bus", "a.jpg", b"1")
    _write(tmp_path, "val", "Car", "b.jpg", b"2")
    items = iter_dataset_images(tmp_path)
    relatives = {relative for relative, _, _ in items}
    assert relatives == {"train/Bus/a.jpg", "val/Car/b.jpg"}


def test_list_classes_uses_train_ordering(tmp_path):
    _write(tmp_path, "train", "Crosswalk", "a.jpg", b"1")
    _write(tmp_path, "train", "Bus", "b.jpg", b"2")
    assert list_classes(tmp_path) == ["Bus", "Crosswalk"]


# ---------- 与识别引擎的接口兼容 ----------


def test_multilabel_service_is_drop_in_for_model_service():
    """多标签服务必须能直接替换 ModelService 交给 RecognitionEngine。"""
    import inspect

    from challenge_images.training.model_service import ModelService
    from challenge_images.training.multilabel_service import MultiLabelModelService

    for method in ("load", "predict_grid", "select_target", "supports_multiview_target"):
        assert hasattr(MultiLabelModelService, method), method

    for method in ("predict_grid", "select_target"):
        single = inspect.signature(getattr(ModelService, method))
        multi = inspect.signature(getattr(MultiLabelModelService, method))
        assert set(single.parameters) <= set(multi.parameters), method

    service = MultiLabelModelService()
    assert service.loaded is False
    assert service.training_imgsz is None


def test_multilabel_never_requests_multiview():
    """每类独立打分后，四视角裁剪补丁失去存在意义。"""
    from challenge_images.training.multilabel_service import MultiLabelModelService

    assert MultiLabelModelService.supports_multiview_target("Crosswalk") is False
    assert MultiLabelModelService.supports_multiview_target(None) is False


def test_select_target_uses_independent_threshold():
    """判定退化为单次阈值比较，不再依赖 Top-K 排名。"""
    from challenge_images.training.model_service import TilePrediction
    from challenge_images.training.multilabel_service import MultiLabelModelService

    # Crosswalk 排名第 3，但独立概率很高——单标签下会被 Top-K 与抑制阈值挡掉。
    tile = TilePrediction(
        index=0,
        label="Car",
        dataset_class="Car",
        zh="车",
        confidence=0.95,
        candidates=[
            {"rank": 1, "label": "Car", "dataset_class": "Car", "zh": "车", "confidence": 0.95},
            {"rank": 2, "label": "Bus", "dataset_class": "Bus", "zh": "公共汽车", "confidence": 0.88},
            {"rank": 3, "label": "Crosswalk", "dataset_class": "Crosswalk", "zh": "人行横道", "confidence": 0.86},
        ],
    )
    selected = MultiLabelModelService.select_target([tile], 0.5, "Crosswalk")
    assert len(selected) == 1
    assert selected[0].target_dataset_class == "Crosswalk"
    assert selected[0].target_confidence == 0.86
    assert selected[0].evidence_view == "多标签独立概率"

    # 低于阈值时不命中。
    assert MultiLabelModelService.select_target([tile], 0.9, "Crosswalk") == []


def test_composite_tile_can_match_two_targets():
    """同一个格子对两个不同目标类别都应命中——这正是多标签的意义。"""
    from challenge_images.training.model_service import TilePrediction
    from challenge_images.training.multilabel_service import MultiLabelModelService

    tile = TilePrediction(
        index=3,
        label="Bus",
        dataset_class="Bus",
        zh="公共汽车",
        confidence=0.93,
        candidates=[
            {"rank": 1, "label": "Bus", "dataset_class": "Bus", "zh": "公共汽车", "confidence": 0.93},
            {"rank": 2, "label": "Crosswalk", "dataset_class": "Crosswalk", "zh": "人行横道", "confidence": 0.91},
        ],
    )
    assert len(MultiLabelModelService.select_target([tile], 0.5, "Bus")) == 1
    assert len(MultiLabelModelService.select_target([tile], 0.5, "Crosswalk")) == 1
