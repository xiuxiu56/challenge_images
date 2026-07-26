import pytest

from challenge_images.data.segmentation_prelabel import (
    MAX_POLYGON_POINTS,
    SEGMENTATION_CLASSES,
    _normalise_polygon,
    collect_challenge_images,
    format_prelabel_report,
    uncovered_classes,
    PrelabelReport,
)


def test_class_order_matches_data_yaml_template():
    """标签里的类别编号必须与 data.yaml 顺序一致，否则训练标签全错位。"""
    from pathlib import Path

    template = Path("data/segmentation/recaptcha_seg_v1/data.yaml.example")
    if not template.is_file():
        pytest.skip("缺少 data.yaml.example 模板")
    parsed: dict[int, str] = {}
    inside = False
    for line in template.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            inside = True
            continue
        if inside and line.strip() and line.startswith("  "):
            index, name = line.strip().split(":", 1)
            parsed[int(index)] = name.strip()
        elif inside and line.strip() and not line.startswith("  "):
            break
    assert [parsed[i] for i in sorted(parsed)] == SEGMENTATION_CLASSES


def test_normalise_polygon_scales_to_unit_range():
    points = [(0, 0), (100, 0), (100, 50), (0, 50)]
    flat = _normalise_polygon(points, 200, 100)
    assert flat == [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
    assert all(0.0 <= value <= 1.0 for value in flat)


def test_normalise_polygon_clamps_out_of_bounds():
    """模型偶尔给出略微越界的坐标，必须夹到 [0,1] 否则训练器报错。"""
    flat = _normalise_polygon([(-5, -5), (250, 120), (10, 10)], 200, 100)
    assert all(0.0 <= value <= 1.0 for value in flat)


def test_normalise_polygon_rejects_degenerate_shapes():
    assert _normalise_polygon([(0, 0), (1, 1)], 100, 100) == []
    assert _normalise_polygon([], 100, 100) == []


def test_normalise_polygon_downsamples_dense_contours():
    dense = [(index, index % 50) for index in range(500)]
    flat = _normalise_polygon(dense, 500, 50)
    assert len(flat) == MAX_POLYGON_POINTS * 2


def test_uncovered_classes_lists_manual_work():
    """COCO 只覆盖 6 个类别，其余必须人工标注。"""
    coco_like = {
        0: "car",
        1: "bus",
        2: "bicycle",
        3: "motorcycle",
        4: "traffic light",
        5: "fire hydrant",
        6: "person",
    }
    missing = uncovered_classes(coco_like)
    assert set(missing) == {
        "Bridge",
        "Chimney",
        "Crosswalk",
        "Mountain",
        "Palm",
        "Stair",
        "Tractor",
    }
    for covered in ("Car", "Bus", "Bicycle", "Motorcycle", "Traffic Light", "Hydrant"):
        assert covered not in missing


def test_collect_images_reads_chinese_category_folders(tmp_path):
    for challenge, category, count in (
        ("multicaptcha", "自行车", 3),
        ("multicaptcha", "公共汽车", 2),
        ("dynamic", "消防栓", 4),
    ):
        directory = tmp_path / challenge / category
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.jpg").write_bytes(b"x")

    items = collect_challenge_images(tmp_path, challenge_types=("multicaptcha",))
    assert len(items) == 5
    targets = {target for _, target in items}
    assert targets == {"Bicycle", "Bus"}

    # dynamic 未被请求时不应混入。
    assert all("dynamic" not in str(path) for path, _ in items)


def test_collect_images_respects_per_class_limit(tmp_path):
    directory = tmp_path / "multicaptcha" / "自行车"
    directory.mkdir(parents=True)
    for index in range(10):
        (directory / f"{index:02d}.jpg").write_bytes(b"x")

    items = collect_challenge_images(tmp_path, limit_per_class=4)
    assert len(items) == 4


def test_report_lists_manual_classes():
    report = PrelabelReport(
        output="out",
        images_total=10,
        images_with_labels=8,
        images_empty=2,
        instances_per_class={"Car": 20, "Bus": 3},
        train_images=8,
        val_images=2,
        uncovered_classes=["Crosswalk", "Bridge"],
    )
    text = format_prelabel_report(report)
    assert "Car" in text
    assert "必须人工标注" in text
    assert "Crosswalk" in text
    assert report.as_dict()["图片总数"] == 10


def test_overwrite_preserves_template_files(tmp_path, monkeypatch):
    """overwrite 只能清理生成目录，不能删掉 README 与 data.yaml.example。"""
    from challenge_images.data import segmentation_prelabel as module

    output = tmp_path / "seg"
    (output / "images" / "train").mkdir(parents=True)
    (output / "labels" / "train").mkdir(parents=True)
    (output / "images" / "train" / "old.jpg").write_bytes(b"stale")
    (output / "README.md").write_text("模板说明", encoding="utf-8")
    (output / "data.yaml.example").write_text("names:\n  0: Bicycle\n", encoding="utf-8")

    class _StubService:
        class_names = {0: "car"}
        device = "cpu"
        model = None

        def load(self, weights, device=None):
            return {}

    monkeypatch.setattr(
        module, "collect_challenge_images", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        "challenge_images.segmentation.model_service.SegmentationModelService",
        _StubService,
    )

    module.build_segmentation_prelabels(tmp_path / "src", output, overwrite=True)

    assert (output / "README.md").read_text(encoding="utf-8") == "模板说明"
    assert (output / "data.yaml.example").is_file()
    # 旧的生成产物被清理。
    assert not (output / "images" / "train" / "old.jpg").exists()


def test_refuses_to_overwrite_without_flag(tmp_path):
    from challenge_images.data.segmentation_prelabel import build_segmentation_prelabels

    output = tmp_path / "seg"
    (output / "labels" / "train").mkdir(parents=True)
    (output / "labels" / "train" / "a.txt").write_text("0 0 0 1 1 0 1", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_segmentation_prelabels(tmp_path / "src", output)
