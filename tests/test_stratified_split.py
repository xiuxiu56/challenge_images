import json

import pytest

from challenge_images.data.stratified_split import (
    build_stratified_dataset,
    collect_class_files,
    format_split_report,
    plan_val_count,
)


def _write_images(root, split, class_name, count, *, start=0, content=None):
    directory = root / split / class_name
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(start, start + count):
        payload = content if content is not None else f"{class_name}-{index}".encode()
        (directory / f"{index:04d}.jpg").write_bytes(payload)


def test_plan_val_count_respects_floor_and_ceiling():
    # 大类：按比例取，但不超过上限。
    assert plan_val_count(10000, val_ratio=0.15, val_min=50, val_max=300, max_val_fraction=1 / 3) == 300
    # 中类：比例结果高于下限时按比例。
    assert plan_val_count(1000, val_ratio=0.15, val_min=50, val_max=300, max_val_fraction=1 / 3) == 150
    # 小类：比例结果低于下限时抬到下限。
    assert plan_val_count(200, val_ratio=0.05, val_min=50, val_max=300, max_val_fraction=1 / 3) == 50


def test_rare_class_never_loses_most_of_its_training_data():
    """Tractor 只有 26 张，下限 50 不能把训练样本全部划走。"""
    count = plan_val_count(26, val_ratio=0.15, val_min=50, val_max=300, max_val_fraction=1 / 3)
    assert count == 8
    assert count < 26 / 2


def test_single_sample_class_yields_no_validation():
    assert plan_val_count(1, val_ratio=0.15, val_min=50, val_max=300, max_val_fraction=1 / 3) == 0


def test_collect_merges_train_and_val_pools(tmp_path):
    _write_images(tmp_path, "train", "Car", 5)
    _write_images(tmp_path, "val", "Car", 3, start=100)
    _write_images(tmp_path, "train", "Tractor", 2)

    pool = collect_class_files(tmp_path)
    assert len(pool["Car"]) == 8
    assert len(pool["Tractor"]) == 2


def test_build_stratified_dataset_balances_validation(tmp_path):
    source = tmp_path / "source"
    _write_images(source, "train", "Car", 400)
    _write_images(source, "val", "Car", 40, start=1000)
    _write_images(source, "train", "Tractor", 30)
    _write_images(source, "val", "Tractor", 1, start=1000)

    output = tmp_path / "out"
    report = build_stratified_dataset(
        source, output, val_ratio=0.15, val_min=50, val_max=300
    )

    car = report.classes["Car"]
    tractor = report.classes["Tractor"]
    assert car.total == 440
    assert car.val == 66
    # 稀有类此前只有 1 张验证样本，重划后指标才有意义。
    assert tractor.total == 31
    assert tractor.val == 10
    assert car.train + car.val == car.total
    assert tractor.train + tractor.val == tractor.total

    # 落盘结构与数量一致。
    assert len(list((output / "val" / "Tractor").glob("*.jpg"))) == 10
    assert len(list((output / "train" / "Tractor").glob("*.jpg"))) == 21
    saved = json.loads((output / "split_report.json").read_text(encoding="utf-8"))
    assert saved["类别"]["Tractor"]["验证"] == 10


def test_duplicate_images_are_removed(tmp_path):
    source = tmp_path / "source"
    # 20 张完全相同的图片只应保留 1 张。
    _write_images(source, "train", "Car", 20, content=b"identical")
    _write_images(source, "train", "Car", 10, start=500)

    report = build_stratified_dataset(source, tmp_path / "out", val_min=2, val_max=5)
    car = report.classes["Car"]
    assert car.duplicates_removed == 19
    assert car.total == 11


def test_split_is_deterministic(tmp_path):
    source = tmp_path / "source"
    _write_images(source, "train", "Car", 50)

    first = build_stratified_dataset(source, tmp_path / "a", val_min=5, val_max=10)
    second = build_stratified_dataset(source, tmp_path / "b", val_min=5, val_max=10)

    def val_targets(root):
        return sorted(path.resolve().name for path in (root / "val" / "Car").iterdir())

    assert val_targets(tmp_path / "a") == val_targets(tmp_path / "b")
    assert first.classes["Car"].val == second.classes["Car"].val


def test_existing_output_requires_overwrite(tmp_path):
    source = tmp_path / "source"
    _write_images(source, "train", "Car", 10)
    output = tmp_path / "out"
    build_stratified_dataset(source, output, val_min=2, val_max=3)

    with pytest.raises(FileExistsError):
        build_stratified_dataset(source, output, val_min=2, val_max=3)

    report = build_stratified_dataset(source, output, val_min=2, val_max=3, overwrite=True)
    assert report.classes["Car"].total == 10


def test_report_highlights_weakest_class(tmp_path):
    source = tmp_path / "source"
    _write_images(source, "train", "Car", 300)
    _write_images(source, "train", "Tractor", 6)

    report = build_stratified_dataset(source, tmp_path / "out", val_min=50, val_max=100)
    assert report.smallest_val_class == ("Tractor", 2)
    text = format_split_report(report)
    assert "Tractor" in text
    assert "验证样本最少的类别" in text


def test_source_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_stratified_dataset(tmp_path / "missing", tmp_path / "out")


def test_multi_label_conflicts_are_excluded_from_validation(tmp_path):
    """同一张图被标注为两个类别时不能进验证集：预测哪个都会被判错。"""
    source = tmp_path / "source"
    shared = b"bus-and-crosswalk-tile"
    _write_images(source, "train", "Bus", 30)
    _write_images(source, "train", "Crosswalk", 30)
    # 同一张复合图块同时出现在两个类别下。
    (source / "train" / "Bus" / "shared.jpg").write_bytes(shared)
    (source / "train" / "Crosswalk" / "shared.jpg").write_bytes(shared)

    output = tmp_path / "out"
    report = build_stratified_dataset(source, output, val_min=5, val_max=10)

    assert len(report.multi_label_conflicts) == 1
    conflict_classes = next(iter(report.multi_label_conflicts.values()))
    assert conflict_classes == ["Bus", "Crosswalk"]

    # 冲突图保留在训练集，未进入验证集。
    for class_name in ("Bus", "Crosswalk"):
        assert report.classes[class_name].multi_label_held_out == 1
        val_bytes = {p.resolve().read_bytes() for p in (output / "val" / class_name).iterdir()}
        train_bytes = {p.resolve().read_bytes() for p in (output / "train" / class_name).iterdir()}
        assert shared not in val_bytes
        assert shared in train_bytes


def test_multi_label_seeds_are_exported(tmp_path):
    source = tmp_path / "source"
    shared = b"composite"
    _write_images(source, "train", "Bus", 10)
    _write_images(source, "train", "Crosswalk", 10)
    (source / "train" / "Bus" / "s.jpg").write_bytes(shared)
    (source / "train" / "Crosswalk" / "s.jpg").write_bytes(shared)

    output = tmp_path / "out"
    build_stratified_dataset(source, output, val_min=2, val_max=4)
    seeds = json.loads((output / "multi_label_seeds.json").read_text(encoding="utf-8"))
    assert seeds["数量"] == 1
    assert list(seeds["冲突"].values())[0] == ["Bus", "Crosswalk"]


def test_no_train_val_leakage_after_split(tmp_path):
    """重划分后同一张图不得同时出现在 train 与 val。"""
    import hashlib

    source = tmp_path / "source"
    # 模拟原始数据集的问题：val 里放的就是 train 里的副本。
    _write_images(source, "train", "Chimney", 40)
    _write_images(source, "val", "Chimney", 40)

    output = tmp_path / "out"
    build_stratified_dataset(source, output, val_min=5, val_max=10)

    def digests(split):
        return {
            hashlib.sha256(p.resolve().read_bytes()).hexdigest()
            for p in (output / split / "Chimney").iterdir()
        }

    assert digests("train") & digests("val") == set()
