import pytest

from challenge_images.data.class_weights import (
    DEFAULT_MAX_WEIGHT,
    SMOOTHING_LINEAR,
    SMOOTHING_NONE,
    SMOOTHING_SQRT,
    compute_class_weights,
    compute_positive_weights,
    count_images_per_class,
    format_balance_report,
)


COUNTS = {"Car": 10000, "Motorcycle": 1000, "Tractor": 20}


def test_rare_classes_get_larger_weight():
    balance = compute_class_weights(COUNTS)
    assert balance.weights["Tractor"] > balance.weights["Motorcycle"] > balance.weights["Car"]


def test_median_class_weight_is_about_one():
    """以中位数为基准，多数类权重接近 1，损失量级不随类别数漂移。"""
    balance = compute_class_weights(COUNTS)
    assert balance.weights["Motorcycle"] == pytest.approx(1.0, abs=0.01)


def test_weights_are_capped():
    """Tractor 原始权重 500，不截断会让单个样本主导整批梯度。"""
    balance = compute_class_weights({"Car": 10000, "Tractor": 20}, max_weight=5.0)
    assert max(balance.weights.values()) <= 5.0


def test_sqrt_smoothing_compresses_extremes():
    linear = compute_class_weights(COUNTS, smoothing=SMOOTHING_LINEAR, max_weight=1e6)
    sqrt = compute_class_weights(COUNTS, smoothing=SMOOTHING_SQRT, max_weight=1e6)
    assert sqrt.weights["Tractor"] < linear.weights["Tractor"]
    # sqrt 后仍保持排序关系。
    assert sqrt.weights["Tractor"] > sqrt.weights["Car"]


def test_smoothing_none_disables_weighting():
    balance = compute_class_weights(COUNTS, smoothing=SMOOTHING_NONE)
    assert set(balance.weights.values()) == {1.0}


def test_imbalance_ratio_reported():
    balance = compute_class_weights(COUNTS)
    assert balance.imbalance_ratio == pytest.approx(500.0)


def test_empty_classes_get_neutral_weight():
    balance = compute_class_weights({"Car": 100, "Boat": 0})
    assert balance.weights["Boat"] == 1.0


def test_weight_vector_follows_class_order():
    """权重顺序必须与模型 names 顺序一致，错位会把权重加到别的类别上。"""
    balance = compute_class_weights(COUNTS, classes=["Tractor", "Car", "Motorcycle"])
    vector = balance.weight_vector()
    assert vector[0] == balance.weights["Tractor"]
    assert vector[1] == balance.weights["Car"]
    assert len(vector) == 3


# ---------- 多标签 pos_weight ----------


def test_pos_weight_is_negative_over_positive_ratio():
    """pos_weight 定义为负样本数 / 正样本数。"""
    balance = compute_positive_weights(
        {"Car": 100}, total_samples=1000, smoothing=SMOOTHING_LINEAR, max_weight=1e6
    )
    assert balance.weights["Car"] == pytest.approx(9.0)


def test_pos_weight_never_below_one():
    """多数类不应被降权到 1 以下，否则等于抑制正样本。"""
    balance = compute_positive_weights({"Car": 900}, total_samples=1000)
    assert balance.weights["Car"] >= 1.0


def test_pos_weight_rare_class_hits_cap():
    balance = compute_positive_weights(
        {"Car": 9894, "Tractor": 18}, total_samples=44499, max_weight=DEFAULT_MAX_WEIGHT
    )
    assert balance.weights["Tractor"] == DEFAULT_MAX_WEIGHT
    assert balance.weights["Car"] < 3.0


def test_pos_weight_handles_absent_class():
    balance = compute_positive_weights({"Boat": 0}, total_samples=100)
    assert balance.weights["Boat"] == 1.0


# ---------- 统计与报告 ----------


def test_count_images_per_class(tmp_path):
    for name, count in (("Car", 3), ("Tractor", 1)):
        directory = tmp_path / "train" / name
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.jpg").write_bytes(b"x")
    # 非图片文件与隐藏文件不计入。
    (tmp_path / "train" / "Car" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "train" / "Car" / ".DS_Store").write_bytes(b"x")

    counts = count_images_per_class(tmp_path)
    assert counts == {"Car": 3, "Tractor": 1}


def test_count_images_requires_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_images_per_class(tmp_path, "train")


def test_report_lists_every_class():
    text = format_balance_report(compute_class_weights(COUNTS))
    for name in COUNTS:
        assert name in text
    assert "不均衡倍数" in text
