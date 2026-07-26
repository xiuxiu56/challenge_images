import pytest

from challenge_images.thresholds import Thresholds, load_thresholds


def test_defaults_match_previously_hardcoded_values():
    """配置化不得改变任何既有数值，否则等于悄悄改了识别行为。"""
    t = Thresholds()
    assert t.mask_evidence.min_overlap_pixels == 20
    assert t.mask_evidence.min_cell_ratio == 0.002
    assert t.mask_evidence.min_mask_ratio == 0.10
    assert t.instance_validation.classification_threshold == 0.80
    assert t.instance_validation.confidence_threshold == 0.60
    assert t.instance_validation.recovery_confidence == 0.05
    assert t.instance_validation.strong_classification_rescue == 0.95
    assert t.motorcycle.fringe_min_cell_ratio == 0.01
    assert t.motorcycle.fringe_min_mask_ratio == 0.02
    assert t.motorcycle.horizontal_expansion_cell_ratio == 0.05
    assert t.motorcycle.horizontal_expansion_mask_ratio == 0.03
    assert t.motorcycle.box_fill_cell_ratio == 0.005
    assert t.motorcycle.dominant_min_area_ratio == 0.20
    assert t.bus.major_cell_ratio == 0.05
    assert t.bus.thin_cell_ratio == 0.002
    assert t.traffic_light.fringe_classification_score == 0.25
    assert t.traffic_light.fringe_mask_ratio == 0.02
    assert t.weak_evidence.weak_classification_threshold == 0.10
    assert t.weak_evidence.strong_mask_cell_ratio == 0.01


def test_modules_read_from_config():
    """三个模块的常量必须来自配置，而不是各自再写一份。"""
    from challenge_images.segmentation import result_fusion
    from challenge_images.segmentation.model_service import SEGMENTATION_RECOVERY_CONFIDENCE
    from challenge_images.training.model_service import CANDIDATE_SUPPRESSOR_THRESHOLDS

    t = Thresholds()
    assert result_fusion.BUS_MAJOR_CELL_RATIO == t.bus.major_cell_ratio
    assert result_fusion.MOTORCYCLE_FRINGE_MIN_CELL_RATIO == t.motorcycle.fringe_min_cell_ratio
    assert result_fusion.MIN_OVERLAP_PIXELS == t.mask_evidence.min_overlap_pixels
    assert SEGMENTATION_RECOVERY_CONFIDENCE == t.instance_validation.recovery_confidence
    assert CANDIDATE_SUPPRESSOR_THRESHOLDS[("Crosswalk", "Car")] == 0.40


def test_yaml_override_only_touches_listed_keys(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text("bus:\n  major_cell_ratio: 0.09\n", encoding="utf-8")
    loaded = load_thresholds(path)
    assert loaded.bus.major_cell_ratio == 0.09
    # 同组其他键与其他分组保持默认。
    assert loaded.bus.thin_cell_ratio == 0.002
    assert loaded.motorcycle.fringe_min_cell_ratio == 0.01


def test_missing_file_returns_defaults(tmp_path):
    assert load_thresholds(tmp_path / "absent.yaml") == Thresholds()


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "thresholds.yaml"
    path.write_text("bus:\n  typo_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知的阈值键"):
        load_thresholds(path)

    path.write_text("nonexistent_group:\n  a: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知的阈值分组"):
        load_thresholds(path)


def test_candidate_suppressor_lookup():
    t = Thresholds()
    assert t.candidate_suppressor("Crosswalk", "Car") == 0.40
    assert t.candidate_suppressor("Crosswalk", "Bus") is None


def test_example_template_covers_every_key():
    """模板必须列出全部键，否则用户不知道有哪些可调。"""
    from dataclasses import fields

    from challenge_images.thresholds import dump_default_yaml

    text = dump_default_yaml()
    for section in fields(Thresholds()):
        assert f"{section.name}:" in text
        for item in fields(getattr(Thresholds(), section.name)):
            assert item.name in text
