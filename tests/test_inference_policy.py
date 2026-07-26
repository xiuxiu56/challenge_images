from challenge_images.config import DEFAULT_TRAIN_IMGSZ
from challenge_images.training.inference_policy import CATEGORY_PROFILES, DEFAULT_PROFILE, profile_for


def test_crosswalk_uses_multiview_profile():
    profile = profile_for("/m/014xcs")
    assert profile.allow_multiview is True
    assert profile.top1_threshold == 0.60
    assert profile.local_threshold == 0.65


def test_category_profiles_no_longer_pin_per_class_resolution():
    """推理分辨率跟随权重元数据，类别配置只保留统一兜底值。"""
    assert DEFAULT_PROFILE.imgsz == DEFAULT_TRAIN_IMGSZ
    assert {profile.imgsz for profile in CATEGORY_PROFILES.values()} == {DEFAULT_TRAIN_IMGSZ}


def test_car_only_accepts_top1():
    profile = profile_for("车")
    assert profile.top_k == 1
    assert profile.top1_threshold == 0.85


def test_hydrant_uses_small_object_threshold():
    profile = profile_for("消防栓")
    assert profile.top1_threshold == 0.50
    assert profile.candidate_threshold == 0.20


def test_bus_only_accepts_top1():
    profile = profile_for("Bus")
    assert profile.top_k == 1
    assert profile.allow_multiview is False
