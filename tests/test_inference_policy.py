from challenge_images.config import DEFAULT_TRAIN_IMGSZ
from challenge_images.training.inference_policy import (
    CATEGORY_PROFILES,
    CONTINUOUS_GROUPS,
    DEFAULT_PROFILE,
    challenge_family,
    profile_for,
)


def test_crosswalk_uses_multiview_profile():
    profile = profile_for("/m/014xcs")
    assert profile.allow_multiview is True
    assert profile.top1_threshold == 0.60
    assert profile.local_threshold == 0.65


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


def test_category_profiles_no_longer_pin_per_class_resolution():
    """推理分辨率跟随权重元数据，类别配置只保留统一兜底值。"""
    assert DEFAULT_PROFILE.imgsz == DEFAULT_TRAIN_IMGSZ
    assert {profile.imgsz for profile in CATEGORY_PROFILES.values()} == {DEFAULT_TRAIN_IMGSZ}


# ---------- 挑战类型维度 ----------


def test_challenge_family_splits_3x3_and_4x4():
    for challenge in ("dynamic", "imageselect"):
        assert challenge_family(challenge) == "grid3"
    # tileselect 实测 450×450 且 pmeta 写 4,4，属于 4×4 家族。
    for challenge in ("multicaptcha", "tileselect"):
        assert challenge_family(challenge) == "grid4"
    # 未知或缺失类型按 3×3 处理。
    assert challenge_family(None) == "grid3"
    assert challenge_family("unknown") == "grid3"


def test_3x3_challenges_keep_category_baseline():
    """3×3 家族不改动经过调校的类别基线。"""
    for challenge in ("dynamic", "imageselect"):
        assert profile_for("Car", challenge) == CATEGORY_PROFILES["Car"]
        assert profile_for("Crosswalk", challenge) == CATEGORY_PROFILES["Crosswalk"]


def test_4x4_relaxes_strict_top1_vehicles_to_recall_edge_tiles():
    """Car/Bus 在 4×4 下必须能从 Top-2/Top-3 召回跨格边缘格。"""
    for target in ("Car", "Bus"):
        grid3 = profile_for(target, "dynamic")
        grid4 = profile_for(target, "multicaptcha")
        # 3×3 只认 Top-1；4×4 若沿用会漏掉全部边缘格。
        assert grid3.candidate_threshold == 1.0
        assert grid3.top_k == 1
        assert grid4.candidate_threshold == 0.35
        assert grid4.top_k == 3
        # 主体格仍要求同样的高置信度，不牺牲精度。
        assert grid4.top1_threshold == grid3.top1_threshold


def test_4x4_small_objects_lower_top1():
    grid3 = profile_for("Hydrant", "dynamic")
    grid4 = profile_for("Hydrant", "multicaptcha")
    assert grid4.top1_threshold < grid3.top1_threshold
    assert grid4.candidate_threshold < grid3.candidate_threshold


def test_4x4_scene_classes_do_not_widen_candidates():
    """山丘/棕榈树几乎每格都有弱证据，放宽会导致整屏全选。"""
    grid3 = profile_for("Mountain", "dynamic")
    grid4 = profile_for("Mountain", "multicaptcha")
    assert grid4.candidate_threshold == grid3.candidate_threshold
    assert grid4.top_k >= 3


def test_continuous_groups_cover_every_dataset_class():
    """除负样本 Other 外，每个数据集类别都要有 4×4 分组。"""
    from challenge_images.category_map import DATASET_CLASS_TO_MID

    expected = set(DATASET_CLASS_TO_MID) - {"Other"}
    assert expected <= set(CONTINUOUS_GROUPS)


def test_thresholds_stay_within_valid_range():
    """任何组合都不得产生越界阈值。"""
    from challenge_images.category_map import DATASET_CLASS_TO_MID

    for target in DATASET_CLASS_TO_MID:
        for challenge in ("dynamic", "imageselect", "tileselect", "multicaptcha"):
            profile = profile_for(target, challenge)
            assert 0.0 <= profile.top1_threshold <= 1.0
            assert 0.0 <= profile.candidate_threshold <= 1.0
            assert profile.top_k >= 1


def test_4x4_never_tightens_candidate_threshold():
    """4×4 调整层只允许放宽候选证据。

    Car/Bus 基线为 1.0，Motorcycle/Tractor 同组但基线是 0.25。
    若把上限当成绝对值写入，后者会被从 0.25 收紧到 0.35 而漏掉边缘格。
    """
    from challenge_images.category_map import DATASET_CLASS_TO_MID

    for target in DATASET_CLASS_TO_MID:
        grid3 = profile_for(target, "dynamic")
        grid4 = profile_for(target, "multicaptcha")
        assert grid4.candidate_threshold <= grid3.candidate_threshold, target
        assert grid4.top1_threshold <= grid3.top1_threshold, target
        assert grid4.top_k >= grid3.top_k, target


def test_motorcycle_keeps_loose_baseline_in_4x4():
    """与 Car/Bus 同组但基线更松的类别不应被上限反向收紧。"""
    assert profile_for("Motorcycle", "dynamic").candidate_threshold == 0.25
    assert profile_for("Motorcycle", "multicaptcha").candidate_threshold == 0.25
