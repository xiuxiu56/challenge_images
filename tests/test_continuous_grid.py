"""4×4 连续照片格子后处理测试。"""

from challenge_images.grid.grid_engine import GridSpec
from challenge_images.recognition.continuous_grid import refine_continuous_grid


def test_crosswalk_keeps_main_strip_expands_inward_and_drops_isolated_cell():
    result = refine_continuous_grid(
        "Crosswalk",
        GridSpec(4, 4),
        [8, 9, 15],
    )

    assert result == [8, 9, 10]


def test_crosswalk_keeps_three_cell_main_strip_and_drops_other_isolated_cells():
    result = refine_continuous_grid(
        "Crosswalk",
        GridSpec(4, 4),
        [4, 5, 6, 8, 10],
    )

    assert result == [4, 5, 6]


def test_crosswalk_keeps_an_overlapping_secondary_continuous_strip():
    result = refine_continuous_grid(
        "Crosswalk",
        GridSpec(4, 4),
        [4, 5, 8, 9],
    )

    assert result == [4, 5, 6, 8, 9]


def test_continuous_refinement_does_not_change_other_classes_or_three_by_three():
    assert refine_continuous_grid("Bus", GridSpec(4, 4), [5, 8]) == [5, 8]
    assert refine_continuous_grid("Crosswalk", GridSpec(3, 3), [3, 8]) == [3, 8]


# ---------- 通用连通性约束 ----------

from challenge_images.recognition.continuous_grid import connected_components  # noqa: E402

GRID4 = GridSpec(4, 4)


def test_connected_components_uses_four_connectivity():
    """只在对角相接的格子不算连通：目标未必真的连续。"""
    # 0 与 5 是对角关系
    assert len(connected_components({0, 5}, GRID4)) == 2
    # 0 与 1 水平相邻
    assert len(connected_components({0, 1}, GRID4)) == 1
    # 0 与 4 垂直相邻
    assert len(connected_components({0, 4}, GRID4)) == 1


def test_connected_components_does_not_wrap_rows():
    """格子 3 在第 0 行末尾，格子 4 在第 1 行开头，不相邻。"""
    assert len(connected_components({3, 4}, GRID4)) == 2


def test_isolated_low_score_tile_is_dropped():
    """成片格子有连通性背书，孤立格证据不足即判为误检。"""
    indices = [5, 6, 9, 10, 15]
    scores = {5: 0.9, 6: 0.9, 9: 0.9, 10: 0.9, 15: 0.3}
    assert refine_continuous_grid("Bus", GRID4, indices, scores=scores) == [5, 6, 9, 10]


def test_isolated_high_score_tile_is_kept():
    """画面里可以有两辆车、两棵树；证据足够就保留。"""
    indices = [5, 6, 9, 10, 15]
    scores = {5: 0.9, 6: 0.9, 9: 0.9, 10: 0.9, 15: 0.95}
    assert refine_continuous_grid("Bus", GRID4, indices, scores=scores) == [5, 6, 9, 10, 15]


def test_single_small_target_is_never_dropped():
    """整图只有一个孤立目标时不做任何删减，否则会直接漏选。"""
    assert refine_continuous_grid("Hydrant", GRID4, [7], scores={7: 0.2}) == [7]


def test_all_isolated_tiles_are_kept():
    """全部都是孤立格时无从判断哪个是主体，保守全留。"""
    indices = [0, 2, 8, 10]
    scores = dict.fromkeys(indices, 0.1)
    assert refine_continuous_grid("Palm", GRID4, indices, scores=scores) == indices


def test_without_scores_nothing_is_removed():
    """没有证据就不删格子。"""
    indices = [5, 6, 15]
    assert refine_continuous_grid("Bus", GRID4, indices) == indices


def test_missing_score_defaults_to_keep():
    """个别格子缺少概率时保守保留。"""
    indices = [5, 6, 15]
    assert refine_continuous_grid("Bus", GRID4, indices, scores={5: 0.9, 6: 0.9}) == indices


def test_threshold_is_configurable():
    indices = [5, 6, 15]
    scores = {5: 0.9, 6: 0.9, 15: 0.75}
    assert refine_continuous_grid("Bus", GRID4, indices, scores=scores) == [5, 6, 15]
    assert refine_continuous_grid(
        "Bus", GRID4, indices, scores=scores, isolated_min_score=0.85
    ) == [5, 6]


def test_3x3_is_untouched():
    """3×3 每格是独立照片，连通性没有意义。"""
    indices = [0, 4, 8]
    scores = dict.fromkeys(indices, 0.1)
    assert refine_continuous_grid("Bus", GridSpec(3, 3), indices, scores=scores) == indices


def test_crosswalk_keeps_specialised_logic():
    """人行横道走已调校的连续带规则，不受通用连通性影响。"""
    # 贴左边缘的两格短段会向内补一格——这是通用规则不会做的。
    assert refine_continuous_grid("Crosswalk", GRID4, [4, 5]) == [4, 5, 6]
