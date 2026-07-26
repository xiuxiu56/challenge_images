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
