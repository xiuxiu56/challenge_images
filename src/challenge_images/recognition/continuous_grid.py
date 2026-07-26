"""4×4 连续照片的类别专用格子后处理。"""

from __future__ import annotations

from ..category_map import normalize_dataset_class
from ..grid.grid_engine import GridSpec


def _row_segments(indices: set[int], columns: int) -> list[list[int]]:
    """把格子索引拆成同一行内的连续片段。"""
    segments: list[list[int]] = []
    for row in range(columns):
        row_indices = sorted(
            index for index in indices if index // columns == row
        )
        current: list[int] = []
        for index in row_indices:
            if current and index != current[-1] + 1:
                segments.append(current)
                current = []
            current.append(index)
        if current:
            segments.append(current)
    return segments


def refine_continuous_grid(
    target_class: str,
    spec: GridSpec,
    indices: list[int],
) -> list[int]:
    """收敛 4×4 人行横道结果，保留主体连续区域并去掉孤立误格。

    人行横道在连续照片里通常形成同一行的连续带。模型对贴左右边缘的
    两格短段容易少报相邻一格，因此只对这种明确形态向图片内部补一格；
    其他行必须至少连续命中两格并与主带列区间相交才会保留。
    """
    selected = {
        int(index) for index in indices if 0 <= int(index) < spec.count
    }
    if (
        normalize_dataset_class(target_class) != "Crosswalk"
        or spec.rows != 4
        or spec.columns != 4
        or not selected
    ):
        return sorted(selected)

    segments = _row_segments(selected, spec.columns)
    if not segments:
        return []
    main = min(
        segments,
        key=lambda segment: (-len(segment), segment[0] // spec.columns, segment[0]),
    )
    main_columns = {index % spec.columns for index in main}
    refined = set(main)
    for segment in segments:
        if segment == main or len(segment) < 2:
            continue
        columns = {index % spec.columns for index in segment}
        if columns & main_columns:
            refined.update(segment)

    if len(main) == 2:
        row = main[0] // spec.columns
        columns = sorted(index % spec.columns for index in main)
        if columns == [0, 1]:
            refined.add(row * spec.columns + 2)
        elif columns == [2, 3]:
            refined.add(row * spec.columns + 1)
    return sorted(refined)
