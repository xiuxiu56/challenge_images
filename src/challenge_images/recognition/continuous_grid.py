"""4×4 连续照片的格子后处理。

4×4 挑战是一张 450×450 的连续照片被切成 16 块，因此目标在网格上必然是
**连通**的：一辆公交车不可能在格子 0 和格子 15 各出现一半而中间没有。
逐格分类看不到这个约束，孤立的单格命中往往是误检。

但「只保留最大连通域」是错的：画面里可以有两棵相隔的棕榈树、
两个不相邻的红绿灯、停在不同位置的两辆车。因此这里采用的规则是

    孤立单格需要比成片格子更强的分类证据

有证据时按证据判断，没有证据时保持原样——绝不在缺少信息的情况下删格子。

人行横道保留原有的专用逻辑：它在连续照片里形成同一行的连续带，
这个形态约束比通用连通性更强，且已用真实样本调校过。
"""

from __future__ import annotations

from ..category_map import normalize_dataset_class
from ..grid.grid_engine import GridSpec
from ..thresholds import THRESHOLDS


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


def connected_components(indices: set[int], spec: GridSpec) -> list[set[int]]:
    """按四邻接把选中的格子分成连通块。

    使用四邻接而非八邻接：只在对角相接的两个格子，目标未必真的连续，
    更可能是两处独立的检出。
    """
    remaining = set(indices)
    components: list[set[int]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            row, column = divmod(current, spec.columns)
            neighbours = []
            if row > 0:
                neighbours.append(current - spec.columns)
            if row + 1 < spec.rows:
                neighbours.append(current + spec.columns)
            if column > 0:
                neighbours.append(current - 1)
            if column + 1 < spec.columns:
                neighbours.append(current + 1)
            for neighbour in neighbours:
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _refine_crosswalk(spec: GridSpec, selected: set[int]) -> list[int]:
    """人行横道专用：保留主体连续带并去掉孤立误格。

    模型对贴左右边缘的两格短段容易少报相邻一格，因此只对这种明确形态
    向图片内部补一格；其他行必须至少连续命中两格并与主带列区间相交。
    """
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
        main_column_list = sorted(index % spec.columns for index in main)
        if main_column_list == [0, 1]:
            refined.add(row * spec.columns + 2)
        elif main_column_list == [2, 3]:
            refined.add(row * spec.columns + 1)
    return sorted(refined)


def _drop_weak_isolated(
    selected: set[int],
    spec: GridSpec,
    scores: dict[int, float],
    minimum_score: float,
) -> list[int]:
    """删掉证据不足的孤立单格。

    只处理大小为 1 的连通块，且必须同时存在更大的连通块——否则整张图
    可能本来就只有一个小目标，删掉会直接漏选。
    """
    components = connected_components(selected, spec)
    if len(components) <= 1:
        return sorted(selected)
    if not any(len(component) > 1 for component in components):
        return sorted(selected)

    refined: set[int] = set()
    for component in components:
        if len(component) > 1:
            refined.update(component)
            continue
        index = next(iter(component))
        # 缺少该格的分类证据时保守保留，不在无信息的情况下删格子。
        if scores.get(index, 1.0) >= minimum_score:
            refined.add(index)
    return sorted(refined)


def refine_continuous_grid(
    target_class: str,
    spec: GridSpec,
    indices: list[int],
    *,
    scores: dict[int, float] | None = None,
    isolated_min_score: float | None = None,
) -> list[int]:
    """对 4×4 连续照片的选择结果做形态学收敛。

    ``scores`` 是每个格子对目标类别的分类概率。缺省时通用连通性规则
    不生效——没有证据就不删格子。
    """
    selected = {
        int(index) for index in indices if 0 <= int(index) < spec.count
    }
    if spec.rows != 4 or spec.columns != 4 or not selected:
        return sorted(selected)

    if normalize_dataset_class(target_class) == "Crosswalk":
        return _refine_crosswalk(spec, selected)

    if not scores:
        return sorted(selected)
    threshold = (
        float(isolated_min_score)
        if isolated_min_score is not None
        else THRESHOLDS.continuous_grid.isolated_min_score
    )
    return _drop_weak_isolated(selected, spec, scores, threshold)
