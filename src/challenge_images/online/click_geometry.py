"""reCAPTCHA 挑战图块点击坐标计算。

实测来源（chrome-devtools MCP，2026-07-22，api2/demo）：
- 主页面有 anchor iframe（复选框）与 bframe iframe（挑战窗）
- 点击复选框后 bframe 可见尺寸约 400×580
- 图块区近似正方形，位于指令区下方、验证按钮上方
- 图块索引：行优先，左上角为 0

优先策略：
1. 在 bframe 内用 DOM 查询 `.rc-imageselect-tile` 直接点元素
2. DOM 不可用时，按 bframe 外框 + 布局常量估算格子中心
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class TileLayout:
    """图块表在页面坐标系中的布局。"""

    table: Rect
    rows: int
    columns: int
    cells: tuple[Rect, ...]

    def cell(self, index: int) -> Rect:
        if index < 0 or index >= len(self.cells):
            raise IndexError(f"格子编号越界：{index}，总数={len(self.cells)}")
        return self.cells[index]


# MCP 实测默认布局常量（bframe 本地坐标）
DEFAULT_HEADER_PX = 110.0
DEFAULT_FOOTER_PX = 70.0
DEFAULT_PAD_X = 12.0
DEFAULT_PAD_Y = 10.0


def estimate_table_rect(
    bframe: Rect,
    *,
    header_px: float = DEFAULT_HEADER_PX,
    footer_px: float = DEFAULT_FOOTER_PX,
    pad_x: float = DEFAULT_PAD_X,
    pad_y: float = DEFAULT_PAD_Y,
) -> Rect:
    """根据 bframe 外框估算图块表区域。"""
    table_left = bframe.x + pad_x
    table_top = bframe.y + header_px + pad_y
    table_w = max(1.0, bframe.width - pad_x * 2.0)
    available_h = max(1.0, bframe.height - header_px - footer_px - pad_y * 2.0)
    # 图块区接近正方形
    table_h = min(table_w, available_h)
    return Rect(table_left, table_top, table_w, table_h)


def build_tile_layout(
    bframe: Rect,
    rows: int,
    columns: int,
    *,
    header_px: float = DEFAULT_HEADER_PX,
    footer_px: float = DEFAULT_FOOTER_PX,
    pad_x: float = DEFAULT_PAD_X,
    pad_y: float = DEFAULT_PAD_Y,
) -> TileLayout:
    """把 bframe 切成 rows×columns 格子，返回页面坐标。"""
    if rows <= 0 or columns <= 0:
        raise ValueError(f"网格尺寸无效：{rows}x{columns}")
    table = estimate_table_rect(
        bframe,
        header_px=header_px,
        footer_px=footer_px,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    cell_w = table.width / columns
    cell_h = table.height / rows
    cells: list[Rect] = []
    for row in range(rows):
        for col in range(columns):
            cells.append(
                Rect(
                    table.x + col * cell_w,
                    table.y + row * cell_h,
                    cell_w,
                    cell_h,
                )
            )
    return TileLayout(table=table, rows=rows, columns=columns, cells=tuple(cells))


def cell_center(layout: TileLayout, index: int) -> tuple[float, float]:
    """返回指定格子中心点（页面 CSS 像素）。"""
    return layout.cell(index).center


# pmeta 分类行中行列所在的下标。实测（8626 条在线记录）：
#   dynamic      ["/m/09d_r", null, 3, 3, 3, null, "Mountain"]  → [3],[4] = 3,3
#   tileselect   ["/m/0199g", null, 2, 4, 4]                    → [3],[4] = 4,4
#   multicaptcha ["/m/0199g", null, 2, 4, 4]                    → [3],[4] = 4,4
# 下标 [3],[4] 在全部题型上命中率 100%；旧实现读的 [2],[3] 只在 3×3 题型
# 偶然命中（因为该位置恰好也是 3），对全部 4×4 挑战都会返回 None。
PMETA_GRID_INDICES = (3, 4)
# 兼容可能存在的旧格式，作为兜底再试一次。
PMETA_GRID_FALLBACK_INDICES = (2, 3)


def grid_from_pmeta(pmeta: object) -> tuple[int, int] | None:
    """从 pmeta 分类行读取网格尺寸。

    深度优先寻找第一个以 ``/m/`` 开头的分类行，再按已验证的下标取行列。
    """
    if not isinstance(pmeta, list) or not pmeta:
        return None
    rows_cols: tuple[int, int] | None = None

    def read(node: list, indices: tuple[int, int]) -> tuple[int, int] | None:
        row_index, column_index = indices
        if len(node) <= column_index:
            return None
        try:
            rows, columns = int(node[row_index]), int(node[column_index])
        except (TypeError, ValueError):
            return None
        return (rows, columns) if rows in (3, 4) and columns in (3, 4) else None

    def visit(node: object) -> None:
        nonlocal rows_cols
        if rows_cols is not None:
            return
        if isinstance(node, list):
            if node and isinstance(node[0], str) and node[0].startswith("/m/"):
                found = read(node, PMETA_GRID_INDICES) or read(node, PMETA_GRID_FALLBACK_INDICES)
                if found is not None:
                    rows_cols = found
                    return
            for child in node:
                visit(child)

    visit(pmeta)
    return rows_cols
