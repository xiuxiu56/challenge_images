"""大图网格切割、编号绘制和命中图标叠加。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class GridSpec:
    """网格配置。"""

    rows: int
    columns: int

    @property
    def count(self) -> int:
        return self.rows * self.columns

    @property
    def text(self) -> str:
        return f"{self.rows}×{self.columns}"


CHALLENGE_GRID_DEFAULTS = {
    "dynamic": GridSpec(3, 3),
    "imageselect": GridSpec(3, 3),
    "tileselect": GridSpec(3, 3),
    "multicaptcha": GridSpec(4, 4),
}


def grid_for_challenge(challenge_type: str) -> GridSpec:
    """根据挑战类型返回默认网格；未知类型按 3×3 处理。"""
    return CHALLENGE_GRID_DEFAULTS.get(challenge_type.lower(), GridSpec(3, 3))


def resolve_challenge_grid(
    challenge_type: str | None,
    detected: tuple[int, int] | None = None,
) -> GridSpec:
    """统一解析在线挑战网格。

    ``multicaptcha`` 在本项目的大图流程固定按 4×4 处理。
    reload/pmeta 偶尔缺少网格字段，或保留上一轮的 3×3 值，
    因此该类型优先级高于检测值。其他类型优先使用合法
    检测值，最后再使用类型默认值。
    """
    normalized = str(challenge_type or "").strip().lower()
    if normalized == "multicaptcha":
        return GridSpec(4, 4)
    if detected is not None:
        rows, columns = detected
        if rows in (3, 4) and columns in (3, 4):
            return GridSpec(rows, columns)
    return grid_for_challenge(normalized)


def parse_grid(text: str) -> GridSpec:
    """解析 ``3x3``、``3×3``、``4*4`` 等输入。"""
    cleaned = text.lower().replace("×", "x").replace("*", "x").replace(" ", "")
    try:
        rows, columns = (int(x) for x in cleaned.split("x", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"网格格式错误：{text}，示例：3x3") from exc
    if rows not in (3, 4) or columns not in (3, 4):
        raise ValueError("当前只支持 3x3 或 4x4 网格")
    return GridSpec(rows, columns)


def grid_edges(size: int, cells: int) -> list[int]:
    """按比例生成边界，避免不能整除时丢像素。"""
    return [round(index * size / cells) for index in range(cells + 1)]


def grid_index_from_point(
    x: float,
    y: float,
    width: float,
    height: float,
    spec: GridSpec,
) -> int | None:
    """把显示区域内的点转换为网格编号。

    ``x``、``y`` 使用已经显示的图片坐标；图片外的留白返回 ``None``。
    该函数不依赖 GUI 框架，供 PySide6/Tkinter 标注和测试共同使用。
    """
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x >= width or y >= height:
        return None
    column = min(spec.columns - 1, int(x / width * spec.columns))
    row = min(spec.rows - 1, int(y / height * spec.rows))
    return row * spec.columns + column


def split_grid(image: Image.Image, spec: GridSpec) -> list[Image.Image]:
    """按行优先顺序切图，索引与 GUI 显示完全一致。"""
    width, height = image.size
    x_edges = grid_edges(width, spec.columns)
    y_edges = grid_edges(height, spec.rows)
    tiles: list[Image.Image] = []
    for row in range(spec.rows):
        for column in range(spec.columns):
            box = (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
            tiles.append(image.crop(box))
    return tiles


def replace_grid_tile(
    image: Image.Image,
    replacement: Image.Image,
    spec: GridSpec,
    index: int,
) -> Image.Image:
    """把一张动态替换图回填到整图的指定格子。

    reCAPTCHA dynamic 点击后的 ``payload`` 通常只是单格新图，
    不是新的 3×3 整图。本函数保留其他格子，并把新图缩放到
    目标格子的精确像素边界。
    """
    if index < 0 or index >= spec.count:
        raise IndexError(f"格子索引越界：{index}，网格共 {spec.count} 格")

    canvas = image.convert("RGB").copy()
    x_edges = grid_edges(canvas.width, spec.columns)
    y_edges = grid_edges(canvas.height, spec.rows)
    row, column = divmod(index, spec.columns)
    box = (
        x_edges[column],
        y_edges[row],
        x_edges[column + 1],
        y_edges[row + 1],
    )
    cell_size = (box[2] - box[0], box[3] - box[1])
    tile = replacement.convert("RGB")
    if tile.size != cell_size:
        tile = tile.resize(cell_size, Image.Resampling.LANCZOS)
    canvas.paste(tile, box[:2])
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    """尽量使用系统字体；找不到时回退 Pillow 默认字体。"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_grid(
    image: Image.Image,
    spec: GridSpec,
    selected: Iterable[int] = (),
    low_confidence: Iterable[int] = (),
    icon_path: str | Path | None = None,
) -> Image.Image:
    """绘制编号、边界，并把命中图标放在所选格子的正中央。"""
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    selected_set = set(selected)
    low_set = set(low_confidence)
    width, height = canvas.size
    x_edges = grid_edges(width, spec.columns)
    y_edges = grid_edges(height, spec.rows)
    font = _font(max(14, min(width, height) // (spec.rows * 5)))
    icon: Image.Image | None = None
    if icon_path and Path(icon_path).is_file():
        try:
            icon = Image.open(icon_path).convert("RGBA")
        except OSError:
            icon = None
    for row in range(spec.rows):
        for column in range(spec.columns):
            index = row * spec.columns + column
            left, top = x_edges[column], y_edges[row]
            right, bottom = x_edges[column + 1], y_edges[row + 1]
            if index in selected_set:
                color = (40, 190, 80, 255) if index not in low_set else (245, 175, 30, 255)
                draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=max(3, width // 100))
                if icon is not None:
                    cell_width = right - left
                    cell_height = bottom - top
                    # 图标约占格子的 36%，既明显又不会遮住整个目标。
                    icon_size = max(20, round(min(cell_width, cell_height) * 0.36))
                    mark = icon.copy()
                    mark.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
                    icon_left = left + (cell_width - mark.width) // 2
                    icon_top = top + (cell_height - mark.height) // 2
                    canvas.alpha_composite(mark, (icon_left, icon_top))
            else:
                draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 255, 255, 210), width=max(1, width // 300))
            label = str(index)
            bx0, by0, bx1, by1 = draw.textbbox((0, 0), label, font=font)
            tw, th = bx1 - bx0, by1 - by0
            tx, ty = left + 5, top + 4
            draw.rounded_rectangle((tx - 2, ty - 2, tx + tw + 6, ty + th + 4), radius=4, fill=(0, 0, 0, 150))
            draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))
    return canvas
