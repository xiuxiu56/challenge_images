"""把分割 mask 转换为3×3/4×4格子证据，并生成可视化叠加图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from ..grid.grid_engine import GridSpec, grid_edges
from ..thresholds import THRESHOLDS


@dataclass(frozen=True)
class CellMaskEvidence:
    """一个 mask 在一个格子内的覆盖证据。"""

    index: int
    overlap_pixels: int
    cell_ratio: float
    mask_ratio: float
    selected: bool
    touches_left: bool = False
    touches_right: bool = False
    touches_top: bool = False
    touches_bottom: bool = False


def normalize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把模型 mask 归一化到原图宽高，返回0～1浮点数组。"""
    array = np.asarray(mask, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"分割 mask 维度错误：{array.shape}")
    width, height = size
    if array.shape != (height, width):
        source = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
        source = source.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(source, dtype=np.float32) / 255.0
    return array


def mask_grid_evidence(
    mask: np.ndarray,
    spec: GridSpec,
    *,
    min_overlap_pixels: int = THRESHOLDS.mask_evidence.min_overlap_pixels,
    min_cell_ratio: float = THRESHOLDS.mask_evidence.min_cell_ratio,
    min_mask_ratio: float = THRESHOLDS.mask_evidence.min_mask_ratio,
) -> list[CellMaskEvidence]:
    """计算 mask 对每个格子的覆盖比例和最终命中状态。"""
    binary = np.asarray(mask) >= 0.5
    height, width = binary.shape
    total_mask_pixels = max(int(binary.sum()), 1)
    x_edges = grid_edges(width, spec.columns)
    y_edges = grid_edges(height, spec.rows)
    evidence: list[CellMaskEvidence] = []
    for row in range(spec.rows):
        for column in range(spec.columns):
            index = row * spec.columns + column
            cell = binary[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            overlap_pixels = int(cell.sum())
            cell_ratio = overlap_pixels / max(int(cell.size), 1)
            mask_ratio = overlap_pixels / total_mask_pixels
            selected = overlap_pixels >= int(min_overlap_pixels) and (
                cell_ratio >= float(min_cell_ratio)
                and mask_ratio >= float(min_mask_ratio)
            )
            edge_band = max(1, min(cell.shape) // 40)
            evidence.append(
                CellMaskEvidence(
                    index=index,
                    overlap_pixels=overlap_pixels,
                    cell_ratio=cell_ratio,
                    mask_ratio=mask_ratio,
                    selected=selected,
                    touches_left=bool(cell[:, :edge_band].any()),
                    touches_right=bool(cell[:, -edge_band:].any()),
                    touches_top=bool(cell[:edge_band, :].any()),
                    touches_bottom=bool(cell[-edge_band:, :].any()),
                )
            )
    return evidence


def mask_bottom_extensions(
    mask: np.ndarray,
    spec: GridSpec,
    *,
    minimum_band_pixels: int = 12,
) -> list[int]:
    """找出紧贴水平格线、但被分割边缘截断的下一行格子。

    大型车辆的 mask 底部有时会停在格线前几像素。这里仅在格线上方
    有连续证据、下方尚无足够 mask 时生成候选，最终仍由实例类别规则复核。
    """
    binary = np.asarray(mask) >= 0.5
    height, width = binary.shape
    x_edges = grid_edges(width, spec.columns)
    y_edges = grid_edges(height, spec.rows)
    occupied_rows = np.flatnonzero(binary.any(axis=1))
    if not occupied_rows.size:
        return []
    mask_bottom = int(occupied_rows[-1]) + 1
    candidate: tuple[int, int, int] | None = None
    for row in range(spec.rows - 1):
        boundary = y_edges[row + 1]
        cell_height = y_edges[row + 1] - y_edges[row]
        band = max(3, round(cell_height * 0.05))
        if 0 <= boundary - mask_bottom <= band:
            candidate = (row, boundary, band)
            break
    if candidate is None:
        return []

    row, boundary, band = candidate
    extensions: set[int] = set()
    for column in range(spec.columns):
        left, right = x_edges[column], x_edges[column + 1]
        above = binary[max(0, boundary - band) : boundary, left:right]
        below = binary[boundary : min(height, boundary + band), left:right]
        if (
            int(above.sum()) >= int(minimum_band_pixels)
            and int(below.sum()) < int(minimum_band_pixels)
        ):
            extensions.add((row + 1) * spec.columns + column)
    return sorted(extensions)


def render_mask_overlay(
    image: Image.Image,
    masks: Iterable[np.ndarray],
    spec: GridSpec,
    selected_indices: Iterable[int],
) -> Image.Image:
    """将目标 mask、网格和命中格子叠加到原图。"""
    canvas = image.convert("RGBA")
    union = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
    for mask in masks:
        normalized = normalize_mask(mask, canvas.size)
        union = np.maximum(union, (normalized >= 0.5).astype(np.uint8) * 150)
    if union.any():
        alpha = Image.fromarray(union, mode="L")
        color = Image.new("RGBA", canvas.size, (0, 190, 160, 255))
        canvas = Image.composite(color, canvas, alpha)

    draw = ImageDraw.Draw(canvas, "RGBA")
    selected = set(selected_indices)
    x_edges = grid_edges(canvas.width, spec.columns)
    y_edges = grid_edges(canvas.height, spec.rows)
    line_width = max(2, min(canvas.width, canvas.height) // 180)
    for row in range(spec.rows):
        for column in range(spec.columns):
            index = row * spec.columns + column
            box = (
                x_edges[column],
                y_edges[row],
                x_edges[column + 1] - 1,
                y_edges[row + 1] - 1,
            )
            color = (32, 190, 90, 255) if index in selected else (255, 255, 255, 210)
            draw.rectangle(box, outline=color, width=line_width)
            draw.text((box[0] + 5, box[1] + 4), str(index), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 180))
    return canvas
