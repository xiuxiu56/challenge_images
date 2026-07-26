from PIL import Image

from challenge_images.grid.grid_engine import (
    GridSpec,
    draw_grid,
    grid_for_challenge,
    grid_index_from_point,
    replace_grid_tile,
    resolve_challenge_grid,
    split_grid,
)


def test_grid_count_and_order():
    image = Image.new("RGB", (300, 300))
    assert len(split_grid(image, GridSpec(3, 3))) == 9
    assert len(split_grid(image, GridSpec(4, 4))) == 16


def test_challenge_defaults():
    assert grid_for_challenge("dynamic").count == 9
    assert grid_for_challenge("imageselect").count == 9
    assert grid_for_challenge("tileselect").count == 9
    assert grid_for_challenge("multicaptcha").count == 16


def test_multicaptcha_grid_overrides_stale_three_by_three_metadata():
    """multicaptcha 必须按 4×4，不受缺失或过期 pmeta 影响。"""
    assert resolve_challenge_grid("multicaptcha").text == "4×4"
    assert resolve_challenge_grid("multicaptcha", (3, 3)).text == "4×4"
    assert resolve_challenge_grid("dynamic", (3, 3)).text == "3×3"


def test_selected_icon_is_drawn_at_cell_center(tmp_path):
    base = Image.new("RGB", (300, 300), "white")
    icon = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    icon_path = tmp_path / "icon.png"
    icon.save(icon_path)
    rendered = draw_grid(base, GridSpec(3, 3), selected=[4], icon_path=icon_path)
    # 第 4 格中心为 (150, 150)，应出现图标红色像素。
    red, green, blue, alpha = rendered.getpixel((150, 150))
    assert red > 200 and green < 80 and blue < 80 and alpha > 200


def test_grid_index_from_point_handles_edges_and_outside_area():
    spec = GridSpec(3, 3)
    assert grid_index_from_point(150, 150, 300, 300, spec) == 4
    assert grid_index_from_point(299.9, 299.9, 300, 300, spec) == 8
    assert grid_index_from_point(-1, 100, 300, 300, spec) is None
    assert grid_index_from_point(300, 100, 300, 300, spec) is None


def test_grid_index_from_point_supports_four_by_four():
    spec = GridSpec(4, 4)
    assert grid_index_from_point(0, 0, 640, 480, spec) == 0
    assert grid_index_from_point(639, 479, 640, 480, spec) == 15
    assert grid_index_from_point(320, 240, 640, 480, spec) == 10


def test_replace_grid_tile_only_updates_requested_position():
    """动态新图只回填点击格子，其他格子保持原样。"""
    base = Image.new("RGB", (300, 300), "red")
    replacement = Image.new("RGB", (100, 100), "blue")

    merged = replace_grid_tile(base, replacement, GridSpec(3, 3), 5)

    assert merged.getpixel((250, 150)) == (0, 0, 255)  # 格子 5
    assert merged.getpixel((50, 50)) == (255, 0, 0)  # 格子 0
    assert merged.getpixel((250, 250)) == (255, 0, 0)  # 格子 8
