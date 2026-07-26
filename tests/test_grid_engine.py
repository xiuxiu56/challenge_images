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
    # tileselect 实测 450×450，pmeta 也写 4,4；此前被误配为 3×3。
    assert grid_for_challenge("tileselect").count == 16
    assert grid_for_challenge("multicaptcha").count == 16


def test_multicaptcha_grid_overrides_stale_three_by_three_metadata():
    """4×4 题型必须按 4×4，不受缺失或过期 pmeta 影响。

    实测 pmeta 会残留上一轮的 3×3，若采信它会把 16 格图按 9 格切分。
    """
    for challenge in ("multicaptcha", "tileselect"):
        assert resolve_challenge_grid(challenge).text == "4×4"
        assert resolve_challenge_grid(challenge, (3, 3)).text == "4×4"
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


# ---------- 图片尺寸信号 ----------


def test_grid_from_image_size_recognises_standard_sizes():
    """300×300 与 450×450 是 reCAPTCHA 的标准完整挑战图尺寸。"""
    from challenge_images.grid.grid_engine import grid_from_image_size

    assert grid_from_image_size((300, 300)) == GridSpec(3, 3)
    assert grid_from_image_size((450, 450)) == GridSpec(4, 4)


def test_grid_from_image_size_ignores_single_tiles():
    """单格替换图约 100~112px，不能被当成完整挑战图推断网格。"""
    from challenge_images.grid.grid_engine import grid_from_image_size

    assert grid_from_image_size((100, 100)) is None
    assert grid_from_image_size((112, 112)) is None


def test_grid_from_image_size_rejects_non_square():
    from challenge_images.grid.grid_engine import grid_from_image_size

    assert grid_from_image_size((300, 450)) is None
    assert grid_from_image_size(None) is None
    assert grid_from_image_size((0, 0)) is None


def test_grid_from_image_size_estimates_unknown_sizes():
    """非标准尺寸按每格约 112px 估算并夹到 3 或 4。"""
    from challenge_images.grid.grid_engine import grid_from_image_size

    assert grid_from_image_size((320, 320)) == GridSpec(3, 3)
    assert grid_from_image_size((440, 440)) == GridSpec(4, 4)
    # 超大图仍夹到 4，不会返回 5×5 这种不支持的网格。
    assert grid_from_image_size((900, 900)) == GridSpec(4, 4)


def test_image_size_outranks_stale_detection():
    """pmeta 可能残留上一轮的值，像素证据必须优先。"""
    spec = resolve_challenge_grid("multicaptcha", detected=(3, 3), image_size=(450, 450))
    assert spec == GridSpec(4, 4)

    spec = resolve_challenge_grid("dynamic", detected=(4, 4), image_size=(300, 300))
    assert spec == GridSpec(3, 3)


def test_detection_used_when_size_unavailable():
    assert resolve_challenge_grid("unknown", detected=(4, 4)) == GridSpec(4, 4)


def test_type_default_is_last_resort():
    assert resolve_challenge_grid("multicaptcha") == GridSpec(4, 4)
    assert resolve_challenge_grid("dynamic") == GridSpec(3, 3)
    assert resolve_challenge_grid(None) == GridSpec(3, 3)


def test_tileselect_is_4x4():
    """tileselect 实测为 450×450，其 pmeta 也明确写 4,4。

    此前被配置成 3×3，会把 16 格图按 9 格切分导致完全错位。
    """
    assert grid_for_challenge("tileselect") == GridSpec(4, 4)
    assert resolve_challenge_grid("tileselect") == GridSpec(4, 4)
    assert resolve_challenge_grid("tileselect", image_size=(450, 450)) == GridSpec(4, 4)
