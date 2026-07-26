from pathlib import Path

from PIL import Image

from challenge_images.data.online_stats import scan_online_capture


def _save_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color).save(path)


def test_online_stats_counts_categories_and_exact_duplicates(tmp_path: Path):
    root = tmp_path / "online_capture"
    first = root / "dynamic" / "人行横道" / "a.png"
    duplicate = root / "dynamic" / "人行横道" / "b.png"
    _save_image(first, "red")
    duplicate.write_bytes(first.read_bytes())
    _save_image(root / "imageselect" / "车" / "c.png", "blue")
    tile = root / "replacements" / "dynamic" / "人行横道" / "tile-1.png"
    tile_duplicate = root / "replacements" / "dynamic" / "人行横道" / "tile-2.png"
    _save_image(tile, "green")
    tile_duplicate.write_bytes(tile.read_bytes())

    result = scan_online_capture(root)

    assert result["total"] == 5
    assert result["unique"] == 3
    assert result["duplicate_groups"] == 2
    assert result["duplicate_files"] == 4
    assert result["extra"] == 2
    assert len(result["category_rows"]) == 3
    assert result["duplicate_rows"][0]["count"] == 2


def test_online_stats_filters_archive_kind_and_challenge_type(tmp_path: Path):
    root = tmp_path / "online_capture"
    _save_image(root / "dynamic" / "人行横道" / "full.png", "red")
    _save_image(root / "imageselect" / "车" / "other.png", "blue")
    _save_image(
        root / "replacements" / "dynamic" / "人行横道" / "tile.png",
        "green",
    )

    full = scan_online_capture(root, archive_kind="full_challenge")
    replacements = scan_online_capture(root, archive_kind="replacement_tile")
    dynamic = scan_online_capture(root, challenge_type="dynamic")

    assert full["total"] == 2
    assert {row["archive_label"] for row in full["category_rows"]} == {"完整挑战图"}
    assert replacements["total"] == 1
    assert replacements["category_rows"][0]["archive_label"] == "替换单格图"
    assert dynamic["total"] == 2
    assert {row["challenge_type"] for row in dynamic["category_rows"]} == {"dynamic"}
