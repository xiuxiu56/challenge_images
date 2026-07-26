import json
from pathlib import Path

from PIL import Image

from challenge_images.online.capture_service import (
    ARCHIVE_REPLACEMENT_TILE,
    OnlineCaptureService,
    parse_pmeta_categories,
    parse_reload_response,
)


def _single_reload(category_mid: str = "/m/014xcs", label: str = "Crosswalk") -> list:
    response = [None] * 10
    response[4] = ["pmeta", [category_mid, None, None, None, None, None, label]]
    response[5] = "dynamic"
    response[9] = "只用于测试的临时令牌"
    return response


def test_parse_reload_response_extracts_type_and_category():
    parsed = parse_reload_response(")]}'\n" + json.dumps(_single_reload()))

    assert parsed["challenge_type"] == "dynamic"
    assert parsed["categories"] == [{"id": "/m/014xcs", "label": "Crosswalk"}]
    assert parsed["payload_token_sha256"]
    assert "只用于测试的临时令牌" not in json.dumps(parsed, ensure_ascii=False)


def test_parse_multicaptcha_categories_keeps_order_and_deduplicates():
    pmeta = [
        "pmeta",
        None,
        None,
        None,
        None,
        [[["/m/01bjv", None, None, None, None, None, "Bus"], ["/m/0k4j", None, None, None, None, None, "Car"], ["/m/01bjv"]]],
    ]

    assert parse_pmeta_categories(pmeta) == [
        {"id": "/m/01bjv", "label": "Bus"},
        {"id": "/m/0k4j", "label": "Car"},
    ]


def test_parse_multicaptcha_always_resolves_four_by_four_grid():
    """reload 即使残留 3×3 字段，multicaptcha 也应输出 4×4。"""
    response = _single_reload("/m/01bjv", "Bus")
    response[4][1][2] = 3
    response[4][1][3] = 3
    response[5] = "multicaptcha"

    parsed = parse_reload_response(json.dumps(response))

    assert parsed["challenge_type"] == "multicaptcha"
    assert parsed["grid"] == {"rows": 4, "columns": 4}


def test_import_online_sample_saves_image_and_sidecar_without_metadata_folder(tmp_path: Path):
    image_path = tmp_path / "payload.jpg"
    Image.new("RGB", (300, 300), "white").save(image_path)
    reload_path = tmp_path / "reload.txt"
    reload_path.write_text(json.dumps(_single_reload()), encoding="utf-8")
    service = OnlineCaptureService(tmp_path / "online")

    sample = service.import_sample(image_path, reload_path)

    assert sample.path.is_file()
    assert sample.metadata_path.is_file()
    assert sample.challenge_type == "dynamic"
    assert sample.target_class == "Crosswalk"
    assert sample.category_mid == "/m/014xcs"
    metadata = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
    assert metadata["category_zh"] == "人行横道"
    assert metadata["payload_token_sha256"]
    assert metadata["pmeta"] == _single_reload()[4]
    records_path = tmp_path / "online" / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert isinstance(records, list)
    assert records[0]["image_basename"] == "m_014xcs"
    assert records[0]["image_name"] == sample.path.name == "m_014xcs_1.jpg"
    assert records[0]["pmeta"] == _single_reload()[4]
    assert records[0]["id"] == 1
    assert not (tmp_path / "online" / "records.jsonl").exists()
    assert not (tmp_path / "online" / "metadata").exists()
    assert service.latest_sample() is not None


def test_replacement_tile_uses_separate_directory_and_record_chain(tmp_path: Path):
    image_path = tmp_path / "tile.jpg"
    Image.new("RGB", (80, 80), "green").save(image_path)
    service = OnlineCaptureService(tmp_path / "online")

    sample = service.import_sample(image_path)
    # 重新以带 pmeta 的内存样本归档为替换图，模拟 replaceimage 后的 payload。
    replacement = service.import_bytes(
        image_path.read_bytes(),
        json.dumps(_single_reload()),
        archive_kind=ARCHIVE_REPLACEMENT_TILE,
        source_tile_id=10,
        source_tile_index=4,
    )

    assert sample.path.parent != replacement.path.parent
    assert replacement.path.parent == tmp_path / "online" / "replacements" / "dynamic" / "人行横道"
    assert replacement.archive_kind == ARCHIVE_REPLACEMENT_TILE
    assert replacement.source_tile_id == 10
    assert replacement.source_tile_index == 4
    metadata = json.loads(replacement.metadata_path.read_text(encoding="utf-8"))
    assert metadata["archive_kind"] == ARCHIVE_REPLACEMENT_TILE
    assert metadata["source_tile_id"] == 10
    assert metadata["source_tile_index"] == 4
    assert metadata["pmeta"] == _single_reload()[4]
    records = json.loads(
        (tmp_path / "online" / "replacements" / "records.json").read_text(encoding="utf-8")
    )
    assert records[0]["source_tile_id"] == 10
    assert records[0]["source_tile_index"] == 4
    assert records[0]["pmeta"] == _single_reload()[4]


def test_import_without_reload_uses_manual_category(tmp_path: Path):
    image_path = tmp_path / "payload.png"
    Image.new("RGB", (400, 400), "blue").save(image_path)
    service = OnlineCaptureService(tmp_path / "online")

    sample = service.import_sample(
        image_path,
        challenge_type="multicaptcha",
        category="消防栓",
    )

    assert sample.challenge_type == "multicaptcha"
    assert sample.target_class == "Hydrant"
    assert sample.category_mid == "/m/01pns0"
