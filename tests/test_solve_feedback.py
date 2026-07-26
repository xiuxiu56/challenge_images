import json

from PIL import Image

from challenge_images.online.solve_feedback import (
    FEEDBACK_FILENAME,
    OUTCOME_FAILED,
    OUTCOME_PASSED,
    OUTCOME_UNKNOWN,
    SolveFeedbackStore,
    SolveRecord,
    export_tile_labels,
    format_feedback_report,
)


def _record(name="m_01pns0_1.jpg", clicked=(0, 4), outcome=OUTCOME_PASSED, target="Hydrant", grid=(3, 3)):
    return SolveRecord(
        image_name=name,
        image_sha256="a" * 64,
        challenge_type="dynamic",
        target_class=target,
        grid_rows=grid[0],
        grid_cols=grid[1],
        clicked_indices=list(clicked),
        outcome=outcome,
    )


def _make_capture(tmp_path, name="m_01pns0_1.jpg", size=(300, 300)):
    directory = tmp_path / "dynamic" / "消防栓"
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(directory / name)
    return directory / name


# ---------- 记录语义 ----------


def test_only_passed_records_are_usable():
    """未通过时无法区分点错与漏点，两种情况标签完全不同。"""
    assert _record(outcome=OUTCOME_PASSED).usable is True
    assert _record(outcome=OUTCOME_FAILED).usable is False
    assert _record(outcome=OUTCOME_UNKNOWN).usable is False


def test_record_roundtrip_preserves_grid():
    record = _record(grid=(4, 4), clicked=[2, 3, 6])
    restored = SolveRecord.from_dict(record.as_dict())
    assert restored.spec.rows == 4
    assert restored.spec.columns == 4
    assert restored.clicked_indices == [2, 3, 6]
    assert restored.outcome == OUTCOME_PASSED


def test_record_handles_malformed_grid():
    restored = SolveRecord.from_dict({"网格": "坏数据"})
    assert (restored.grid_rows, restored.grid_cols) == (3, 3)


# ---------- 存储 ----------


def test_store_appends_and_persists(tmp_path):
    store = SolveFeedbackStore(tmp_path / FEEDBACK_FILENAME)
    store.append(_record())
    store.append(_record(name="b.jpg", outcome=OUTCOME_FAILED))

    reloaded = SolveFeedbackStore(tmp_path / FEEDBACK_FILENAME)
    assert len(reloaded.records) == 2
    assert len(reloaded.usable_records()) == 1
    payload = json.loads((tmp_path / FEEDBACK_FILENAME).read_text(encoding="utf-8"))
    assert payload["可用于标注"] == 1


def test_store_stamps_time_automatically(tmp_path):
    store = SolveFeedbackStore(tmp_path / FEEDBACK_FILENAME)
    saved = store.append(_record())
    assert saved.recorded_at


def test_statistics_project_tile_yield(tmp_path):
    store = SolveFeedbackStore(tmp_path / FEEDBACK_FILENAME)
    store.append(_record())                                   # 3×3 → 9 块
    store.append(_record(name="b.jpg", grid=(4, 4)))          # 4×4 → 16 块
    store.append(_record(name="c.jpg", outcome=OUTCOME_FAILED))

    stats = store.statistics()
    assert stats["记录总数"] == 3
    assert stats["可标注记录"] == 2
    assert stats["可产出图块"] == 25
    assert stats["按目标类别"]["Hydrant"] == 2


def test_report_explains_empty_state(tmp_path):
    store = SolveFeedbackStore(tmp_path / FEEDBACK_FILENAME)
    assert "尚无通过的挑战记录" in format_feedback_report(store)


# ---------- 图块标注导出 ----------


def test_export_produces_positive_and_negative_tiles(tmp_path):
    _make_capture(tmp_path)
    output = tmp_path / "labelled"

    report = export_tile_labels([_record(clicked=(0, 4))], tmp_path, output)

    assert report.records_used == 1
    # 3×3 共 9 块：点击 2 块为正，其余 7 块为负。
    assert report.positive_tiles == 2
    assert report.negative_tiles == 7
    assert report.total_tiles == 9
    assert report.per_class["Hydrant"] == 2
    assert len(list((output / "images").glob("*.jpg"))) == 9


def test_export_marks_unclicked_tiles_as_empty_labels(tmp_path):
    """未点击的格子是「不含目标」的负样本，而非某个其他类别。"""
    _make_capture(tmp_path)
    output = tmp_path / "labelled"
    export_tile_labels([_record(clicked=(0,))], tmp_path, output)

    payload = json.loads((output / "tile_labels.json").read_text(encoding="utf-8"))
    by_name = {item["图片"]: item for item in payload["样本"]}
    positive = next(item for item in by_name.values() if item["标签"])
    negative = next(item for item in by_name.values() if not item["标签"])

    assert positive["标签"] == ["Hydrant"]
    assert negative["标签"] == []
    assert "未点击" in negative["来源"]
    # 负样本仍记录本轮目标，便于回溯它是相对哪个类别的负样本。
    assert negative["本轮目标"] == "Hydrant"


def test_export_skips_unusable_records(tmp_path):
    _make_capture(tmp_path)
    output = tmp_path / "labelled"
    report = export_tile_labels(
        [_record(outcome=OUTCOME_FAILED), _record(outcome=OUTCOME_UNKNOWN)],
        tmp_path,
        output,
    )
    assert report.records_used == 0
    assert report.total_tiles == 0


def test_export_counts_missing_images(tmp_path):
    output = tmp_path / "labelled"
    report = export_tile_labels([_record(name="不存在.jpg")], tmp_path, output)
    assert report.skipped_missing_image == 1
    assert report.records_used == 0


def test_export_handles_4x4_grid(tmp_path):
    _make_capture(tmp_path, name="m_01bjv_9.jpg", size=(450, 450))
    output = tmp_path / "labelled"
    report = export_tile_labels(
        [_record(name="m_01bjv_9.jpg", grid=(4, 4), clicked=(5, 6, 9, 10), target="Bus")],
        tmp_path,
        output,
    )
    assert report.total_tiles == 16
    assert report.positive_tiles == 4
    assert report.per_class["Bus"] == 4


def test_export_records_classes_for_multilabel_training(tmp_path):
    _make_capture(tmp_path)
    _make_capture(tmp_path, name="m_01bjv_2.jpg")
    output = tmp_path / "labelled"
    export_tile_labels(
        [
            _record(clicked=(0,)),
            _record(name="m_01bjv_2.jpg", clicked=(1,), target="Bus"),
        ],
        tmp_path,
        output,
    )
    payload = json.loads((output / "tile_labels.json").read_text(encoding="utf-8"))
    assert payload["类别"] == ["Bus", "Hydrant"]
    assert payload["图块数"] == 18
