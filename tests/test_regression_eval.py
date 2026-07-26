import json

from challenge_images.grid.grid_engine import GridSpec
from challenge_images.tools.regression_eval import (
    CaseOutcome,
    RegressionReport,
    SampleCase,
    compare,
    evaluate,
    format_comparison,
    format_report,
    load_cases,
    save_report,
)


def _case(name, target="Car", challenge="dynamic", truth=None, tmp_path=None):
    path = (tmp_path / name) if tmp_path else name
    if tmp_path:
        path.write_bytes(b"img")
    return SampleCase(
        path=path if tmp_path else __import__("pathlib").Path(name),
        challenge_type=challenge,
        target_class=target,
        spec=GridSpec(3, 3),
        truth=truth,
    )


def test_outcome_reports_missed_and_extra():
    outcome = CaseOutcome(
        path="a.jpg", challenge_type="dynamic", target_class="Car",
        predicted=[0, 1, 5], truth=[0, 1, 2],
    )
    assert outcome.missed == [2]
    assert outcome.extra == [5]
    assert outcome.exact_match is False


def test_exact_match_ignores_order():
    outcome = CaseOutcome(
        path="a.jpg", challenge_type="dynamic", target_class="Car",
        predicted=[5, 0, 1], truth=[1, 5, 0],
    )
    assert outcome.exact_match is True


def test_metrics_aggregate_cell_level(tmp_path):
    cases = [
        _case("a.jpg", truth=[0, 1], tmp_path=tmp_path),
        _case("b.jpg", target="Bus", truth=[3], tmp_path=tmp_path),
    ]
    predictions = {"a.jpg": [0, 1], "b.jpg": [3, 4]}
    report = evaluate(cases, lambda case: predictions[case.path.name])

    overall = report.metrics()["整体"]
    # TP=3 (0,1,3), FP=1 (4), FN=0
    assert overall["precision"] == 0.75
    assert overall["recall"] == 1.0
    assert overall["误选格子数"] == 1
    assert overall["漏选格子数"] == 0
    assert overall["完全匹配率"] == 0.5


def test_metrics_split_by_class_and_challenge(tmp_path):
    cases = [
        _case("a.jpg", target="Car", challenge="dynamic", truth=[0], tmp_path=tmp_path),
        _case("b.jpg", target="Bus", challenge="multicaptcha", truth=[1], tmp_path=tmp_path),
    ]
    report = evaluate(cases, lambda case: case.truth)
    metrics = report.metrics()
    assert set(metrics["逐类别"]) == {"Car", "Bus"}
    assert set(metrics["逐挑战类型"]) == {"dynamic", "multicaptcha"}
    assert metrics["逐类别"]["Car"]["f1"] == 1.0


def test_evaluate_survives_single_failure(tmp_path):
    cases = [
        _case("ok.jpg", truth=[0], tmp_path=tmp_path),
        _case("bad.jpg", truth=[1], tmp_path=tmp_path),
    ]

    def recognize(case):
        if case.path.name == "bad.jpg":
            raise RuntimeError("模型加载失败")
        return [0]

    report = evaluate(cases, recognize)
    # 一条失败不应中断整轮评测。
    assert len(report.outcomes) == 1
    assert report.outcomes[0].path.endswith("ok.jpg")


def test_compare_reports_only_changed_samples():
    def build(mapping):
        return RegressionReport(outcomes=[
            CaseOutcome(path=p, challenge_type="dynamic", target_class="Car", predicted=v)
            for p, v in mapping.items()
        ])

    before = build({"a.jpg": [0, 1], "b.jpg": [3], "c.jpg": [5]})
    after = build({"a.jpg": [0, 1], "b.jpg": [3, 4], "c.jpg": []})

    payload = compare(before, after)
    assert payload["对照样本数"] == 3
    assert payload["结果变化样本数"] == 2
    changed = {item["图片"] for item in payload["变化明细"]}
    assert changed == {"b.jpg", "c.jpg"}
    assert payload["新增格子总数"] == 1
    assert payload["移除格子总数"] == 1
    assert "a.jpg" not in changed


def test_compare_handles_identical_runs():
    report = RegressionReport(outcomes=[
        CaseOutcome(path="a.jpg", challenge_type="dynamic", target_class="Car", predicted=[1])
    ])
    payload = compare(report, report)
    assert payload["结果变化样本数"] == 0
    assert payload["变化比例"] == 0.0
    assert "0 张" in format_comparison(payload)


def test_format_report_without_truth_explains_next_step():
    report = RegressionReport(outcomes=[
        CaseOutcome(path="a.jpg", challenge_type="dynamic", target_class="Car", predicted=[1])
    ])
    text = format_report(report)
    assert "没有任何真值标注" in text


def test_load_cases_reads_annotation_store(tmp_path):
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"img")
    annotations = tmp_path / "grid_annotations.json"
    annotations.write_text(json.dumps({
        str(image): {
            "挑战类型": "multicaptcha",
            "网格": "4x4",
            "目标类别": "Bus",
            "真实格子": [2, 3],
        }
    }, ensure_ascii=False), encoding="utf-8")

    cases = load_cases(annotations)
    assert len(cases) == 1
    assert cases[0].target_class == "Bus"
    assert cases[0].spec == GridSpec(4, 4)
    assert cases[0].truth == [2, 3]


def test_load_cases_skips_missing_images(tmp_path):
    annotations = tmp_path / "grid_annotations.json"
    annotations.write_text(json.dumps({
        "/nowhere/gone.jpg": {"挑战类型": "dynamic", "网格": "3x3", "目标类别": "Car", "真实格子": [0]}
    }, ensure_ascii=False), encoding="utf-8")
    assert load_cases(annotations) == []


def test_save_report_roundtrip(tmp_path):
    report = RegressionReport(outcomes=[
        CaseOutcome(path="a.jpg", challenge_type="dynamic", target_class="Car",
                    predicted=[0], truth=[0, 1])
    ])
    output = save_report(report, tmp_path / "r.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["含真值样本"] == 1
    assert payload["样本"][0]["漏选"] == [1]
