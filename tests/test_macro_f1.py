import json
from pathlib import Path

from challenge_images.training.macro_f1 import (
    BEST_MACRO_F1_WEIGHT,
    HISTORY_FILENAME,
    MacroF1Tracker,
    macro_f1_from_counts,
)


NAMES = {0: "Car", 1: "Tractor"}


class _FakeValidator:
    """模拟 Ultralytics 验证器保留的 targets / pred。"""

    def __init__(self, targets, pred, names=None):
        self.targets = targets
        # 真实验证器的 pred 是每个样本的 Top-5 索引。
        self.pred = pred
        self.names = names if names is not None else NAMES


class _FakeTrainer:
    def __init__(self, save_dir: Path, validator, epoch=0, metrics=None):
        self.save_dir = save_dir
        self.validator = validator
        self.epoch = epoch
        self.metrics = metrics or {}


def _make_run(tmp_path: Path, name="run") -> Path:
    run_dir = tmp_path / name
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"weights-v1")
    return run_dir


def test_macro_f1_is_unweighted_average_over_classes():
    """大类完美、小类全错时 macro-F1 必须显著低于 top1。"""
    # 98 个 Car 全对，2 个 Tractor 全错判成 Car。
    truths = [0] * 98 + [1, 1]
    preds = [0] * 100
    macro, scores = macro_f1_from_counts(truths, preds, NAMES)

    accuracy = sum(t == p for t, p in zip(truths, preds)) / len(truths)
    assert accuracy == 0.98
    by_name = {score.name: score for score in scores}
    assert by_name["Tractor"].f1 == 0.0
    # Car 的 precision 被 2 个误判拉低。
    assert by_name["Car"].precision == 0.98
    # 两类中有一类彻底失败，macro-F1 应落在 0.5 附近，而 top1 仍高达 0.98。
    # 这正是本项目 top1 0.9227 与 macro-F1 0.8595 差 7 个点的成因。
    assert 0.45 < macro < 0.55
    assert accuracy - macro > 0.4


def test_macro_f1_perfect_predictions():
    truths = [0, 0, 1, 1]
    macro, scores = macro_f1_from_counts(truths, list(truths), NAMES)
    assert macro == 1.0
    assert all(score.f1 == 1.0 for score in scores)


def test_absent_classes_are_not_counted():
    """验证集中没有出现的类别不应把 macro-F1 拉向 0。"""
    names = {0: "Car", 1: "Tractor", 2: "Boat"}
    truths = [0, 0, 1, 1]
    macro, scores = macro_f1_from_counts(truths, list(truths), names)
    assert {score.name for score in scores} == {"Car", "Tractor"}
    assert macro == 1.0


def test_tracker_promotes_weight_on_improvement(tmp_path):
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()

    # 第 1 轮：小类全错。
    trainer = _FakeTrainer(
        run_dir,
        _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [0, 1]]]),
        epoch=0,
        metrics={"metrics/accuracy_top1": 0.667},
    )
    first = tracker.evaluate(trainer)
    assert first is not None
    promoted = run_dir / "weights" / BEST_MACRO_F1_WEIGHT
    assert promoted.read_bytes() == b"weights-v1"
    assert tracker.best_epoch == 1

    # 第 2 轮：全对，权重内容变化后应被提升。
    (run_dir / "weights" / "last.pt").write_bytes(b"weights-v2")
    better = _FakeTrainer(
        run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [1, 0]]]), epoch=1
    )
    second = tracker.evaluate(better)
    assert second == 1.0
    assert second > first
    assert promoted.read_bytes() == b"weights-v2"
    assert tracker.best_epoch == 2


def test_tracker_keeps_previous_best_when_metric_drops(tmp_path):
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()

    tracker.evaluate(
        _FakeTrainer(run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [1, 0]]]), epoch=0)
    )
    (run_dir / "weights" / "last.pt").write_bytes(b"weights-worse")
    tracker.evaluate(
        _FakeTrainer(run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [0, 1]]]), epoch=1)
    )

    # 第 2 轮更差，最佳权重不应被覆盖。
    assert (run_dir / "weights" / BEST_MACRO_F1_WEIGHT).read_bytes() == b"weights-v1"
    assert tracker.best_epoch == 1


def test_history_is_written_with_per_class_detail(tmp_path):
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()
    tracker.evaluate(
        _FakeTrainer(run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [0, 1]]]), epoch=0)
    )

    payload = json.loads((run_dir / HISTORY_FILENAME).read_text(encoding="utf-8"))
    assert payload["最佳轮次"] == 1
    assert len(payload["历史"]) == 1
    assert "Tractor" in payload["历史"][0]["逐类"]
    assert payload["历史"][0]["逐类"]["Tractor"]["f1"] == 0.0


def test_tracker_ignores_unusable_validation(tmp_path):
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()

    # 没有验证器
    assert tracker.evaluate(_FakeTrainer(run_dir, None)) is None
    # targets 与 pred 数量不一致
    assert tracker.evaluate(_FakeTrainer(run_dir, _FakeValidator([[0, 1]], [[[0]]]))) is None
    assert tracker.best_epoch is None


def test_summary_reports_best_epoch(tmp_path):
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()
    assert "未能记录" in tracker.summary()

    tracker.evaluate(
        _FakeTrainer(run_dir, _FakeValidator([[0, 1]], [[[0, 1], [1, 0]]]), epoch=3)
    )
    assert "轮次=4" in tracker.summary()
    assert "1.0000" in tracker.summary()


def test_final_validation_is_not_recorded(tmp_path):
    """训练后的 final_eval 也会触发回调，但必须跳过。

    此时 Ultralytics 验证的是 best.pt，而 last.pt 是最后一轮的权重。
    记录下来会把 best.pt 的指标配到 last.pt 的权重上——50 轮训练配
    patience 时两者几乎必然不同。
    """
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()

    # 正常的第 3 轮（共 3 轮）
    trainer = _FakeTrainer(
        run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [0, 1]]]), epoch=2
    )
    trainer.epochs = 3
    tracker.evaluate(trainer)
    assert tracker.best_epoch == 3
    assert len(tracker.history) == 1

    # final_eval：epoch 已超出总轮数，且此刻 last.pt 内容已变
    (run_dir / "weights" / "last.pt").write_bytes(b"last-epoch-weights")
    final = _FakeTrainer(
        run_dir, _FakeValidator([[0, 0, 1]], [[[0, 1], [0, 1], [1, 0]]]), epoch=3
    )
    final.epochs = 3
    assert tracker.evaluate(final) is None
    # 历史与最佳轮次都不受影响。
    assert len(tracker.history) == 1
    assert tracker.best_epoch == 3
    # 最佳权重仍是第 3 轮当时的内容，没有被 last.pt 覆盖。
    assert (run_dir / "weights" / BEST_MACRO_F1_WEIGHT).read_bytes() == b"weights-v1"


def test_epochs_attribute_absent_still_records(tmp_path):
    """拿不到总轮数时保持原行为，不因缺少属性而静默失效。"""
    run_dir = _make_run(tmp_path)
    tracker = MacroF1Tracker()
    trainer = _FakeTrainer(run_dir, _FakeValidator([[0, 1]], [[[0, 1], [1, 0]]]), epoch=0)
    assert not hasattr(trainer, "epochs")
    assert tracker.evaluate(trainer) == 1.0
