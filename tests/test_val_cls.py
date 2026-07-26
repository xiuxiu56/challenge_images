from pathlib import Path

from challenge_images.training import val_cls


def test_external_report_uses_val_and_filters_hidden_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "dataset"
    class_dir = root / "val" / "Crosswalk"
    class_dir.mkdir(parents=True)
    (class_dir / "sample.jpg").write_bytes(b"image-placeholder")
    (class_dir / ".DS_Store").write_bytes(b"macos-metadata")

    captured: dict[str, list[str]] = {}

    class FakeProbabilities:
        top1 = 0

    class FakeResult:
        probs = FakeProbabilities()

    class FakeModel:
        names = {0: "Crosswalk"}

        def predict(self, source, **_kwargs):
            captured["source"] = source
            return [FakeResult() for _ in source]

    class FakeYOLO:
        def __new__(cls, _weights):
            return FakeModel()

    monkeypatch.setattr(val_cls, "resolve_weights", lambda _weights: tmp_path / "best.pt")
    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)

    report = val_cls.evaluate_directory(tmp_path / "best.pt", root, device="cpu")

    assert report["data"] == str(root / "val")
    assert report["total"] == 1
    assert captured["source"] == [str(class_dir / "sample.jpg")]
