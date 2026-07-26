import json
from pathlib import Path

from PIL import Image

from challenge_images.data.hard_samples import HardSample, build_m2_dataset, export_hard_samples


def test_export_hard_samples_skips_deferred_class(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (300, 300), "white").save(source)
    output = tmp_path / "review"
    samples = (
        HardSample(str(source), 4, "Hydrant", "小目标", approved=True),
        HardSample(str(source), 1, "Boat", "暂缓", deferred=True),
    )

    result = export_hard_samples(samples, output, tmp_path)

    assert result["导出数量"] == 1
    assert result["暂缓数量"] == 1
    assert (output / "Hydrant" / "source__格子4.jpg").is_file()
    assert not (output / "Boat").exists()
    manifest = json.loads((output / "审核清单.json").read_text(encoding="utf-8"))
    assert manifest["Boat处理"] == "暂缓，不创建 Boat 类。"
    assert manifest["记录"][0]["状态"] == "已审核通过"


def test_build_m2_dataset_only_adds_approved_samples(tmp_path: Path):
    base = tmp_path / "base"
    (base / "train" / "Hydrant").mkdir(parents=True)
    Image.new("RGB", (20, 20), "red").save(base / "train" / "Hydrant" / "base.jpg")
    review = tmp_path / "review"
    (review / "Hydrant").mkdir(parents=True)
    approved = review / "Hydrant" / "approved.jpg"
    pending = review / "Hydrant" / "pending.jpg"
    Image.new("RGB", (20, 20), "blue").save(approved)
    Image.new("RGB", (20, 20), "green").save(pending)
    (review / "审核清单.json").write_text(
        json.dumps(
            {
                "记录": [
                    {"label": "Hydrant", "状态": "已审核通过", "导出文件": str(approved)},
                    {"label": "Hydrant", "状态": "待人工审核", "导出文件": str(pending)},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_m2_dataset(base, review, tmp_path / "m2")

    assert result["基础链接数量"] == 1
    assert result["审核通过并加入数量"] == 1
    assert (tmp_path / "m2" / "train" / "Hydrant" / "base.jpg").is_symlink()
    assert (tmp_path / "m2" / "train" / "Hydrant" / "困难样本__approved.jpg").is_file()
    assert not (tmp_path / "m2" / "train" / "Hydrant" / "困难样本__pending.jpg").exists()
