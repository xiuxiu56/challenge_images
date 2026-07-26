from challenge_images.config import (
    COMMON_TRAIN_PARAMS,
    DEFAULT_SEGMENTATION_IMGSZ,
    DEFAULT_SEGMENTATION_TRAIN,
    DEFAULT_TRAIN,
    DEFAULT_TRAIN_IMGSZ,
    EXPERIMENT_PRESETS,
    NATIVE_TILE_PIXELS,
    available_model_choices,
    model_display_name,
    next_available_run_name,
    read_training_imgsz,
    resolve_model_reference,
    resolve_segmentation_model_reference,
    training_data_for_imgsz,
    training_profile_for_model,
)


def test_gui_choices_only_contain_existing_local_models():
    choices = available_model_choices()
    assert all(__import__("pathlib").Path(path).is_file() for path in choices)


def test_bare_model_name_is_preserved_when_not_local():
    resolved = resolve_model_reference("yolo26m-cls.pt")
    assert resolved.endswith("yolo26m-cls.pt")


def test_segmentation_defaults_use_independent_project_and_data():
    # 4×4 大图原生 450×450，512 已覆盖；显存下降后批次同步放大。
    assert DEFAULT_SEGMENTATION_TRAIN["imgsz"] == DEFAULT_SEGMENTATION_IMGSZ == 512
    assert DEFAULT_SEGMENTATION_TRAIN["batch"] == 16
    assert DEFAULT_SEGMENTATION_TRAIN["epochs"] == 80
    assert str(DEFAULT_SEGMENTATION_TRAIN["data"]).endswith("recaptcha_seg_v1/data.yaml")
    assert str(DEFAULT_SEGMENTATION_TRAIN["project"]).endswith("runs/segment")
    assert resolve_segmentation_model_reference("yolo26m-seg.pt").endswith("yolo26m-seg.pt")


def test_model_display_name_uses_experiment_metadata(tmp_path):
    import json
    model_dir = tmp_path / "recaptcha_v2_n1"
    model_dir.mkdir()
    weight = model_dir / "best.pt"
    weight.write_bytes(b"model")
    (model_dir / "model_meta.json").write_text(
        json.dumps({"模型": "/tmp/yolo26n-cls.pt", "训练参数": {"name": "recaptcha_v2_n1"}}),
        encoding="utf-8",
    )
    assert model_display_name(weight) == "recaptcha_v2_n1｜yolo26n-cls｜最佳模型"


def test_second_training_defaults_target_small_objects():
    # 图块原生尺寸约 100~112px，训练分辨率贴近原生而非向上采样到 320/640。
    assert DEFAULT_TRAIN["imgsz"] == DEFAULT_TRAIN_IMGSZ == 160
    assert DEFAULT_TRAIN["imgsz"] > NATIVE_TILE_PIXELS
    assert DEFAULT_TRAIN["batch"] == 32
    assert DEFAULT_TRAIN["epochs"] == 50
    assert DEFAULT_TRAIN["patience"] == 12
    assert DEFAULT_TRAIN["dropout"] == 0.10
    assert DEFAULT_TRAIN["lr0"] == 0.0005
    assert DEFAULT_TRAIN["lrf"] == 0.05
    assert DEFAULT_TRAIN["warmup_epochs"] == 3.0
    assert DEFAULT_TRAIN["momentum"] == 0.9
    assert DEFAULT_TRAIN["nbs"] == 64
    assert DEFAULT_TRAIN["scale"] == 0.15
    assert DEFAULT_TRAIN["auto_augment"] == "augmix"
    assert DEFAULT_TRAIN["erasing"] == 0.02
    assert DEFAULT_TRAIN["deterministic"] is True
    assert DEFAULT_TRAIN["cache"] is False
    assert DEFAULT_TRAIN["fraction"] == 1.0
    assert DEFAULT_TRAIN["freeze"] is None
    assert "translate" not in DEFAULT_TRAIN
    assert "mixup" not in DEFAULT_TRAIN
    assert "cutmix" not in DEFAULT_TRAIN


def test_experiment_presets_compare_around_native_resolution():
    """对照实验围绕原生分辨率展开，不再包含 320/640 的上采样实验。"""
    sizes = {int(preset["imgsz"]) for preset in EXPERIMENT_PRESETS.values()}
    assert sizes == {128, 160, 224}
    assert EXPERIMENT_PRESETS["m@160"]["model"] == "yolo26m-cls.pt"
    assert EXPERIMENT_PRESETS["m@160"]["name"] == "recaptcha_v3_m_160"
    # 全部预设共用同一份数据，只有分辨率不同，保证对照有效。
    assert len({str(preset["data"]) for preset in EXPERIMENT_PRESETS.values()}) == 1


def test_model_training_profiles_are_independent():
    nano = training_profile_for_model("yolo26n-cls.pt")
    medium = training_profile_for_model("/tmp/yolo26m-cls.pt")
    large = training_profile_for_model("yolo26l-cls.pt")

    assert (nano["imgsz"], nano["batch"]) == (DEFAULT_TRAIN_IMGSZ, 64)
    assert (medium["imgsz"], medium["batch"]) == (DEFAULT_TRAIN_IMGSZ, 64)
    assert medium["name"] == "recaptcha_v3_m_160"
    # 大模型显存占用更高，批次单独降级。
    assert (large["imgsz"], large["batch"]) == (DEFAULT_TRAIN_IMGSZ, 32)


def test_dataset_no_longer_switches_with_resolution():
    """m2_640 实为指向 m2_320 的符号链接，数据版本与分辨率无关。"""
    assert training_data_for_imgsz(128).name == "dataset_cls_m2_320"
    assert training_data_for_imgsz(224).name == "dataset_cls_m2_320"
    assert training_data_for_imgsz(640).name == "dataset_cls_m2_320"

    profile_128 = training_profile_for_model("yolo26m-cls.pt", imgsz=128)
    profile_224 = training_profile_for_model("yolo26m-cls.pt", imgsz=224)
    assert profile_128["data"] == profile_224["data"]
    assert profile_128["name"] == "recaptcha_v3_m_128"
    assert profile_224["name"] == "recaptcha_v3_m_224"


def test_inference_imgsz_follows_model_metadata(tmp_path):
    """推理分辨率从权重元数据读取，缺少元数据时返回 None 交由上层兜底。"""
    import json

    model_dir = tmp_path / "recaptcha_v3_m_160"
    model_dir.mkdir()
    weight = model_dir / "best.pt"
    weight.write_bytes(b"model")
    assert read_training_imgsz(weight) is None

    (model_dir / "model_meta.json").write_text(
        json.dumps({"训练参数": {"imgsz": 320}}, ensure_ascii=False), encoding="utf-8"
    )
    assert read_training_imgsz(weight) == 320


def test_public_training_parameters_are_inherited_by_default_config():
    assert all(DEFAULT_TRAIN[key] == value for key, value in COMMON_TRAIN_PARAMS.items())


def test_run_name_checks_runs_and_exported_models(tmp_path):
    runs_dir = tmp_path / "runs" / "classify"
    models_dir = tmp_path / "models" / "trained"
    runs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    base_name = "recaptcha_v2_m2_320"
    assert next_available_run_name(
        base_name,
        project_dir=runs_dir,
        trained_dir=models_dir,
    ) == base_name

    (runs_dir / base_name).mkdir()
    assert next_available_run_name(
        base_name,
        project_dir=runs_dir,
        trained_dir=models_dir,
    ) == f"{base_name}_v1"

    (models_dir / f"{base_name}_v1").mkdir()
    (runs_dir / f"{base_name}_v2").mkdir()
    assert next_available_run_name(
        base_name,
        project_dir=runs_dir,
        trained_dir=models_dir,
    ) == f"{base_name}_v3"


def test_existing_version_name_continues_same_version_series(tmp_path):
    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"
    runs_dir.mkdir()
    models_dir.mkdir()
    (runs_dir / "experiment").mkdir()
    (runs_dir / "experiment_v1").mkdir()
    (models_dir / "experiment_v2").mkdir()

    assert next_available_run_name(
        "experiment_v2",
        project_dir=runs_dir,
        trained_dir=models_dir,
    ) == "experiment_v3"
