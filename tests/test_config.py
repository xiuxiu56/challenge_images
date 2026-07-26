from challenge_images.config import (
    COMMON_TRAIN_PARAMS,
    DEFAULT_SEGMENTATION_TRAIN,
    DEFAULT_TRAIN,
    EXPERIMENT_PRESETS,
    available_model_choices,
    model_display_name,
    next_available_run_name,
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
    assert DEFAULT_SEGMENTATION_TRAIN["imgsz"] == 640
    assert DEFAULT_SEGMENTATION_TRAIN["batch"] == 8
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
    assert DEFAULT_TRAIN["imgsz"] == 640
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


def test_medium_model_presets_use_formal_batch_and_include_640():
    assert EXPERIMENT_PRESETS["m@224"]["batch"] == 32
    assert EXPERIMENT_PRESETS["m@320"]["batch"] == 32
    assert EXPERIMENT_PRESETS["m@640"]["model"] == "yolo26m-cls.pt"
    assert EXPERIMENT_PRESETS["m@640"]["imgsz"] == 640
    assert EXPERIMENT_PRESETS["m@640"]["batch"] == 32
    assert EXPERIMENT_PRESETS["m@640"]["name"] == "recaptcha_v2_m2_640"
    assert str(EXPERIMENT_PRESETS["m@640"]["data"]).endswith("dataset_cls_m2_640")


def test_model_training_profiles_are_independent():
    nano = training_profile_for_model("yolo26n-cls.pt")
    medium = training_profile_for_model("/tmp/yolo26m-cls.pt")
    large = training_profile_for_model("yolo26l-cls.pt")

    assert (nano["imgsz"], nano["batch"]) == (224, 32)
    assert (medium["imgsz"], medium["batch"]) == (640, 32)
    assert str(medium["data"]).endswith("dataset_cls_m2_640")
    assert medium["name"] == "recaptcha_v2_m2_640"
    assert (large["imgsz"], large["batch"]) == (640, 32)


def test_resolution_switches_dataset_and_run_name():
    profile_320 = training_profile_for_model("yolo26m-cls.pt", imgsz=320)
    profile_640 = training_profile_for_model("yolo26m-cls.pt", imgsz=640)

    assert training_data_for_imgsz(224).name == "dataset_cls_full_57k"
    assert training_data_for_imgsz(320).name == "dataset_cls_m2_320"
    assert training_data_for_imgsz(640).name == "dataset_cls_m2_640"
    assert str(profile_320["data"]).endswith("dataset_cls_m2_320")
    assert profile_320["name"] == "recaptcha_v2_m2_320"
    assert str(profile_640["data"]).endswith("dataset_cls_m2_640")
    assert profile_640["name"] == "recaptcha_v2_m2_640"


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
