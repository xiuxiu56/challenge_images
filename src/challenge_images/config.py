"""
项目配置：路径、设备、模型推荐、训练默认超参。
按 M4 Pro 24GB + 本数据集（约 5.7 万张 / 14 类 / 极不均衡）调过一版。

设备：Mac Apple Silicon 上 PyTorch 用 MPS（Metal Performance Shaders）。
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from .runtime_env import choose_device, prepare_cache_dir

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data" / "classification" / "dataset_cls_full_57k"
CHALLENGE_DIR = ROOT / "data" / "challenge"
ONLINE_CAPTURE_DIR = ROOT / "data" / "online_capture"
ARCHIVE_DIR = ROOT / "data" / "archive"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
SEGMENTATION_MODELS_DIR = MODELS_DIR / "segmentation"
SEGMENTATION_PRETRAINED_DIR = SEGMENTATION_MODELS_DIR / "pretrained"
SEGMENTATION_TRAINED_DIR = SEGMENTATION_MODELS_DIR / "trained"
SEGMENTATION_DATA_DIR = ROOT / "data" / "segmentation" / "recaptcha_seg_v1"
BALANCED_DATA_DIR = ROOT / "data" / "classification" / "dataset_cls_balanced"
HARD_SAMPLES_DIR = ROOT / "data" / "classification" / "hard_samples_review"
M2_DATA_DIR_320 = ROOT / "data" / "classification" / "dataset_cls_m2_320"
M2_DATA_DIR_640 = ROOT / "data" / "classification" / "dataset_cls_m2_640"
# 保留旧名供困难样本数据构建器使用；正式训练由模型配置选择目录。
M2_DATA_DIR = M2_DATA_DIR_320
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
ANNOTATIONS_DIR = ROOT / "annotations"
WEIGHTS_DIR = TRAINED_MODELS_DIR

# ---------- 分辨率 ----------
# 实测原生尺寸（2026-07-27）：
#   3×3 dynamic/imageselect 大图 300×300 → 每格 100×100
#   4×4 multicaptcha        大图 450×450 → 每格 112×112
#   训练图块 88% 为 100×100，10% 为 120×120
# 训练分辨率只需略高于原生尺寸，给骨干网络的 stride 留下采样余量。
# 把 100px 图块上采样到 320/640 不增加任何信息量：实测 @320 相比 @224
# 多花 2.3 倍时间只换来 top1 +0.3%，@640 预期为负收益。
NATIVE_TILE_PIXELS = 112
DEFAULT_TRAIN_IMGSZ = 160
# 整图分割输入：4×4 大图 450×450，512 已覆盖且为 32 的整数倍。
DEFAULT_SEGMENTATION_IMGSZ = 512


# ---------- 设备（Apple Silicon → MPS） ----------
def mps_available() -> bool:
    """当前 PyTorch 是否能用 MPS（Metal GPU）。"""
    try:
        import torch

        return bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except Exception:
        return False


def pick_device(prefer: str | None = None) -> str:
    """
    选择训练/推理设备。
    - prefer 显式指定时优先用（mps / cpu / 0 等）
    - 默认：有 MPS 用 mps，否则 cpu
    Mac 上没有 CUDA；不要写 cuda。
    """
    if prefer:
        p = prefer.strip().lower()
        if p in ("mps", "metal"):
            if mps_available():
                return "mps"
            print("[提示] 请求 MPS，但当前不可用，回退到 cpu。")
            return "cpu"
        return prefer  # cpu / 0 / cpu:0 等原样交给 ultralytics
    return choose_device(prefer)


def device_status() -> str:
    """给人看的设备状态一行。"""
    try:
        import torch

        ver = torch.__version__
    except Exception:
        ver = "未安装"
    mps = mps_available()
    dev = pick_device()
    return f"torch={ver} | MPS={'可用' if mps else '不可用'} | 当前默认={dev}"


# 启动时定一次，菜单/训练都读这个
DEFAULT_DEVICE = pick_device()

# ---------- 模型选择说明 ----------
# YOLO26 分类权重命名：yolo26{n|s|m|l|x}-cls.pt
# 数据特征：
#   train=56186，val=1474，14 类
#   极不均衡：Tractor 仅 23 张；Bus/Bicycle/Hydrant 等近 7k+
# 机器：Mac mini M4 Pro / 24GB 统一内存
#
# 推荐优先级：
#   1) yolo26m-cls  —— 主推：当前数据规模优先比较精度
#   2) yolo26s-cls  —— 速度和部署基线
#   3) yolo26n-cls  —— 冒烟/调参：3~10 epoch 看流程
#   4) yolo26l/x-cls —— 后续对照，需以实测收益决定
#
# 当前正式实验增加 640 对照，重点观察小目标与远景图块识别效果。
RECOMMENDED_MODEL = "yolo26m-cls.pt"
RECOMMENDED_TRAINED_EXPERIMENT = "recaptcha_v2_m2_320"
SMOKE_MODEL = "yolo26n-cls.pt"
UPGRADE_MODEL = "yolo26m-cls.pt"
RECOMMENDED_SEGMENTATION_MODEL = "yolo26m-seg.pt"

MODEL_CHOICES = {
    "1": ("yolo26n-cls.pt", "nano  最快，冒烟/调参"),
    "2": ("yolo26s-cls.pt", "small 速度和部署基线"),
    "3": ("yolo26m-cls.pt", "medium 正式训练主推"),
    "4": ("yolo26l-cls.pt", "large  慢，最后再试"),
    "5": ("yolo26x-cls.pt", "xlarge 最慢，一般不建议"),
}

SEGMENTATION_MODEL_CHOICES = {
    "1": ("yolo26n-seg.pt", "nano  分割流程冒烟"),
    "2": ("yolo26s-seg.pt", "small 数据与速度基线"),
    "3": ("yolo26m-seg.pt", "medium 正式分割主推"),
    "4": ("yolo26l-seg.pt", "large  训练和推理更慢"),
    "5": ("yolo26x-seg.pt", "xlarge 资源占用最高"),
}

# 对照实验围绕原生分辨率（100~112px）展开，不再向上试探 320/640。
EXPERIMENT_PRESETS = {
    "m@128": {
        "model": "yolo26m-cls.pt",
        "data": str(M2_DATA_DIR_320),
        "imgsz": 128,
        "batch": 64,
        "name": "recaptcha_v3_m_128",
    },
    "m@160": {
        "model": "yolo26m-cls.pt",
        "data": str(M2_DATA_DIR_320),
        "imgsz": 160,
        "batch": 64,
        "name": "recaptcha_v3_m_160",
    },
    "m@224": {
        "model": "yolo26m-cls.pt",
        "data": str(M2_DATA_DIR_320),
        "imgsz": 224,
        "batch": 32,
        "name": "recaptcha_v3_m_224",
    },
    "s@160": {
        "model": "yolo26s-cls.pt",
        "data": str(M2_DATA_DIR_320),
        "imgsz": 160,
        "batch": 64,
        "name": "recaptcha_v3_s_160",
    },
}

# ---------- 分类训练公共超参 ----------
# 320/640 对照实验共用这些参数，确保对比时只改变分辨率。
COMMON_TRAIN_PARAMS = {
    "epochs": 50,
    "batch": 32,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.05,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "momentum": 0.9,
    "nbs": 64,
    "weight_decay": 0.001,
    "dropout": 0.10,
    "patience": 12,
    "scale": 0.15,
    "fliplr": 0.5,
    "flipud": 0.0,
    "auto_augment": "augmix",
    "erasing": 0.02,
    # MPS 上 amp 多数情况可用；若 loss 变 nan / 报错，菜单里改 n。
    "amp": True,
    "workers": 4,
    "cache": False,
    "seed": 0,
    "deterministic": True,
    "fraction": 1.0,
    "freeze": None,
}

# ---------- 训练默认超参（分类） ----------
# 正式训练默认使用 m@160：略高于 112px 原生图块，训练耗时约为 @320 的 1/4。
DEFAULT_TRAIN = {
    **COMMON_TRAIN_PARAMS,
    "data": str(M2_DATA_DIR_320),
    "imgsz": DEFAULT_TRAIN_IMGSZ,
    # device 在 train 时用 pick_device() 再确认一次，避免 import 时 MPS 状态过期。
    "device": DEFAULT_DEVICE,
    "project": str(RUNS_DIR / "classify"),
    "name": "recaptcha_v3_m_160",
    "exist_ok": False,
    "pretrained": True,
    "verbose": True,
    "save_period": 2,
}

# ---------- 分割训练默认超参 ----------
# 分割数据使用 YOLO 多边形标签，和分类目录结构不同。
DEFAULT_SEGMENTATION_TRAIN = {
    "data": str(SEGMENTATION_DATA_DIR / "data.yaml"),
    "epochs": 80,
    # 输入从 640 降到 512 后显存占用下降，batch 可以放大。
    "batch": 16,
    "imgsz": DEFAULT_SEGMENTATION_IMGSZ,
    "device": DEFAULT_DEVICE,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.05,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "weight_decay": 0.001,
    "patience": 15,
    "workers": 4,
    "cache": False,
    "amp": True,
    "seed": 0,
    "deterministic": True,
    "fliplr": 0.5,
    "flipud": 0.0,
    "scale": 0.20,
    "translate": 0.05,
    "mosaic": 0.5,
    "close_mosaic": 10,
    "project": str(RUNS_DIR / "segment"),
    "name": "recaptcha_seg_m1_640",
    "exist_ok": False,
    "pretrained": True,
    "verbose": True,
    "save_period": 2,
}

# ---------- 按模型区分的正式训练配置 ----------
# 分类数据不会预先强制缩放；imgsz 由 Ultralytics 训练器处理。
# 全部模型统一使用原生量级分辨率；数据目录不再随 imgsz 变化。
MODEL_TRAIN_PROFILES = {
    "yolo26n-cls.pt": {
        "epochs": 50,
        "batch": 64,
        "imgsz": DEFAULT_TRAIN_IMGSZ,
        "data": str(M2_DATA_DIR_320),
        "name": "recaptcha_v3_n_160",
    },
    "yolo26s-cls.pt": {
        "epochs": 50,
        "batch": 64,
        "imgsz": DEFAULT_TRAIN_IMGSZ,
        "data": str(M2_DATA_DIR_320),
        "name": "recaptcha_v3_s_160",
    },
    "yolo26m-cls.pt": {
        "epochs": 50,
        "batch": 64,
        "imgsz": DEFAULT_TRAIN_IMGSZ,
        "data": str(M2_DATA_DIR_320),
        "name": "recaptcha_v3_m_160",
    },
    "yolo26l-cls.pt": {
        "epochs": 50,
        "batch": 32,
        "imgsz": DEFAULT_TRAIN_IMGSZ,
        "data": str(M2_DATA_DIR_320),
        "name": "recaptcha_v3_l_160",
    },
    "yolo26x-cls.pt": {
        "epochs": 50,
        "batch": 32,
        "imgsz": DEFAULT_TRAIN_IMGSZ,
        "data": str(M2_DATA_DIR_320),
        "name": "recaptcha_v3_x_160",
    },
}


def training_data_for_imgsz(imgsz: int) -> Path:
    """返回正式训练数据目录。

    历史版本按 imgsz 切换 ``dataset_cls_m2_320`` / ``dataset_cls_m2_640``，
    但 ``m2_640`` 实际是指向 ``m2_320`` 的符号链接，两者数据完全相同。
    分辨率由训练器处理，与数据版本无关，因此这里不再随 imgsz 变化。
    """
    del imgsz  # 保留参数以兼容旧调用方。
    return M2_DATA_DIR_320


def read_training_imgsz(weight_path: str | Path) -> int | None:
    """从权重同目录的 ``model_meta.json`` 读取该模型的训练分辨率。

    推理分辨率必须与训练分辨率一致，否则会掉点。返回 ``None`` 表示
    缺少元数据，由调用方回退到类别默认值。
    """
    meta_path = Path(weight_path).parent / "model_meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    parameters = meta.get("训练参数")
    if not isinstance(parameters, dict):
        return None
    try:
        size = int(parameters.get("imgsz"))
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _model_scale(model: str | Path) -> str:
    """从 YOLO26 权重名提取 n/s/m/l/x 规模标识。"""
    model_name = Path(str(model)).name.lower()
    for scale in ("n", "s", "m", "l", "x"):
        if f"yolo26{scale}-cls" in model_name:
            return scale
    return "custom"


def training_profile_for_model(
    model: str | Path,
    *,
    imgsz: int | None = None,
) -> dict[str, object]:
    """按所选 YOLO26 模型返回独立训练默认值。

    本地权重路径使用文件名匹配；自定义模型沿用全局正式配置。
    """
    model_name = Path(str(model)).name.lower()
    profile = MODEL_TRAIN_PROFILES.get(model_name)
    result = dict(profile) if profile is not None else {
        "epochs": int(DEFAULT_TRAIN["epochs"]),
        "batch": int(DEFAULT_TRAIN["batch"]),
        "imgsz": int(DEFAULT_TRAIN["imgsz"]),
        "data": str(DEFAULT_TRAIN["data"]),
        "name": str(DEFAULT_TRAIN["name"]),
    }
    if imgsz is None:
        return result

    selected_size = int(imgsz)
    result["imgsz"] = selected_size
    result["data"] = str(training_data_for_imgsz(selected_size))
    result["name"] = f"recaptcha_v3_{_model_scale(model)}_{selected_size}"
    return result


def next_available_run_name(
    name: str,
    *,
    project_dir: str | Path | None = None,
    trained_dir: str | Path | None = None,
) -> str:
    """同时检查训练运行与已导出模型，生成不覆盖的运行名。

    基础名称已存在时依次尝试 ``_v1``、``_v2``、``_v3``。
    """
    requested = str(name).strip()
    if not requested:
        raise ValueError("运行名称不得为空")
    if Path(requested).name != requested or requested in {".", ".."}:
        raise ValueError("运行名称只能是单个目录名，不得包含路径")

    runs_root = Path(project_dir) if project_dir is not None else RUNS_DIR / "classify"
    models_root = Path(trained_dir) if trained_dir is not None else TRAINED_MODELS_DIR

    def occupied(candidate: str) -> bool:
        return (runs_root / candidate).exists() or (models_root / candidate).exists()

    if not occupied(requested):
        return requested

    match = re.fullmatch(r"(.+)_v\d+", requested)
    base_name = match.group(1) if match else requested
    version = 1
    while occupied(f"{base_name}_v{version}"):
        version += 1
    return f"{base_name}_v{version}"

# 冒烟训练：只验证管线能否跑通
SMOKE_TRAIN_OVERRIDES = {
    "epochs": 3,
    "batch": 32,
    "imgsz": 224,
    "name": "smoke",
    "patience": 3,
}

# ---------- 验证 / 预测默认 ----------
DEFAULT_VAL = {
    "data": str(DATA_DIR),
    "imgsz": 320,
    "batch": 32,
    "device": DEFAULT_DEVICE,
    "workers": 4,
    "split": "val",
    "project": str(RUNS_DIR / "classify"),
    "name": "val",
    "exist_ok": True,
}

DEFAULT_PREDICT = {
    "imgsz": 320,
    "device": DEFAULT_DEVICE,
}

prepare_cache_dir()


def resolve_best_weight(run_name: str = "exp") -> Path:
    """训练完成后 best.pt 的默认位置。"""
    return RUNS_DIR / "classify" / run_name / "weights" / "best.pt"


def resolve_last_weight(run_name: str = "exp") -> Path:
    return RUNS_DIR / "classify" / run_name / "weights" / "last.pt"


def resolve_model_reference(model: str | Path) -> str:
    """优先解析项目内模型路径，未找到时保留 Ultralytics 模型名称。"""
    path = Path(model)
    if path.is_file():
        return str(path.resolve())
    pretrained = PRETRAINED_DIR / path.name
    if pretrained.is_file():
        return str(pretrained)
    trained = TRAINED_MODELS_DIR / path.name
    if trained.is_file():
        return str(trained)
    return str(model)


def resolve_segmentation_model_reference(model: str | Path) -> str:
    """优先解析项目内分割模型路径，未找到时保留 Ultralytics 模型名称。"""
    path = Path(model)
    if path.is_file():
        return str(path.resolve())
    for directory in (SEGMENTATION_PRETRAINED_DIR, SEGMENTATION_TRAINED_DIR):
        direct = directory / path.name
        if direct.is_file():
            return str(direct)
    return str(model)


def resolve_default_weight(run_name: str = "exp") -> Path | None:
    """返回困难样本回归通过的推荐权重，其次才使用最新权重。"""
    del run_name  # 保留参数以兼容旧调用方。
    recommended = TRAINED_MODELS_DIR / RECOMMENDED_TRAINED_EXPERIMENT / "best.pt"
    if recommended.is_file():
        return recommended
    local_trained = sorted(
        TRAINED_MODELS_DIR.glob("*/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if local_trained:
        return local_trained[0]
    legacy = TRAINED_MODELS_DIR / "best.pt"
    return legacy if legacy.is_file() else None


def available_model_paths() -> list[Path]:
    """仅返回 models/trained 各实验目录下的 best.pt。"""
    trained = sorted(
        TRAINED_MODELS_DIR.glob("*/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if TRAINED_MODELS_DIR.is_dir() else []
    unique: list[Path] = []
    seen: set[str] = set()
    for path in trained:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def available_model_choices() -> list[str]:
    """返回 GUI 可选择的本地训练权重路径。"""
    return [str(path) for path in available_model_paths()]


def model_display_name(weight_path: str | Path) -> str:
    """根据实验目录与 model_meta.json 生成可区分的 GUI 名称。"""
    weight = Path(weight_path)
    experiment = weight.parent.name if weight.parent != TRAINED_MODELS_DIR else "未命名实验"
    meta_path = weight.parent / "model_meta.json"
    model_name = "未知模型"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source = Path(str(meta.get("模型", ""))).name
            model_name = source.removesuffix(".pt") or model_name
            experiment = str(meta.get("训练参数", {}).get("name") or experiment)
        except (OSError, json.JSONDecodeError):
            pass
    return f"{experiment}｜{model_name}｜最佳模型"


def available_segmentation_model_paths() -> list[Path]:
    """返回项目内可用的预训练和自定义分割权重。"""
    candidates: list[Path] = []
    if SEGMENTATION_TRAINED_DIR.is_dir():
        candidates.extend(SEGMENTATION_TRAINED_DIR.glob("*/best.pt"))
        candidates.extend(SEGMENTATION_TRAINED_DIR.glob("*.pt"))
    if SEGMENTATION_PRETRAINED_DIR.is_dir():
        candidates.extend(SEGMENTATION_PRETRAINED_DIR.glob("*.pt"))
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)


def available_segmentation_model_choices() -> list[str]:
    """返回 GUI 可选择的项目内分割权重路径。"""
    return [str(path) for path in available_segmentation_model_paths()]


def resolve_default_segmentation_weight() -> Path | None:
    """返回最新的本地分割权重；没有本地权重时返回 None。"""
    paths = available_segmentation_model_paths()
    return paths[0] if paths else None
