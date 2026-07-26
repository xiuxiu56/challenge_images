"""用预训练分割模型批量预标注，把人工工作从「画」降级为「改」。

4×4 连续照片必须靠整图实例分割才能正确建模：把一辆公交车切成 6 块，
每块单独看都不像公交车。但自训分割模型需要 YOLO 多边形标签，纯手工标注
200~500 张图是整个项目最大的一次性成本。

本模块用 COCO 预训练分割模型先跑一遍，把能识别的实例转成标准 YOLO 多边形
标签，人工只需修正边界和补充 COCO 不覆盖的类别。

COCO 覆盖情况（实测 yolo26m-seg）：

已覆盖
    Car、Bus、Bicycle、Motorcycle、Traffic Light、Hydrant
未覆盖，必须人工标注
    Bridge、Chimney、Crosswalk、Mountain、Palm、Stair、Tractor

关键：预标注会标出图中**所有**可识别类别，而不只是本轮挑战的目标类别。
只标目标会让画面里其他类别变成隐式负样本，反而污染训练。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..category_map import normalize_dataset_class
from ..segmentation.category_map import segmentation_category_key

# data.yaml.example 中的类别顺序，标签里的类别编号必须与之一致。
SEGMENTATION_CLASSES = [
    "Bicycle",
    "Bridge",
    "Bus",
    "Car",
    "Chimney",
    "Crosswalk",
    "Hydrant",
    "Motorcycle",
    "Mountain",
    "Palm",
    "Stair",
    "Tractor",
    "Traffic Light",
]

# 多边形点数上限：过密的轮廓会让标签文件膨胀且不便人工修正。
MAX_POLYGON_POINTS = 60
MIN_POLYGON_POINTS = 3
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class PrelabelReport:
    """预标注结果统计。"""

    output: Path
    images_total: int = 0
    images_with_labels: int = 0
    images_empty: int = 0
    instances_per_class: dict[str, int] = field(default_factory=dict)
    train_images: int = 0
    val_images: int = 0
    uncovered_classes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "输出目录": str(self.output),
            "图片总数": self.images_total,
            "含预标注的图片": self.images_with_labels,
            "空标签图片": self.images_empty,
            "训练图片": self.train_images,
            "验证图片": self.val_images,
            "逐类实例数": dict(sorted(self.instances_per_class.items())),
            "预训练模型未覆盖需人工标注": self.uncovered_classes,
        }


def _normalise_polygon(
    points: Iterable[Iterable[float]],
    width: int,
    height: int,
) -> list[float]:
    """把像素多边形转成 YOLO 归一化坐标，并按需抽稀。"""
    coordinates = [(float(x), float(y)) for x, y in points]
    if len(coordinates) < MIN_POLYGON_POINTS:
        return []
    if len(coordinates) > MAX_POLYGON_POINTS:
        step = len(coordinates) / MAX_POLYGON_POINTS
        coordinates = [coordinates[min(len(coordinates) - 1, int(index * step))] for index in range(MAX_POLYGON_POINTS)]
    flat: list[float] = []
    for x, y in coordinates:
        flat.append(min(1.0, max(0.0, x / max(width, 1))))
        flat.append(min(1.0, max(0.0, y / max(height, 1))))
    return flat


def _target_class_from_directory(directory: Path) -> str | None:
    """归档目录名是中文类别，转成数据集类别名。"""
    return normalize_dataset_class(directory.name)


def collect_challenge_images(
    source: str | Path,
    *,
    challenge_types: Iterable[str] = ("multicaptcha",),
    limit_per_class: int | None = None,
) -> list[tuple[Path, str | None]]:
    """收集待标注的完整挑战图，返回 (图片路径, 挑战目标类别)。"""
    root = Path(source)
    collected: list[tuple[Path, str | None]] = []
    for challenge_type in challenge_types:
        type_dir = root / challenge_type
        if not type_dir.is_dir():
            continue
        for class_dir in sorted(p for p in type_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            target = _target_class_from_directory(class_dir)
            files = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMG_SUFFIXES
            )
            if limit_per_class is not None:
                files = files[:limit_per_class]
            collected.extend((path, target) for path in files)
    return collected


def uncovered_classes(model_names: dict[int, str]) -> list[str]:
    """返回预训练模型无法提供预标注、必须人工标注的类别。"""
    available = {
        segmentation_category_key(label)
        for label in model_names.values()
    }
    return [name for name in SEGMENTATION_CLASSES if name not in available]


def build_segmentation_prelabels(
    source: str | Path,
    output: str | Path,
    *,
    weights: str | Path = "yolo26m-seg.pt",
    device: str | None = None,
    challenge_types: Iterable[str] = ("multicaptcha",),
    limit_per_class: int | None = 40,
    confidence: float = 0.30,
    imgsz: int = 512,
    val_fraction: float = 0.2,
    keep_empty: bool = True,
    overwrite: bool = False,
) -> PrelabelReport:
    """批量生成 YOLO 多边形预标注。

    ``keep_empty`` 保留没有检出的图片并写空标签：纯负样本对抑制误检很重要，
    例如把汽车局部识别成摩托车、把路灯识别成红绿灯。
    """
    from ..segmentation.model_service import SegmentationModelService

    output_root = Path(output)
    generated = [output_root / "images", output_root / "labels"]
    if any(directory.exists() and any(directory.iterdir()) for directory in generated):
        if not overwrite:
            raise FileExistsError(f"输出目录已有预标注，请换一个名称或设置 overwrite：{output_root}")
        # 只清理本模块生成的目录；README、data.yaml.example 等模板文件必须保留。
        for directory in generated:
            if directory.exists():
                shutil.rmtree(directory)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    service = SegmentationModelService()
    service.load(weights, device)
    class_index = {name: position for position, name in enumerate(SEGMENTATION_CLASSES)}

    report = PrelabelReport(
        output=output_root,
        uncovered_classes=uncovered_classes(service.class_names),
    )

    images = collect_challenge_images(
        source, challenge_types=challenge_types, limit_per_class=limit_per_class
    )
    report.images_total = len(images)

    from PIL import Image

    for position, (path, _target) in enumerate(images):
        # 按固定间隔划分验证集，保证不同类别都进入 val。
        split = "val" if val_fraction > 0 and position % max(1, round(1 / val_fraction)) == 0 else "train"
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        width, height = image.size
        lines = _predict_polygons(
            service, image, confidence=confidence, imgsz=imgsz, class_index=class_index
        )
        if not lines and not keep_empty:
            continue

        stem = f"{path.parent.name}_{path.stem}".replace(" ", "_")
        target_image = output_root / "images" / split / f"{stem}{path.suffix.lower()}"
        target_label = output_root / "labels" / split / f"{stem}.txt"
        shutil.copy2(path, target_image)
        target_label.write_text("\n".join(line for _, line in lines), encoding="utf-8")

        if lines:
            report.images_with_labels += 1
            for class_name, _ in lines:
                report.instances_per_class[class_name] = (
                    report.instances_per_class.get(class_name, 0) + 1
                )
        else:
            report.images_empty += 1
        if split == "train":
            report.train_images += 1
        else:
            report.val_images += 1

    _write_data_yaml(output_root)
    (output_root / "prelabel_report.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _predict_polygons(
    service: Any,
    image: Any,
    *,
    confidence: float,
    imgsz: int,
    class_index: dict[str, int],
) -> list[tuple[str, str]]:
    """对整图推理，返回 [(类别名, YOLO 标签行)]。"""
    results = service.model.predict(
        source=image,
        imgsz=int(imgsz),
        conf=float(confidence),
        device=service.device,
        verbose=False,
        stream=False,
    )
    if not results:
        return []
    result = results[0]
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return []
    polygons = getattr(masks, "xy", None) or []
    width, height = image.size

    lines: list[tuple[str, str]] = []
    for position, polygon in enumerate(polygons):
        try:
            raw_class = int(boxes.cls[position].item())
        except (AttributeError, IndexError, ValueError):
            continue
        label = service.class_names.get(raw_class, str(raw_class))
        dataset_class = segmentation_category_key(label)
        if dataset_class not in class_index:
            continue
        flat = _normalise_polygon(polygon, width, height)
        if not flat:
            continue
        coordinates = " ".join(f"{value:.6f}" for value in flat)
        lines.append((dataset_class, f"{class_index[dataset_class]} {coordinates}"))
    return lines


def _write_data_yaml(root: Path) -> Path:
    """生成可直接训练的 data.yaml。"""
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(SEGMENTATION_CLASSES))
    content = (
        "# 由 segmentation_prelabel 自动生成。\n"
        "# 预标注仅供人工修正起点，训练前请先复核多边形边界。\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"names:\n{names}\n"
    )
    path = root / "data.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def format_prelabel_report(report: PrelabelReport) -> str:
    """生成可直接打印的中文预标注报告。"""
    lines = [
        f"输出目录: {report.output}",
        f"图片总数: {report.images_total}（训练 {report.train_images} / 验证 {report.val_images}）",
        f"含预标注: {report.images_with_labels}，空标签负样本: {report.images_empty}",
        "",
        "逐类预标注实例数:",
    ]
    if report.instances_per_class:
        for name, count in sorted(report.instances_per_class.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:<16}{count:>6}")
    else:
        lines.append("  无")
    if report.uncovered_classes:
        lines.extend(
            [
                "",
                "预训练模型未覆盖，必须人工标注的类别:",
                "  " + "、".join(report.uncovered_classes),
            ]
        )
    return "\n".join(lines)
