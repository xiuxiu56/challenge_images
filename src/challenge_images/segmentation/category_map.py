"""分类数据集类别与分割模型类别的统一映射。"""

from __future__ import annotations

from ..category_map import normalize_dataset_class


SEGMENTATION_CATEGORY_ALIASES = {
    "car": "Car",
    "cars": "Car",
    "taxi": "Car",
    "bus": "Bus",
    "buses": "Bus",
    "school bus": "Bus",
    "bicycle": "Bicycle",
    "bicycles": "Bicycle",
    "motorcycle": "Motorcycle",
    "motorcycles": "Motorcycle",
    "traffic light": "Traffic Light",
    "traffic lights": "Traffic Light",
    "fire hydrant": "Hydrant",
    "fire hydrants": "Hydrant",
    "hydrant": "Hydrant",
    "crosswalk": "Crosswalk",
    "crosswalks": "Crosswalk",
    "pedestrian crossing": "Crosswalk",
    "bridge": "Bridge",
    "bridges": "Bridge",
    "chimney": "Chimney",
    "chimneys": "Chimney",
    "palm": "Palm",
    "palm tree": "Palm",
    "palm trees": "Palm",
    "stairs": "Stair",
    "stair": "Stair",
    "tractor": "Tractor",
    "tractors": "Tractor",
    "mountain": "Mountain",
    "mountain hill": "Mountain",
    "boat": "Boat",
    "boats": "Boat",
    "parking meter": "Parking meter",
    "parking meters": "Parking meter",
}


def segmentation_category_key(label: str | None) -> str | None:
    """把预训练或自定义分割标签转换为统一比较键。"""
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    normalized = normalize_dataset_class(text)
    if normalized not in (None, "Other"):
        return normalized
    key = " ".join(text.lower().replace("-", " ").replace("_", " ").split())
    return SEGMENTATION_CATEGORY_ALIASES.get(key)
