"""
reCAPTCHA 类别 / 挑战类型映射表。

三层：
  mid (/m/xxx)  ←→  官方中英文词表
  数据集文件夹名  ←→  mid / 中英文（训练标签）
  challenge_type  ←→  中文说明

预测时模型吐出的是文件夹名（如 Hydrant / Traffic Light），
用本模块转成 mid + 中文，方便对接 recaptcha_solver。
"""

from __future__ import annotations

from typing import Any

# ---------- 官方 mid 词表（与 recaptcha_solver 一致） ----------
DEFAULT_CATEGORY_ZH: dict[str, str] = {
    "/m/0pg52": "出租车",
    "/m/01bjv": "公共汽车",
    "/m/02yvhj": "校车",
    "/m/04_sv": "摩托车",
    "/m/013xlm": "拖拉机",
    "/m/01jk_4": "烟囱",
    "/m/014xcs": "人行横道",
    "/m/015qff": "红绿灯",
    "/m/0199g": "自行车",
    "/m/015qbp": "停车计时器",
    "/m/0k4j": "车",
    "/m/015kr": "桥",
    "/m/019jd": "船",
    "/m/0cdl1": "棕榈树",
    "/m/09d_r": "山丘",
    "/m/01pns0": "消防栓",
    "/m/01lynh": "楼梯",
}

DEFAULT_CATEGORY_EN: dict[str, str] = {
    "/m/0pg52": "Taxi",
    "/m/01bjv": "Bus",
    "/m/02yvhj": "School bus",
    "/m/04_sv": "Motorcycle",
    "/m/013xlm": "Tractor",
    "/m/01jk_4": "Chimney",
    "/m/014xcs": "Crosswalk",
    "/m/015qff": "Traffic light",
    "/m/0199g": "Bicycle",
    "/m/015qbp": "Parking meter",
    "/m/0k4j": "Car",
    "/m/015kr": "Bridge",
    "/m/019jd": "Boat",
    "/m/0cdl1": "Palm tree",
    "/m/09d_r": "Mountain hill",
    "/m/01pns0": "Fire hydrant",
    "/m/01lynh": "Stairs",
}

CHALLENGE_TYPE_MAP: dict[str, str] = {
    "imageselect": "标准图像选择验证码",
    "tileselect": "基于图块的动态图像挑战",
    "dynamic": "动态挑战，选择后图块会重新加载",
    "multicaptcha": "多步骤验证码挑战",
    "audio": "基于音频的无障碍挑战",
    "nocaptcha": "仅复选框验证流程",
    "doscaptcha": "如果存在滥用行为，则拒绝验证码验证",
}

IMAGE_CHALLENGE_TYPES = frozenset(
    {"imageselect", "tileselect", "dynamic", "multicaptcha"}
)
UNKNOWN_CATEGORY = "未知类别"
UNKNOWN_CHALLENGE_TYPE = "unknown"

# ---------- 本数据集文件夹名 → mid ----------
# 文件夹名来自 dataset_cls_full_57k/{train,val}/<Class>/
# 注意：与官方英文略有差异（Hydrant / Palm / Stair / Mountain / Traffic Light）
DATASET_CLASS_TO_MID: dict[str, str | None] = {
    "Bicycle": "/m/0199g",
    "Bridge": "/m/015kr",
    "Bus": "/m/01bjv",
    "Car": "/m/0k4j",
    "Chimney": "/m/01jk_4",
    "Crosswalk": "/m/014xcs",
    "Hydrant": "/m/01pns0",  # Fire hydrant
    "Motorcycle": "/m/04_sv",
    "Mountain": "/m/09d_r",  # Mountain hill
    "Other": None,  # 负样本/杂类，无 mid
    "Palm": "/m/0cdl1",  # Palm tree
    "Stair": "/m/01lynh",  # Stairs
    "Tractor": "/m/013xlm",
    "Traffic Light": "/m/015qff",
}

# 官方词表里有、本训练集没有单独目录的类（推理侧可能遇到 mid）
MID_NOT_IN_DATASET = {
    "/m/0pg52",  # Taxi → 通常并到 Car
    "/m/02yvhj",  # School bus → 通常并到 Bus
    "/m/015qbp",  # Parking meter
    "/m/019jd",  # Boat
}

# 可选：挑战 mid 合并到训练类（对接解题器时用）
MID_FALLBACK_TO_DATASET_CLASS: dict[str, str] = {
    "/m/0pg52": "Car",  # Taxi → Car
    "/m/02yvhj": "Bus",  # School bus → Bus
}


def _norm_key(s: str) -> str:
    """统一比较：去空格、小写、去连字符差异。"""
    return (
        str(s)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
    )


# 反向索引：任意别名 → 规范数据集类名
def _build_alias_to_dataset_class() -> dict[str, str]:
    out: dict[str, str] = {}
    for cls, mid in DATASET_CLASS_TO_MID.items():
        out[_norm_key(cls)] = cls
        # 文件夹名本身
        out[_norm_key(cls.replace(" ", ""))] = cls
        if mid and mid in DEFAULT_CATEGORY_EN:
            en = DEFAULT_CATEGORY_EN[mid]
            out[_norm_key(en)] = cls
            out[_norm_key(en.replace(" ", ""))] = cls
        if mid and mid in DEFAULT_CATEGORY_ZH:
            out[_norm_key(DEFAULT_CATEGORY_ZH[mid])] = cls
        # mid 本身也能当 key
        if mid:
            out[_norm_key(mid)] = cls

    # 额外常见别名
    extras = {
        "fire hydrant": "Hydrant",
        "hydrant": "Hydrant",
        "palm tree": "Palm",
        "palm": "Palm",
        "stairs": "Stair",
        "stair": "Stair",
        "traffic light": "Traffic Light",
        "trafficlight": "Traffic Light",
        "mountain hill": "Mountain",
        "mountain": "Mountain",
        "hill": "Mountain",
        "cross walk": "Crosswalk",
        "zebra crossing": "Crosswalk",
        "bus": "Bus",
        "school bus": "Bus",
        "taxi": "Car",
        "car": "Car",
        "other": "Other",
        "bicycles": "Bicycle",
        "bicycle": "Bicycle",
        "bridges": "Bridge",
        "bridge": "Bridge",
        "buses": "Bus",
        "cars": "Car",
        "chimneys": "Chimney",
        "crosswalks": "Crosswalk",
        "cross walks": "Crosswalk",
        "fire hydrants": "Hydrant",
        "hydrants": "Hydrant",
        "motorcycles": "Motorcycle",
        "mountain hills": "Mountain",
        "palm trees": "Palm",
        "parking meter": "Other",
        "parking meters": "Other",
        "boats": "Other",
        "stairways": "Stair",
        "traffic lights": "Traffic Light",
        "tractors": "Tractor",
    }
    for k, v in extras.items():
        out[_norm_key(k)] = v
    return out


_ALIAS_TO_CLASS = _build_alias_to_dataset_class()

# 官方中文 / 英文 → mid（含训练集没有的类）
_ZH_TO_MID: dict[str, str] = {_norm_key(v): k for k, v in DEFAULT_CATEGORY_ZH.items()}
_EN_TO_MID: dict[str, str] = {_norm_key(v): k for k, v in DEFAULT_CATEGORY_EN.items()}


def mid_to_zh(mid: str) -> str:
    return DEFAULT_CATEGORY_ZH.get(mid, UNKNOWN_CATEGORY)


def mid_to_en(mid: str) -> str:
    return DEFAULT_CATEGORY_EN.get(mid, mid)


def challenge_type_zh(challenge_type: str | None) -> tuple[str, str]:
    """
    返回 (规范 key, 中文说明)。
    未知类型: (unknown 或 safe_key, 说明)。
    """
    raw = (challenge_type or "").strip()
    key = raw.lower() if raw else UNKNOWN_CHALLENGE_TYPE
    if key in CHALLENGE_TYPE_MAP:
        return key, CHALLENGE_TYPE_MAP[key]
    if not raw:
        return UNKNOWN_CHALLENGE_TYPE, "未知挑战类型"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return safe or UNKNOWN_CHALLENGE_TYPE, f"未收录挑战类型: {raw}"


def is_image_challenge(challenge_type: str | None) -> bool:
    key, _ = challenge_type_zh(challenge_type)
    return key in IMAGE_CHALLENGE_TYPES


def mid_to_dataset_class(mid: str | None) -> str | None:
    """挑战 mid → 训练类名（含 Taxi→Car 等回退）。"""
    if not mid:
        return None
    if mid in MID_FALLBACK_TO_DATASET_CLASS:
        return MID_FALLBACK_TO_DATASET_CLASS[mid]
    for cls, m in DATASET_CLASS_TO_MID.items():
        if m == mid:
            return cls
    return None


def class_to_mid(label: str | None) -> str | None:
    """
    任意标签 → 官方 mid。
    优先级：mid 本身 → 官方中/英 → 数据集文件夹名 → 别名。
    注意：「出租车」保持 /m/0pg52，不会被回退成 Car 的 mid。
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    # 1) 直接是 mid
    if s in DEFAULT_CATEGORY_ZH:
        return s
    n = _norm_key(s)
    # 2) 官方中英文（保留 Taxi/校车 等原始 mid）
    if n in _ZH_TO_MID:
        return _ZH_TO_MID[n]
    if n in _EN_TO_MID:
        return _EN_TO_MID[n]
    # 3) 数据集文件夹名
    if s in DATASET_CLASS_TO_MID:
        return DATASET_CLASS_TO_MID[s]
    # 4) 别名（Hydrant / fire hydrant / Traffic Light …）
    cls = _ALIAS_TO_CLASS.get(n)
    if cls is not None:
        return DATASET_CLASS_TO_MID.get(cls)
    return None


def normalize_dataset_class(label: str | None) -> str | None:
    """
    任意标签 → 数据集文件夹名（训练类）。
    Taxi/校车等会回退到 Car/Bus；无对应类返回 None。
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    if s in DATASET_CLASS_TO_MID:
        return s
    hit = _ALIAS_TO_CLASS.get(_norm_key(s))
    if hit:
        return hit
    # mid / 官方中英文 → 训练类（含回退）
    mid = class_to_mid(s)
    if mid:
        return mid_to_dataset_class(mid)
    return None


def class_to_zh(label: str | None) -> str:
    mid = class_to_mid(label)
    if mid:
        return mid_to_zh(mid)
    cls = normalize_dataset_class(label)
    if cls == "Other":
        return "其他"
    if cls:
        return cls
    return UNKNOWN_CATEGORY


def class_to_en(label: str | None) -> str:
    mid = class_to_mid(label)
    if mid:
        return mid_to_en(mid)
    cls = normalize_dataset_class(label)
    if cls == "Other":
        return "Other"
    if cls:
        return cls
    return "Unknown"


def describe_label(label: str | None) -> dict[str, Any]:
    """
    统一描述一条预测标签。
    mid 保留官方语义；dataset_class 是训练时能对上的文件夹名。
    """
    mid = class_to_mid(label)
    cls = normalize_dataset_class(label)
    return {
        "raw": label,
        "dataset_class": cls,
        "mid": mid,
        "zh": class_to_zh(label),
        "en": class_to_en(label),
        "in_official_vocab": mid is not None and mid in DEFAULT_CATEGORY_ZH,
        "fallback": bool(
            mid and mid in MID_FALLBACK_TO_DATASET_CLASS and cls is not None
        ),
    }


def format_predict_line(
    path_name: str,
    label: str,
    conf: float,
    *,
    with_mid: bool = True,
) -> str:
    """预测日志一行：文件名 + 类名 + 中文 + mid + conf。"""
    info = describe_label(label)
    zh = info["zh"]
    mid = info["mid"]
    mid_part = f"  mid={mid}" if with_mid and mid else ""
    other = "" if info["dataset_class"] else "  [未映射]"
    return f"  {path_name}: {label}（{zh}）{mid_part}  conf={conf:.4f}{other}"


def dump_mapping_table() -> str:
    """打印给人看的对照表。"""
    lines: list[str] = []
    lines.append("======== 数据集类 ↔ mid ↔ 中/英 ========")
    lines.append(f"{'数据集类':<16} {'mid':<14} {'中文':<10} {'官方英文'}")
    lines.append("-" * 56)
    for cls in sorted(DATASET_CLASS_TO_MID.keys()):
        mid = DATASET_CLASS_TO_MID[cls]
        if mid is None:
            lines.append(f"{cls:<16} {'—':<14} {'其他':<10} Other")
        else:
            lines.append(
                f"{cls:<16} {mid:<14} {mid_to_zh(mid):<10} {mid_to_en(mid)}"
            )
    lines.append("")
    lines.append("官方有、训练集无独立目录:")
    for mid in sorted(MID_NOT_IN_DATASET):
        lines.append(f"  {mid}  {mid_to_zh(mid)} / {mid_to_en(mid)}")
        fb = MID_FALLBACK_TO_DATASET_CLASS.get(mid)
        if fb:
            lines.append(f"    → 回退训练类: {fb}")
    lines.append("")
    lines.append("======== 挑战类型 ========")
    for k, v in CHALLENGE_TYPE_MAP.items():
        img = " [图像]" if k in IMAGE_CHALLENGE_TYPES else ""
        lines.append(f"  {k:<14} {v}{img}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(dump_mapping_table())
    print()
    for s in ["Hydrant", "Traffic Light", "Palm", "fire hydrant", "/m/0k4j", "出租车"]:
        print(s, "→", describe_label(s))
