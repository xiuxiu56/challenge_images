"""融合与识别阈值的集中配置。

这些数字原本以模块级常量的形式散落在 ``segmentation/result_fusion.py``、
``training/model_service.py`` 和 ``segmentation/mask_grid.py`` 里，注释显示
它们是对着具体样本调出来的（「能恢复 483 的格子8，同时排除 486 的格子11」）。
问题不在于数值本身，而在于：

- 改一个数只能靠肉眼看单张图，看不到对全量样本的影响
- 数值与代码耦合，做 A/B 对照必须改源码
- 没有集中清单，新人无法判断哪些是可调参数、哪些是硬性约束

集中到一处之后，配合 ``tools/regression_eval.py`` 就能回答
「把这个阈值从 0.05 改成 0.03，会让多少张图的结果发生变化」。

覆盖方式：在项目根目录放 ``config/thresholds.yaml``，只写要改的键即可，
未写的沿用这里的默认值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# 配置文件位置：项目根目录下的 config/thresholds.yaml
_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = _ROOT / "config" / "thresholds.yaml"


@dataclass(frozen=True)
class MaskEvidenceThresholds:
    """mask 覆盖格子的基础判定。"""

    # 少于这个像素数的重叠一律视为噪声，不参与任何判定。
    min_overlap_pixels: int = 20
    # mask 占该格面积的最小比例。
    min_cell_ratio: float = 0.002
    # 该格占整个 mask 的最小比例，用于剔除跨格线的少量泄漏像素。
    min_mask_ratio: float = 0.10


@dataclass(frozen=True)
class InstanceValidationThresholds:
    """mask 实例的分类复核。"""

    # 实例覆盖格中最大分类分需达到此值，整个实例才被接受。
    classification_threshold: float = 0.80
    # 实例自身的分割置信度下限。
    confidence_threshold: float = 0.60
    # 常规分割完全漏检时，用于召回候选实例的低置信阈值。
    recovery_confidence: float = 0.05
    # 低置信实例被「救回」所需的强分类锚点。
    strong_classification_rescue: float = 0.95


@dataclass(frozen=True)
class MotorcycleThresholds:
    """摩托车实例的跨格边缘恢复。

    摩托车常出现「车轮落在相邻格」的情况，且近处主体与远处小目标可能同时
    被高置信检出，需要压制明显更小的孤立实例。
    """

    fringe_min_cell_ratio: float = 0.01
    fringe_min_mask_ratio: float = 0.02
    horizontal_expansion_cell_ratio: float = 0.05
    horizontal_expansion_mask_ratio: float = 0.03
    box_fill_cell_ratio: float = 0.005
    box_fill_mask_ratio: float = 0.005
    # 面积占主实例比例低于此值且只占一格的实例会被压制。
    dominant_min_area_ratio: float = 0.20


@dataclass(frozen=True)
class BusThresholds:
    """公共汽车车身的跨格连通扩散。"""

    # 主体格判定：占格面积达到此值即视为车身主要部分。
    major_cell_ratio: float = 0.05
    # 细长边缘：仅在与已接纳格共享水平 mask 边界时才加入。
    thin_cell_ratio: float = 0.002


@dataclass(frozen=True)
class TrafficLightThresholds:
    """红绿灯竖直方向的边缘恢复。"""

    fringe_classification_score: float = 0.25
    fringe_mask_ratio: float = 0.02
    fringe_cell_ratio: float = 0.002


@dataclass(frozen=True)
class ClassificationThresholds:
    """单标签分类链路的补丁参数。

    多标签模型上线后这一组会失去意义：每类独立打分不存在「被主类别压制」，
    也就不需要抑制阈值和局部裁剪复核。
    """

    # (目标类别, 完整格 Top-1 类别) → 该目标作为候选时的额外阈值。
    # Car 图块常把路面纹理压成第二候选 Crosswalk，需要更严格的门槛。
    candidate_suppressors: dict[str, float] = field(
        default_factory=lambda: {"Crosswalk|Car": 0.40}
    )
    # 允许使用局部裁剪复核的目标类别。
    multiview_targets: list[str] = field(default_factory=lambda: ["Crosswalk"])
    # 只有完整格被这些大目标压制时，才允许局部复核介入。
    multiview_suppressors: list[str] = field(
        default_factory=lambda: ["Car", "Bridge", "Bus", "Crosswalk"]
    )
    # 局部裁剪视角：从格子顶部裁掉的比例。
    multiview_crop_ratios: list[float] = field(default_factory=lambda: [0.20, 0.35])
    # 中央裁剪保留比例。
    multiview_center_inset: float = 0.10


@dataclass(frozen=True)
class ContinuousGridThresholds:
    """4×4 连续照片的形态学收敛。"""

    # 孤立单格（四邻接下没有任何相邻选中格）需要达到的分类概率。
    # 成片格子有连通性背书，孤立格只能靠自身证据；低于此值判为误检。
    # 仅在同时存在更大连通块时才生效，避免整图只有一个小目标时误删。
    isolated_min_score: float = 0.70


@dataclass(frozen=True)
class FusionWeakEvidenceThresholds:
    """分割未产生实例明细时的旧格式弱证据融合。"""

    weak_classification_threshold: float = 0.10
    strong_mask_cell_ratio: float = 0.01


@dataclass(frozen=True)
class Thresholds:
    """全部可调阈值。"""

    mask_evidence: MaskEvidenceThresholds = field(default_factory=MaskEvidenceThresholds)
    instance_validation: InstanceValidationThresholds = field(
        default_factory=InstanceValidationThresholds
    )
    motorcycle: MotorcycleThresholds = field(default_factory=MotorcycleThresholds)
    bus: BusThresholds = field(default_factory=BusThresholds)
    traffic_light: TrafficLightThresholds = field(default_factory=TrafficLightThresholds)
    classification: ClassificationThresholds = field(default_factory=ClassificationThresholds)
    continuous_grid: ContinuousGridThresholds = field(
        default_factory=ContinuousGridThresholds
    )
    weak_evidence: FusionWeakEvidenceThresholds = field(
        default_factory=FusionWeakEvidenceThresholds
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def candidate_suppressor(self, target_class: str, top1_class: str) -> float | None:
        """返回 (目标, Top-1) 组合的额外候选阈值。"""
        return self.classification.candidate_suppressors.get(f"{target_class}|{top1_class}")


def _merge_section(section: Any, overrides: dict[str, Any]) -> Any:
    """把 YAML 覆盖合并进一个 dataclass 分组。"""
    known = {item.name for item in fields(section)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"未知的阈值键：{sorted(unknown)}")
    values = {item.name: getattr(section, item.name) for item in fields(section)}
    values.update({key: overrides[key] for key in overrides})
    return type(section)(**values)


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """载入阈值配置；缺少覆盖文件时返回默认值。"""
    config_path = Path(path) if path is not None else THRESHOLDS_PATH
    base = Thresholds()
    if not config_path.is_file():
        return base
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print(f"[阈值配置] 未安装 PyYAML，忽略 {config_path}，继续使用默认值。")
        return base
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        print(f"[阈值配置] 读取失败，继续使用默认值：{error}")
        return base
    if not isinstance(payload, dict):
        return base

    known_sections = {item.name for item in fields(base)}
    unknown = set(payload) - known_sections
    if unknown:
        raise ValueError(f"未知的阈值分组：{sorted(unknown)}")
    merged = {item.name: getattr(base, item.name) for item in fields(base)}
    for name, overrides in payload.items():
        if isinstance(overrides, dict):
            merged[name] = _merge_section(merged[name], overrides)
    return Thresholds(**merged)


# 全局单例：模块级常量从这里取值，保持既有调用方式不变。
THRESHOLDS = load_thresholds()


def dump_default_yaml() -> str:
    """生成带注释的默认配置模板，供用户复制为 config/thresholds.yaml。"""
    lines = [
        "# 融合与识别阈值覆盖文件。",
        "# 只需写出要修改的键，其余沿用代码内默认值。",
        "# 修改后用 tools/regression_eval.py 对照全量样本，不要只看单张图。",
        "",
    ]
    for section in fields(Thresholds()):
        lines.append(f"{section.name}:")
        instance = getattr(Thresholds(), section.name)
        for item in fields(instance):
            value = getattr(instance, item.name)
            lines.append(f"  {item.name}: {value!r}")
        lines.append("")
    return "\n".join(lines)
