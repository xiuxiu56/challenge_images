"""根据 3×3 / 4×4 题型生成稳定的网页点击计划。"""

from __future__ import annotations

from dataclasses import dataclass

from ..grid.grid_engine import GridSpec


@dataclass(frozen=True)
class ClickSettings:
    """在线点击的公共配置。"""

    delay_ms: int = 220
    dynamic_wait_ms: int = 8_000
    auto_verify: bool = False
    maximum_selected_ratio: float = 0.90


@dataclass(frozen=True)
class ClickPlan:
    """识别结果转换后的实际点击动作。"""

    strategy: str
    strategy_label: str
    indices: list[int]
    delay_ms: int
    watch_after_ms: int
    click_verify: bool
    blocked: bool = False
    reason: str = ""


def build_click_plan(
    challenge_type: str,
    spec: GridSpec,
    indices: list[int],
    settings: ClickSettings | None = None,
) -> ClickPlan:
    """构建静态批量、动态逐格或 4×4 连续图点击计划。"""
    settings = settings or ClickSettings()
    clean = sorted({int(index) for index in indices if 0 <= int(index) < spec.count})
    ratio = len(clean) / spec.count if spec.count else 0.0
    if clean and ratio > float(settings.maximum_selected_ratio):
        return ClickPlan(
            strategy="blocked",
            strategy_label="异常结果保护",
            indices=clean,
            delay_ms=int(settings.delay_ms),
            watch_after_ms=0,
            click_verify=False,
            blocked=True,
            reason=(
                f"识别命中 {len(clean)}/{spec.count} 格，超过 "
                f"{settings.maximum_selected_ratio:.0%} 保护阈值"
            ),
        )

    challenge = str(challenge_type or "").strip().lower()
    if spec.count == 9 and challenge == "dynamic":
        return ClickPlan(
            strategy="dynamic_sequential",
            strategy_label="3×3 动态逐格点击并等待换图",
            indices=clean,
            delay_ms=int(settings.delay_ms),
            watch_after_ms=int(settings.dynamic_wait_ms),
            click_verify=bool(settings.auto_verify),
            reason="动态题按格保留点击索引与替换图的对应关系",
        )
    if spec.count == 16:
        return ClickPlan(
            strategy="continuous_batch",
            strategy_label="4×4 连续图批量点击",
            indices=clean,
            delay_ms=int(settings.delay_ms),
            watch_after_ms=0,
            click_verify=bool(settings.auto_verify),
            reason="4×4 使用整图投影结果，按 DOM 图块批量点击",
        )
    return ClickPlan(
        strategy="static_batch",
        strategy_label="3×3 静态批量点击",
        indices=clean,
        delay_ms=int(settings.delay_ms),
        watch_after_ms=0,
        click_verify=bool(settings.auto_verify),
        reason="静态独立图按行优先索引批量点击",
    )
