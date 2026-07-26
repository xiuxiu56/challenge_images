"""统一识别策略、执行引擎和网页点击计划。"""

from .click_plan import ClickPlan, ClickSettings, build_click_plan
from .engine import RecognitionEngine, RecognitionResult, format_recognition_report
from .policy import (
    ENGINE_MODE_LABELS,
    PARAMETER_PRESET_LABELS,
    RecognitionParameters,
    RecognitionRoute,
    parameters_for,
    resolve_recognition_route,
)

__all__ = [
    "ClickPlan",
    "ClickSettings",
    "ENGINE_MODE_LABELS",
    "PARAMETER_PRESET_LABELS",
    "RecognitionEngine",
    "RecognitionParameters",
    "RecognitionResult",
    "RecognitionRoute",
    "build_click_plan",
    "format_recognition_report",
    "parameters_for",
    "resolve_recognition_route",
]
