"""浏览器人工验证与在线样本导入。"""

from .browser_session import (
    AutomatedQueryRestrictionError,
    BrowserSession,
    CapturedChallenge,
    detect_automated_query_restriction,
    detect_image_ext,
)
from .capture_service import OnlineCaptureService, OnlineSample, parse_reload_response
from .click_geometry import Rect, TileLayout, build_tile_layout, cell_center, grid_from_pmeta
from .online_session import OnlineRoundResult, OnlineSolveSession

__all__ = [
    "BrowserSession",
    "CapturedChallenge",
    "AutomatedQueryRestrictionError",
    "OnlineCaptureService",
    "OnlineRoundResult",
    "OnlineSample",
    "OnlineSolveSession",
    "Rect",
    "TileLayout",
    "build_tile_layout",
    "cell_center",
    "detect_image_ext",
    "detect_automated_query_restriction",
    "grid_from_pmeta",
    "parse_reload_response",
]
