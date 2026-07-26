"""GUI 识别请求与结果的统一载体。

离线验证页、在线采集页和分割融合页此前各自维护一套平行状态：

    self.image        / self.predictions        / self.all_predictions
    self.online_image / self.online_predictions / self.online_all_predictions
    self.fusion_image / self.fusion_indices     / self.fusion_preview_image

三套状态导致 ``_recognize`` / ``_recognize_online`` / ``_recognize_fusion``
有约九成重复：都是「取图 → 取目标 → 取网格 → 取引擎 → 调用识别 →
更新状态 → 渲染 → 拼报告」，差别只在数据源和报告开头几行。
加一个功能要改三处。

把「一次识别需要的全部输入」收敛成一个不可变请求对象之后，
三个页面只需各自实现「怎么构造请求」和「报告开头写什么」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..grid.grid_engine import GridSpec
from ..recognition.policy import RecognitionParameters

# 三个功能页的稳定标识，用于把异步结果路由回正确的页面。
SOURCE_OFFLINE = "offline"
SOURCE_ONLINE = "online"
SOURCE_FUSION = "fusion"

SOURCE_LABELS = {
    SOURCE_OFFLINE: "识别与标注",
    SOURCE_ONLINE: "在线采集",
    SOURCE_FUSION: "分割与融合",
}


@dataclass(frozen=True)
class RecognitionRequest:
    """一次识别所需的全部输入。

    不可变：请求一旦提交给工作线程，UI 线程继续操作控件也不会影响
    正在进行的这一次识别。
    """

    source: str
    image: Any
    challenge_type: str
    spec: GridSpec
    target_class: str
    requested_mode: str
    parameters: RecognitionParameters
    image_key: str | None = None
    # 报告开头的来源描述，由各页面自行准备。
    header: str = ""

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)


@dataclass(frozen=True)
class RecognitionOutcome:
    """一次识别的完整产出，回传给发起页面。"""

    request: RecognitionRequest
    result: Any

    @property
    def source(self) -> str:
        return self.request.source

    @property
    def indices(self) -> list[int]:
        return list(getattr(self.result, "indices", []) or [])
