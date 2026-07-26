"""GUI 识别链路的无头测试。

重点验证两件事：

1. 识别不在 UI 线程执行——原实现靠 WaitCursor + processEvents 硬扛，
   4×4 图会让窗口冻结数秒。
2. 三个功能页共用同一条请求/回调链路，不再各写一遍识别流程。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from challenge_images.gui.state import (  # noqa: E402
    SOURCE_FUSION,
    SOURCE_OFFLINE,
    SOURCE_ONLINE,
    RecognitionOutcome,
    RecognitionRequest,
)
from challenge_images.gui.workers import RecognitionWorker  # noqa: E402
from challenge_images.grid.grid_engine import GridSpec  # noqa: E402
from challenge_images.recognition.policy import RecognitionParameters  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _request(source=SOURCE_OFFLINE, target="Car"):
    return RecognitionRequest(
        source=source,
        image=object(),
        challenge_type="dynamic",
        spec=GridSpec(3, 3),
        target_class=target,
        requested_mode="classifier",
        parameters=RecognitionParameters(),
        image_key="key",
        header="图片: a.jpg\n",
    )


class _StubEngine:
    """记录调用并返回可控结果。"""

    def __init__(self, indices=(1, 2), error=None, delay=0.0):
        self.indices = list(indices)
        self.error = error
        self.delay = delay
        self.calls = []

    def recognize(self, image, **kwargs):
        import time
        import types

        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return types.SimpleNamespace(indices=list(self.indices))


def _wait_for(app, predicate, timeout_ms=8000):
    from PySide6.QtCore import QDeadlineTimer, QEventLoop

    deadline = QDeadlineTimer(timeout_ms)
    while not predicate() and not deadline.hasExpired():
        app.processEvents(QEventLoop.AllEvents, 20)
    return predicate()


# ---------- 请求与结果载体 ----------


def test_request_is_immutable():
    """请求提交后 UI 继续操作控件不得影响进行中的识别。"""
    request = _request()
    with pytest.raises(Exception):
        request.target_class = "Bus"  # type: ignore[misc]


def test_source_label_is_human_readable():
    assert _request(SOURCE_ONLINE).source_label == "在线采集"
    assert _request(SOURCE_FUSION).source_label == "分割与融合"


def test_outcome_exposes_indices():
    import types

    outcome = RecognitionOutcome(
        request=_request(), result=types.SimpleNamespace(indices=[0, 4])
    )
    assert outcome.indices == [0, 4]
    assert outcome.source == SOURCE_OFFLINE


# ---------- 工作线程 ----------


def test_worker_runs_off_the_ui_thread(qt_app):
    """识别必须在独立线程执行，否则窗口会冻结。"""
    import threading

    seen = {}

    class _ThreadProbe(_StubEngine):
        def recognize(self, image, **kwargs):
            seen["worker_thread"] = threading.current_thread().ident
            return super().recognize(image, **kwargs)

    engine = _ThreadProbe()
    worker = RecognitionWorker(engine)
    results = []
    worker.finished.connect(results.append)

    worker.submit(_request())
    assert _wait_for(qt_app, lambda: bool(results))

    assert seen["worker_thread"] != threading.current_thread().ident
    assert results[0].indices == [1, 2]
    worker.shutdown()


def test_worker_reports_busy_state(qt_app):
    engine = _StubEngine(delay=0.05)
    worker = RecognitionWorker(engine)
    states = []
    worker.busy_changed.connect(states.append)
    done = []
    worker.finished.connect(done.append)

    worker.submit(_request())
    assert _wait_for(qt_app, lambda: bool(done))
    # 提交时置忙、完成后置闲，供 UI 禁用按钮。
    assert states[0] is True
    assert states[-1] is False
    worker.shutdown()


def test_failure_keeps_worker_alive(qt_app):
    """单次识别失败不应让工作线程停摆。"""
    engine = _StubEngine(error=RuntimeError("模型未加载"))
    worker = RecognitionWorker(engine)
    failures = []
    worker.failed.connect(lambda source, message: failures.append((source, message)))

    worker.submit(_request(SOURCE_ONLINE))
    assert _wait_for(qt_app, lambda: bool(failures))
    assert failures[0][0] == SOURCE_ONLINE
    assert "模型未加载" in failures[0][1]

    # 线程仍然可以继续处理下一次请求。
    engine.error = None
    done = []
    worker.finished.connect(done.append)
    worker.submit(_request(SOURCE_ONLINE))
    assert _wait_for(qt_app, lambda: bool(done))
    worker.shutdown()


def test_results_route_back_to_requesting_page(qt_app):
    """三个页面共用一条链路，结果必须按 source 分发回正确页面。"""
    worker = RecognitionWorker(_StubEngine())
    received: list[str] = []
    worker.finished.connect(lambda outcome: received.append(outcome.source))

    expected = [SOURCE_OFFLINE, SOURCE_ONLINE, SOURCE_FUSION]
    for index, source in enumerate(expected, start=1):
        worker.submit(_request(source))
        assert _wait_for(qt_app, lambda count=index: len(received) >= count)

    assert received == expected
    worker.shutdown()


def test_request_carries_challenge_type_to_engine(qt_app):
    """挑战类型必须透传给引擎——3×3 与 4×4 的策略不同。"""
    engine = _StubEngine()
    worker = RecognitionWorker(engine)
    done = []
    worker.finished.connect(done.append)

    request = RecognitionRequest(
        source=SOURCE_FUSION,
        image=object(),
        challenge_type="multicaptcha",
        spec=GridSpec(4, 4),
        target_class="Bus",
        requested_mode="smart",
        parameters=RecognitionParameters(),
    )
    worker.submit(request)
    assert _wait_for(qt_app, lambda: bool(done))
    assert engine.calls[0]["challenge_type"] == "multicaptcha"
    assert engine.calls[0]["spec"] == GridSpec(4, 4)
    worker.shutdown()


def test_shutdown_stops_thread(qt_app):
    worker = RecognitionWorker(_StubEngine())
    assert worker._thread.isRunning()
    worker.shutdown()
    assert not worker._thread.isRunning()


# ---------- 主窗口 ----------


def test_main_window_builds_with_all_tabs(qt_app):
    from challenge_images.gui.qt_gui import QtChallengeGUI

    window = QtChallengeGUI()
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == ["识别与标注", "在线采集", "在线图片数据", "分割与融合", "设置配置"]
        # 识别线程随窗口启动。
        assert window.recognition_worker._thread.isRunning()
    finally:
        window.close()


def test_recognition_buttons_disable_while_busy(qt_app):
    from challenge_images.gui.qt_gui import QtChallengeGUI

    window = QtChallengeGUI()
    try:
        buttons = (
            window.recognize_button,
            window.online_recognize_button,
            window.fusion_recognize_button,
        )
        window._on_recognition_busy_changed(True)
        assert not any(button.isEnabled() for button in buttons)
        window._on_recognition_busy_changed(False)
        assert all(button.isEnabled() for button in buttons)
    finally:
        window.close()


def test_mixins_provide_extracted_tabs(qt_app):
    """四个功能页已抽成 mixin，主窗口仍应具备全部方法。"""
    from challenge_images.gui.fusion_tab import FusionTabMixin
    from challenge_images.gui.online_data_tab import OnlineDataTabMixin
    from challenge_images.gui.online_tab import OnlineTabMixin
    from challenge_images.gui.qt_gui import QtChallengeGUI
    from challenge_images.gui.settings_tab import SettingsTabMixin

    for mixin in (OnlineTabMixin, OnlineDataTabMixin, FusionTabMixin, SettingsTabMixin):
        assert issubclass(QtChallengeGUI, mixin), mixin.__name__

    expected = {
        "_refresh_online_stats", "_show_online_duplicate_detail",   # 在线数据页
        "_build_settings_tab", "_build_fusion_settings",            # 设置页
        "_build_online_tab", "_recognize_online", "_show_online_sample",  # 在线采集页
        "_build_segmentation_tab", "_recognize_fusion", "_render_fusion_preview",  # 融合页
    }
    for method in expected:
        assert hasattr(QtChallengeGUI, method), method


def test_recognition_dispatch_covers_every_page(qt_app):
    """三条识别链路的结果处理函数都必须存在，否则结果会被静默丢弃。"""
    from challenge_images.gui.qt_gui import QtChallengeGUI

    for method in (
        "_apply_offline_result",
        "_apply_online_result",
        "_apply_fusion_result",
        "_submit_recognition",
        "_compose_report",
    ):
        assert hasattr(QtChallengeGUI, method), method


def test_main_window_stays_under_line_budget():
    """主窗口只应保留骨架与离线页；功能页逻辑必须留在各自 mixin 中。

    原本 2429 行的 God Object 已拆到 1100 行以内，这条断言防止回退。
    """
    from pathlib import Path

    import challenge_images.gui.qt_gui as module

    lines = Path(module.__file__).read_text(encoding="utf-8").splitlines()
    assert len(lines) < 1200, f"qt_gui.py 已增长到 {len(lines)} 行，考虑继续拆分"
