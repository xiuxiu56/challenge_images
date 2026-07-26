"""在线会话常驻后台线程：同一线程内串行执行全部 Playwright 动作。

Playwright sync API 不能跨线程；因此本 worker 启动后保持运行，
用队列接收 start/capture/click/refresh/stop 指令。

空闲时周期 poll 网络，把点击后/换图后的新 payload 推给 GUI。
"""

from __future__ import annotations

from queue import Empty, Queue
import threading
import time
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from .browser_session import AutomatedQueryRestrictionError
from .capture_service import OnlineCaptureService
from .online_session import OnlineSolveSession


AUTO_REFRESH_INTERVAL_SEC = 3.0
SITE_DATA_CLEAR_INTERVAL_SEC = 180.0
CHECKBOX_MONITOR_INTERVAL_SEC = 5.0


class _OnlineEngine(QObject):
    """真正执行动作的对象，搬到 QThread 里。"""

    status = Signal(str)
    failed = Signal(str)
    started_ok = Signal(dict)
    challenge_ready = Signal(object)
    clicks_done = Signal(list)
    stopped = Signal()
    query_restricted = Signal(str)
    idle = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._queue: Queue[tuple[str, dict[str, Any]] | None] = Queue()
        self._stop_flag = threading.Event()
        self.session = OnlineSolveSession(on_status=self._on_status)
        # 空闲轮询间隔（秒）；过短会抢 CPU，过长会拖慢新图刷新
        self._poll_interval_sec = 0.35
        self._auto_refresh_enabled = False
        self._auto_refresh_interval_sec = AUTO_REFRESH_INTERVAL_SEC
        self._next_auto_refresh_at = 0.0
        self._site_data_clear_enabled = False
        self._site_data_clear_interval_sec = SITE_DATA_CLEAR_INTERVAL_SEC
        self._next_site_data_clear_at = 0.0
        self._checkbox_monitor_enabled = False
        self._checkbox_monitor_interval_sec = CHECKBOX_MONITOR_INTERVAL_SEC
        self._next_checkbox_check_at = 0.0
        self._checkbox_missing_since: float | None = None
        self._query_restricted = False

    def _on_status(self, message: str) -> None:
        self.status.emit(message)

    def submit(self, action: str, **payload: Any) -> None:
        self._queue.put((action, payload))

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._queue.put(None)

    def loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                item = self._queue.get(timeout=self._poll_interval_sec)
            except Empty:
                # 无指令时：驱动 Playwright 事件循环并检查是否有新图
                self._poll_idle()
                continue
            if item is None:
                break
            action, payload = item
            try:
                self._dispatch(action, payload)
            except AutomatedQueryRestrictionError as exc:
                self._halt_for_query_restriction(str(exc))
            except Exception as exc:
                checker = getattr(
                    self.session.browser,
                    "automated_query_restriction_message",
                    None,
                )
                restriction = checker() if callable(checker) else None
                if restriction:
                    self._halt_for_query_restriction(restriction)
                else:
                    self.failed.emit(f"{type(exc).__name__}: {exc}")
            finally:
                self.idle.emit()

        try:
            self.session.stop()
        except Exception:
            pass

    def _poll_idle(self) -> None:
        """空闲持续监控：新 payload → 归档 → challenge_ready。"""
        try:
            if not self.session.browser.is_automation_ready():
                return
            restriction = self.session.browser.automated_query_restriction_message()
            if restriction:
                self._halt_for_query_restriction(restriction)
                return
            if self._maybe_clear_site_data():
                return
            self._maybe_monitor_checkbox()
            self._maybe_auto_refresh()
            if not self.session.browser.continuous_monitor:
                # 即使不归档，也要让 Playwright 事件循环转起来
                page = self.session.browser._page
                if page is not None:
                    page.wait_for_timeout(20)
                return
            result = self.session.poll_new_challenge()
            if result is not None:
                self.challenge_ready.emit(result)
        except AutomatedQueryRestrictionError as exc:
            self._halt_for_query_restriction(str(exc))
        except Exception as exc:
            # 空闲轮询失败不打断会话，只记日志
            self.status.emit(f"持续监控异常：{exc}")

    def _halt_for_query_restriction(self, message: str) -> None:
        """清空在线动作并关闭浏览器会话。"""
        if self._query_restricted:
            return
        self._query_restricted = True
        self._configure_auto_refresh(False, emit_status=False)
        self._configure_site_data_clear(False, emit_status=False)
        self._configure_checkbox_monitor(False, emit_status=False)
        # 丢弃已排队的点击、换图和采集指令，避免关闭后被重新执行。
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        try:
            self.session.stop()
        except Exception:
            pass
        reason = str(message).strip() or "检测到自动查询限制提示"
        self.stopped.emit()
        self.query_restricted.emit(reason)

    def _configure_auto_refresh(
        self,
        enabled: bool,
        *,
        interval_sec: float = AUTO_REFRESH_INTERVAL_SEC,
        emit_status: bool = True,
    ) -> None:
        """更新自动刷新状态，Playwright 操作仍在工作线程执行。"""
        self._auto_refresh_enabled = bool(enabled)
        self._auto_refresh_interval_sec = max(1.0, float(interval_sec))
        self._next_auto_refresh_at = (
            time.monotonic() + self._auto_refresh_interval_sec
            if self._auto_refresh_enabled
            else 0.0
        )
        if emit_status:
            if self._auto_refresh_enabled:
                self.status.emit(
                    f"已开启自动刷新挑战，间隔 {self._auto_refresh_interval_sec:g} 秒"
                )
            else:
                self.status.emit("已关闭自动刷新挑战")

    def _maybe_auto_refresh(self, now: float | None = None) -> bool:
        """到达时间后点击挑战刷新按钮，新图由持续监控归档。"""
        if not self._auto_refresh_enabled:
            return False
        current = time.monotonic() if now is None else float(now)
        if current < self._next_auto_refresh_at:
            return False

        # 先安排下一次，避免刷新按钮暂时不可用时形成忙循环。
        self._next_auto_refresh_at = current + self._auto_refresh_interval_sec
        challenge_checker = getattr(
            self.session.browser,
            "is_graphic_challenge_open",
            None,
        )
        if callable(challenge_checker) and not challenge_checker():
            return False
        self.status.emit("自动刷新计时到达，正在点击 Chrome 挑战刷新按钮…")
        self.session.browser.refresh_challenge()
        self.status.emit("已点击挑战刷新按钮，等待新的 reload/payload 图片")
        return True

    def _configure_site_data_clear(
        self,
        enabled: bool,
        *,
        interval_sec: float = SITE_DATA_CLEAR_INTERVAL_SEC,
        emit_status: bool = True,
    ) -> None:
        """更新定时清理站点数据开关。"""
        self._site_data_clear_enabled = bool(enabled)
        self._site_data_clear_interval_sec = max(10.0, float(interval_sec))
        self._next_site_data_clear_at = (
            time.monotonic() + self._site_data_clear_interval_sec
            if self._site_data_clear_enabled
            else 0.0
        )
        if emit_status:
            text = (
                f"已开启定时清理站点数据，间隔 {self._site_data_clear_interval_sec:g} 秒"
                if self._site_data_clear_enabled
                else "已关闭定时清理站点数据"
            )
            self.status.emit(text)

    def _maybe_clear_site_data(self, now: float | None = None) -> bool:
        """到时清理上下文 Cookie/页面存储，再刷新当前页面。"""
        if not self._site_data_clear_enabled:
            return False
        current = time.monotonic() if now is None else float(now)
        if current < self._next_site_data_clear_at:
            return False
        self._next_site_data_clear_at = current + self._site_data_clear_interval_sec
        if self._checkbox_monitor_enabled:
            self._next_checkbox_check_at = current + self._checkbox_monitor_interval_sec
            self._checkbox_missing_since = None
        if self._auto_refresh_enabled:
            self._next_auto_refresh_at = current + self._auto_refresh_interval_sec
        self.status.emit("定时清理到达，正在清空网站数据（含第三方 Cookie）…")
        self.session.browser.clear_site_data()
        self.status.emit("网站数据已清空，等待复选框监控重新触发挑战")
        return True

    def _configure_checkbox_monitor(
        self,
        enabled: bool,
        *,
        interval_sec: float = CHECKBOX_MONITOR_INTERVAL_SEC,
        emit_status: bool = True,
    ) -> None:
        """更新复选框监控开关。"""
        self._checkbox_monitor_enabled = bool(enabled)
        self._checkbox_monitor_interval_sec = max(1.0, float(interval_sec))
        self._next_checkbox_check_at = (
            time.monotonic() + self._checkbox_monitor_interval_sec
            if self._checkbox_monitor_enabled
            else 0.0
        )
        self._checkbox_missing_since = None
        if emit_status:
            text = (
                f"已开启复选框监控，关闭后每 {self._checkbox_monitor_interval_sec:g} 秒重试"
                if self._checkbox_monitor_enabled
                else "已关闭复选框监控"
            )
            self.status.emit(text)

    def _maybe_monitor_checkbox(self, now: float | None = None) -> bool:
        """挑战页面关闭后，按间隔检查并重新点击 anchor 复选框。"""
        if not self._checkbox_monitor_enabled:
            return False
        current = time.monotonic() if now is None else float(now)
        if current < self._next_checkbox_check_at:
            return False
        self._next_checkbox_check_at = current + self._checkbox_monitor_interval_sec
        if not self.session.browser.checkbox_needs_click():
            self._checkbox_missing_since = None
            return False
        if self._checkbox_missing_since is None:
            self._checkbox_missing_since = current
            self.status.emit(
                "检测到图形挑战暂时关闭，开始 5 秒稳定性检查；"
                "期间恢复则取消复选框重试"
            )
            return False
        missing_seconds = current - self._checkbox_missing_since
        if missing_seconds < self._checkbox_monitor_interval_sec:
            return False
        self._checkbox_missing_since = None
        self.status.emit("图形挑战已连续关闭 5 秒，正在重新点击复选框…")
        self.session.browser.click_checkbox()
        self.status.emit("复选框已重新点击，等待新的 reload/payload 图片")
        return True

    def _dispatch(self, action: str, payload: dict[str, Any]) -> None:
        if action == "start_and_capture":
            self._query_restricted = False
            capture_root = str(payload.get("capture_root") or "").strip()
            if capture_root:
                self.session.capture = OnlineCaptureService(capture_root)
            auto_refresh = bool(payload.get("auto_refresh", False))
            clear_site_data = bool(payload.get("clear_site_data", False))
            monitor_checkbox = bool(payload.get("monitor_checkbox", False))
            refresh_interval = float(
                payload.get("refresh_interval", AUTO_REFRESH_INTERVAL_SEC)
            )
            self._configure_auto_refresh(False, emit_status=False)
            self._configure_site_data_clear(False, emit_status=False)
            self._configure_checkbox_monitor(False, emit_status=False)
            started = self.session.start(
                # 强制有头；GUI 已去掉无头开关
                headless=False,
                auto_click_checkbox=bool(payload.get("auto_click_checkbox", True)),
            )
            self.started_ok.emit(self.session.status_dict())
            if not started or not self.session.browser.is_automation_ready():
                raise RuntimeError("Google Chrome 自动采集会话未就绪")
            result = self.session.capture_current_challenge(
                timeout_sec=float(payload.get("timeout", 45.0)),
                source="initial",
            )
            self.challenge_ready.emit(result)
            self._configure_auto_refresh(
                auto_refresh,
                interval_sec=refresh_interval,
                emit_status=auto_refresh,
            )
            self._configure_site_data_clear(clear_site_data, emit_status=clear_site_data)
            self._configure_checkbox_monitor(monitor_checkbox, emit_status=monitor_checkbox)
        elif action == "capture":
            result = self.session.capture_current_challenge(
                timeout_sec=float(payload.get("timeout", 30.0)),
                source="manual",
            )
            self.challenge_ready.emit(result)
        elif action == "click":
            clicked, follow_ups = self.session.apply_clicks(
                list(payload.get("indices") or []),
                click_verify=bool(payload.get("click_verify", False)),
                delay_ms=int(payload.get("delay_ms", 220)),
                watch_after_ms=int(payload.get("watch_after_ms", 8_000)),
            )
            self.clicks_done.emit(clicked)
            for follow_up in follow_ups:
                # 每个点击格子各带自己的 replacement_index，
                # GUI 按信号顺序合成，避免多格替换图互相覆盖。
                self.challenge_ready.emit(follow_up)
        elif action == "refresh":
            result = self.session.refresh_and_capture(
                timeout_sec=float(payload.get("timeout", 30.0))
            )
            self.challenge_ready.emit(result)
            if self._auto_refresh_enabled:
                self._next_auto_refresh_at = (
                    time.monotonic() + self._auto_refresh_interval_sec
                )
        elif action == "set_auto_refresh":
            self._configure_auto_refresh(
                bool(payload.get("enabled", False)),
                interval_sec=float(
                    payload.get("interval", AUTO_REFRESH_INTERVAL_SEC)
                ),
            )
        elif action == "set_site_data_clear":
            self._configure_site_data_clear(
                bool(payload.get("enabled", False)),
                interval_sec=float(
                    payload.get("interval", SITE_DATA_CLEAR_INTERVAL_SEC)
                ),
            )
        elif action == "set_checkbox_monitor":
            self._configure_checkbox_monitor(
                bool(payload.get("enabled", False)),
                interval_sec=float(
                    payload.get("interval", CHECKBOX_MONITOR_INTERVAL_SEC)
                ),
            )
        elif action == "stop":
            self._configure_auto_refresh(False, emit_status=False)
            self._configure_site_data_clear(False, emit_status=False)
            self._configure_checkbox_monitor(False, emit_status=False)
            self.session.stop()
            self.stopped.emit()
        else:
            self.failed.emit(f"未知在线动作：{action}")


class OnlineWorker(QObject):
    """GUI 侧门面：管理线程与引擎，暴露与旧接口兼容的 request_*。"""

    status = Signal(str)
    failed = Signal(str)
    started_ok = Signal(dict)
    challenge_ready = Signal(object)
    clicks_done = Signal(list)
    stopped = Signal()
    query_restricted = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._thread = QThread(parent)
        self._engine = _OnlineEngine()
        self._engine.moveToThread(self._thread)
        self._busy = False

        self._engine.status.connect(self.status)
        self._engine.failed.connect(self._on_failed)
        self._engine.started_ok.connect(self.started_ok)
        self._engine.challenge_ready.connect(self.challenge_ready)
        self._engine.clicks_done.connect(self.clicks_done)
        self._engine.stopped.connect(self.stopped)
        self._engine.query_restricted.connect(self.query_restricted)
        self._engine.idle.connect(self._on_idle)
        self._thread.started.connect(self._engine.loop)
        self._thread.start()

    @property
    def session(self) -> OnlineSolveSession:
        return self._engine.session

    def isRunning(self) -> bool:  # noqa: N802 — 兼容旧调用
        return self._busy

    def wait(self, _msec: int = 0) -> None:
        return None

    def _on_failed(self, message: str) -> None:
        self._busy = False
        self.failed.emit(message)

    def _on_idle(self) -> None:
        self._busy = False

    def _submit(self, action: str, **payload: Any) -> None:
        # 始终入队串行执行；不要因 busy 丢弃“识别后点击”等后续动作
        self._busy = True
        self._engine.submit(action, **payload)

    def request_start(
        self,
        *,
        headless: bool = False,
        auto_click_checkbox: bool = True,
        capture_timeout: float = 45.0,
        auto_refresh: bool = False,
        refresh_interval: float = AUTO_REFRESH_INTERVAL_SEC,
        clear_site_data: bool = False,
        monitor_checkbox: bool = False,
        capture_root: str | None = None,
    ) -> None:
        # headless 参数保留兼容；真正执行时固定有头 Google Chrome
        self._submit(
            "start_and_capture",
            headless=False,
            auto_click_checkbox=auto_click_checkbox,
            timeout=capture_timeout,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
            clear_site_data=clear_site_data,
            monitor_checkbox=monitor_checkbox,
            capture_root=capture_root,
        )

    def request_capture(self, timeout: float = 30.0) -> None:
        self._submit("capture", timeout=timeout)

    def request_apply_clicks(
        self,
        indices: list[int],
        *,
        click_verify: bool = False,
        watch_after_ms: int = 8_000,
        delay_ms: int = 220,
    ) -> None:
        self._submit(
            "click",
            indices=list(indices),
            click_verify=click_verify,
            watch_after_ms=watch_after_ms,
            delay_ms=delay_ms,
        )

    def request_refresh(self, timeout: float = 30.0) -> None:
        self._submit("refresh", timeout=timeout)

    def request_auto_refresh(
        self,
        enabled: bool,
        interval: float = AUTO_REFRESH_INTERVAL_SEC,
    ) -> None:
        self._submit("set_auto_refresh", enabled=enabled, interval=interval)

    def request_site_data_clear(
        self,
        enabled: bool,
        interval: float = SITE_DATA_CLEAR_INTERVAL_SEC,
    ) -> None:
        self._submit("set_site_data_clear", enabled=enabled, interval=interval)

    def request_checkbox_monitor(
        self,
        enabled: bool,
        interval: float = CHECKBOX_MONITOR_INTERVAL_SEC,
    ) -> None:
        self._submit("set_checkbox_monitor", enabled=enabled, interval=interval)

    def request_stop(self) -> None:
        self._submit("stop")

    def shutdown(self) -> None:
        self._engine.shutdown()
        self._thread.quit()
        self._thread.wait(3000)
