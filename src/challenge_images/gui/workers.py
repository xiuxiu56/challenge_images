"""把识别推理搬出 UI 线程。

原实现直接在 UI 线程调用 ``RecognitionEngine.recognize``，只能靠
``QApplication.setOverrideCursor(Qt.WaitCursor)`` 加 ``processEvents()``
硬扛。一张 4×4 图要跑 16 格分类加整图分割，在 MPS 上耗时数秒，
这几秒内窗口完全冻结、无法拖动也无法取消。

在线采集侧（``online/online_worker.py``）已经是正确写法：
QObject + moveToThread + 信号回调。本模块把同一范式用到识别上。

设计要点：

- 请求串行执行。识别会独占 GPU，并发提交只会互相拖慢。
- 只保留最后一次请求。用户快速连点「开始识别」时，中间那些结果
  没人要，丢弃比排队更符合预期。
- 单次失败只发 ``failed`` 信号，线程继续存活等待下一次请求。
"""

from __future__ import annotations

import threading
from queue import Queue
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from .state import RecognitionOutcome, RecognitionRequest


class _RecognitionEngineWorker(QObject):
    """真正执行识别的对象，搬到 QThread 里。"""

    finished = Signal(object)  # RecognitionOutcome
    failed = Signal(str, str)  # (source, message)

    def __init__(self, engine: Any) -> None:
        super().__init__()
        self.engine = engine
        self._queue: Queue[RecognitionRequest | None] = Queue()
        self._stop_flag = threading.Event()

    def submit(self, request: RecognitionRequest) -> None:
        self._queue.put(request)

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._queue.put(None)

    def _drain_to_latest(self, request: RecognitionRequest) -> RecognitionRequest:
        """丢弃积压的中间请求，只保留最新一次。"""
        latest = request
        while not self._queue.empty():
            pending = self._queue.get_nowait()
            if pending is None:
                self._stop_flag.set()
                break
            latest = pending
        return latest

    def loop(self) -> None:
        while not self._stop_flag.is_set():
            request = self._queue.get()
            if request is None:
                break
            request = self._drain_to_latest(request)
            if self._stop_flag.is_set():
                break
            try:
                result = self.engine.recognize(
                    request.image,
                    challenge_type=request.challenge_type,
                    spec=request.spec,
                    target_class=request.target_class,
                    requested_mode=request.requested_mode,
                    parameters=request.parameters,
                    image_key=request.image_key,
                )
            except Exception as error:  # 单次失败不能拖垮工作线程
                self.failed.emit(request.source, f"{type(error).__name__}: {error}")
                continue
            self.finished.emit(RecognitionOutcome(request=request, result=result))


class RecognitionWorker(QObject):
    """GUI 侧门面：管理识别线程，暴露提交接口与结果信号。"""

    finished = Signal(object)  # RecognitionOutcome
    failed = Signal(str, str)  # (source, message)
    busy_changed = Signal(bool)

    def __init__(self, engine: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread(parent)
        self._worker = _RecognitionEngineWorker(engine)
        self._worker.moveToThread(self._thread)
        self._busy = False

        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.started.connect(self._worker.loop)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy

    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busy_changed.emit(value)

    def submit(self, request: RecognitionRequest) -> None:
        """提交一次识别；UI 线程立即返回。"""
        self._set_busy(True)
        self._worker.submit(request)

    def _on_finished(self, outcome: RecognitionOutcome) -> None:
        self._set_busy(False)
        self.finished.emit(outcome)

    def _on_failed(self, source: str, message: str) -> None:
        self._set_busy(False)
        self.failed.emit(source, message)

    def shutdown(self) -> None:
        """关闭窗口时调用，确保线程干净退出。"""
        self._worker.shutdown()
        self._thread.quit()
        self._thread.wait(3000)
