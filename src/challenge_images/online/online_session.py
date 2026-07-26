"""在线采集会话编排：捕获 → 归档 → GUI 自动载入。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

from PIL import Image

from ..config import ONLINE_CAPTURE_DIR
from ..grid.grid_engine import GridSpec, replace_grid_tile
from .browser_session import BrowserSession, CapturedChallenge, detect_image_ext
from .solve_feedback import (
    FEEDBACK_FILENAME,
    OUTCOME_UNKNOWN,
    SolveFeedbackStore,
    SolveRecord,
)
from .capture_service import (
    ARCHIVE_FULL_CHALLENGE,
    ARCHIVE_REPLACEMENT_TILE,
    OnlineCaptureService,
    OnlineSample,
)


StatusCallback = Callable[[str], None]


@dataclass
class OnlineRoundResult:
    sample: OnlineSample
    challenge: CapturedChallenge
    clicked_indices: list[int]
    auto_clicked: bool
    # 来源：initial / poll / post_click / refresh / manual
    source: str = "initial"
    # dynamic 单格 payload 对应的网格位置；整图时为 None
    replacement_index: int | None = None
    replacement_order: int = 0
    replacement_total: int = 0


class OnlineSolveSession:
    """把浏览器会话与归档服务串成一条可复用管线。"""

    def __init__(
        self,
        capture: OnlineCaptureService | None = None,
        browser: BrowserSession | None = None,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.capture = capture or OnlineCaptureService()
        self.browser = browser or BrowserSession()
        self.on_status = on_status
        # dynamic 单格图可能在主动等待窗口后才到达；
        # 保留格子索引，让空闲监听仍能正确回填。
        self._pending_replacements: list[tuple[int, int, int]] = []
        # 解题反馈：通过的挑战可直接产出图块级真值标注。
        self.feedback = SolveFeedbackStore(ONLINE_CAPTURE_DIR / FEEDBACK_FILENAME)
        # 最近一次完整挑战图与其网格，供记录反馈时关联。
        self._last_full_sample: OnlineSample | None = None
        self._last_grid: tuple[int, int] = (3, 3)
        if on_status and self.browser.on_status is None:
            self.browser.on_status = on_status

    def _emit(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)

    def start(
        self,
        *,
        url: str | None = None,
        headless: bool = False,
        auto_click_checkbox: bool = True,
        use_playwright: bool = True,
    ) -> bool:
        if url:
            self.browser.url = url
        # GUI 已去掉无头；这里固定有头 Google Chrome
        del headless  # 保留形参兼容旧调用，实际不启用无头
        self.browser.headless = False
        self.browser.auto_click_checkbox = auto_click_checkbox
        self.browser.continuous_monitor = True
        ok = self.browser.open(use_playwright=use_playwright)
        if ok and self.browser.is_automation_ready():
            self._emit("在线会话已启动（Google Chrome 自动化，持续监控接口）")
        elif ok:
            self._emit("在线会话已启动（系统 Google Chrome，半自动）")
        else:
            self._emit("在线会话启动失败")
        return ok

    def stop(self) -> None:
        self.browser.close()
        self._emit("在线会话已关闭")

    def capture_current_challenge(
        self,
        timeout_sec: float = 30.0,
        *,
        source: str = "initial",
    ) -> OnlineRoundResult:
        """等待当前挑战的 reload/payload 并归档。"""
        if not self.browser.is_automation_ready():
            raise RuntimeError("当前不是自动化会话，请用“导入在线样本”")
        self._emit("正在等待 reload 与 payload 图片响应…")
        challenge = self.browser.wait_for_challenge(timeout_sec=timeout_sec)
        self._emit("已取得挑战类型与图片，正在归档")
        sample = self.archive_challenge(challenge)
        self._emit(f"归档完成：{sample.path}")
        return OnlineRoundResult(
            sample=sample,
            challenge=challenge,
            clicked_indices=[],
            auto_clicked=False,
            source=source,
        )

    def poll_new_challenge(self) -> OnlineRoundResult | None:
        """被动轮询：会话空闲时若有新 payload，则归档并返回。"""
        if not self.browser.is_automation_ready():
            return None
        challenge = self.browser.poll_new_challenge()
        if challenge is None:
            return None
        self._emit("持续监控发现新图，正在归档…")
        manual_index = getattr(challenge, "source_tile_index", None)
        pending = None
        if self._pending_replacements:
            pending = self._pending_replacements.pop(0)
            if manual_index is not None and int(manual_index) != pending[0]:
                self._emit(
                    f"迟到单格图位置由 replaceimage 确认为 {manual_index}，"
                    f"替代等待队列中的预期位置 {pending[0]}"
                )
        replacement_index = int(manual_index) if manual_index is not None else (pending[0] if pending else None)
        sample = self.archive_challenge(
            challenge,
            replacement_index=replacement_index,
            is_replacement=replacement_index is not None,
        )
        self._emit(f"新图已归档：{sample.path.name}")
        if manual_index is not None:
            raw_tile_id = getattr(challenge, "source_tile_id", None)
            self._emit(
                f"replaceimage 已定位：ds=[{raw_tile_id}] "
                f"→ 固定格子 {replacement_index}"
            )
        return OnlineRoundResult(
            sample=sample,
            challenge=challenge,
            clicked_indices=[replacement_index] if replacement_index is not None else [],
            auto_clicked=False,
            source="post_click" if replacement_index is not None else "poll",
            replacement_index=replacement_index,
            replacement_order=pending[1] if pending else (1 if replacement_index is not None else 0),
            replacement_total=pending[2] if pending else (1 if replacement_index is not None else 0),
        )

    def open_and_capture(
        self,
        *,
        timeout_sec: float = 35.0,
        headless: bool = False,
        auto_click_checkbox: bool = True,
    ) -> OnlineRoundResult:
        """打开 Google Chrome → 点复选框 → 等 reload/payload → 归档。"""
        self.start(
            headless=headless,
            auto_click_checkbox=auto_click_checkbox,
            use_playwright=True,
        )
        if not self.browser.is_automation_ready():
            raise RuntimeError("Google Chrome 自动采集会话未就绪")
        return self.capture_current_challenge(timeout_sec=timeout_sec, source="initial")

    def archive_challenge(
        self,
        challenge: CapturedChallenge,
        *,
        replacement_index: int | None = None,
        is_replacement: bool = False,
    ) -> OnlineSample:
        """把内存中的 reload/payload 写入 data/online_capture。"""
        ext = detect_image_ext(challenge.payload_bytes, challenge.payload_content_type)
        source_tile_id = getattr(challenge, "source_tile_id", None)
        source_tile_index = getattr(challenge, "source_tile_index", None)
        if source_tile_index is None:
            source_tile_index = replacement_index
        archive_kind = (
            ARCHIVE_REPLACEMENT_TILE
            if is_replacement or source_tile_id is not None or source_tile_index is not None
            else ARCHIVE_FULL_CHALLENGE
        )
        sample = self.capture.import_bytes(
            image_bytes=challenge.payload_bytes,
            reload_text=challenge.reload_text,
            challenge_type=challenge.challenge_type or "imageselect",
            category=challenge.category_label,
            suffix=ext,
            source_image="playwright://payload",
            archive_kind=archive_kind,
            source_tile_id=source_tile_id,
            source_tile_index=source_tile_index,
        )
        if archive_kind == ARCHIVE_FULL_CHALLENGE:
            # 记住当前整图，点击验证后据此写入图块级真值。
            self._last_full_sample = sample
            self._last_grid = (challenge.grid_rows, challenge.grid_cols)
        kind_text = "replaceimage 单格图" if archive_kind == ARCHIVE_REPLACEMENT_TILE else "完整挑战图"
        self._emit(
            f"{kind_text}已归档：{sample.challenge_type} / "
            f"{sample.category_zh} / {sample.path.name}"
        )
        return sample

    def apply_clicks(
        self,
        indices: list[int],
        *,
        click_verify: bool = False,
        delay_ms: int = 220,
        watch_after_ms: int = 8_000,
    ) -> tuple[list[int], list[OnlineRoundResult]]:
        """按格子串行点击，并为每个点击保留对应的动态新图。

        返回：(已点索引, 点击后新图列表)。

        dynamic 挑战一次点多格时，每个格子可能各发一个单格
        payload。先点完所有格子再等响应会丢失“索引 ↔ 图片”关系，
        因此这里按格子执行“点击 → 等新 payload → 记录索引”。
        """
        if not self.browser.is_automation_ready():
            raise RuntimeError("浏览器自动化未启动")
        if not indices:
            self._emit("没有需要点击的图块")
            return [], []
        requested = sorted({int(index) for index in indices if int(index) >= 0})
        challenge_type = self._current_challenge_type()
        if challenge_type != "dynamic":
            clicked = self.browser.click_tiles(requested, delay_ms=delay_ms)
            if click_verify and clicked:
                try:
                    self.browser.click_verify()
                except Exception as exc:
                    self._emit(f"验证按钮点击失败：{exc}")
            return clicked, []

        clicked: list[int] = []
        follow_ups: list[OnlineRoundResult] = []
        per_tile_timeout = max(0.2, watch_after_ms / 1000.0)

        for index in requested:
            current = self.browser.click_tiles([index], delay_ms=delay_ms)
            if index not in current:
                continue
            clicked.append(index)
            if watch_after_ms <= 0:
                continue
            self._emit(f"格子 {index} 已点击，等待对应的动态新图…")
            challenge = self.browser.wait_for_new_challenge(timeout_sec=per_tile_timeout)
            if challenge is None:
                batch_total = len(follow_ups) + 1
                for order, result in enumerate(follow_ups, start=1):
                    result.replacement_order = order
                    result.replacement_total = batch_total
                self._pending_replacements.append((index, batch_total, batch_total))
                self._emit(
                    f"格子 {index} 的新 payload 尚未到达，"
                    "已转交持续监听；为避免索引串位，暂停后续格子点击"
                )
                break
            response_index = getattr(challenge, "source_tile_index", None)
            replacement_index = int(response_index) if response_index is not None else index
            sample = self.archive_challenge(
                challenge,
                replacement_index=replacement_index,
                is_replacement=True,
            )
            if replacement_index != index:
                raw_tile_id = getattr(challenge, "source_tile_id", None)
                self._emit(
                    f"replaceimage ds=[{raw_tile_id}] 解析为格子 {replacement_index}，"
                    f"与预期点击格子 {index} 不同，按映射位置回填"
                )
            follow_ups.append(
                OnlineRoundResult(
                    sample=sample,
                    challenge=challenge,
                    clicked_indices=[replacement_index],
                    auto_clicked=False,
                    source="post_click",
                    replacement_index=replacement_index,
                )
            )
        if not self._pending_replacements:
            for order, result in enumerate(follow_ups, start=1):
                result.replacement_order = order
                result.replacement_total = len(follow_ups)
        # GUI 已移除“自动点验证”；保留参数仅兼容旧调用
        if click_verify and clicked:
            try:
                self.browser.click_verify()
            except Exception as exc:
                self._emit(f"验证按钮点击失败：{exc}")
            else:
                self._record_solve_outcome(clicked)
        if clicked and not follow_ups and watch_after_ms > 0:
            self._emit("点击后未收到单格新图（会话仍保持持续监听）")
        return clicked, follow_ups

    def _record_solve_outcome(self, clicked: list[int]) -> SolveRecord | None:
        """点击验证后记录本轮结果，供后续自动标注使用。

        只有通过的挑战才是可信真值：点击的格子含目标、未点击的不含。
        未通过时无法区分点错与漏点，记录下来但不参与标注。
        """
        sample = self._last_full_sample
        if sample is None:
            return None
        try:
            outcome = self.browser.detect_solve_outcome()
        except Exception as exc:
            self._emit(f"解题结果判定失败：{exc}")
            outcome = OUTCOME_UNKNOWN
        rows, cols = self._last_grid
        record = self.feedback.append(
            SolveRecord(
                image_name=sample.path.name,
                image_sha256=sample.sha256,
                challenge_type=sample.challenge_type,
                target_class=sample.target_class,
                grid_rows=rows,
                grid_cols=cols,
                clicked_indices=sorted(clicked),
                outcome=outcome,
            )
        )
        if record.usable:
            self._emit(
                f"挑战通过，已记录 {rows * cols} 个图块的真值标注"
                f"（{len(clicked)} 正 / {rows * cols - len(clicked)} 负）"
            )
        else:
            self._emit(f"本轮结果={record.outcome}，已记录但不用于标注")
        return record

    def _current_challenge_type(self) -> str:
        """读取当前 reload 的挑战类型，未知时按静态挑战处理。"""
        reload_text = getattr(self.browser, "_latest_reload", None)
        if not reload_text:
            return "imageselect"
        try:
            from .capture_service import parse_reload_response

            parsed = parse_reload_response(reload_text)
            return str(parsed.get("challenge_type") or "imageselect").lower()
        except Exception:
            return "imageselect"

    @staticmethod
    def merge_replacement_payload(
        base_image: Image.Image,
        challenge: CapturedChallenge,
        index: int,
    ) -> Image.Image:
        """把点击后单格 payload 合成回当前整图。"""
        replacement = Image.open(BytesIO(challenge.payload_bytes)).convert("RGB")
        spec = GridSpec(challenge.grid_rows or 3, challenge.grid_cols or 3)
        return replace_grid_tile(base_image, replacement, spec, index)

    def refresh_and_capture(self, timeout_sec: float = 30.0) -> OnlineRoundResult:
        """换图并捕获新一轮。"""
        self.browser.refresh_challenge()
        return self.capture_current_challenge(timeout_sec=timeout_sec, source="refresh")

    def status_dict(self) -> dict[str, Any]:
        return {
            "opened": self.browser.opened,
            "automation": self.browser.is_automation_ready(),
            "playwright": self.browser.playwright_available(),
            "url": self.browser.url,
            "capture_root": str(self.capture.root),
            "continuous_monitor": bool(self.browser.continuous_monitor),
        }
