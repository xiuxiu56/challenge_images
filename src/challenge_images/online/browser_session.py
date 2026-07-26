"""Playwright 在线会话：监听 reload/payload/replaceimage 并归档。

坐标与 DOM 策略来自 chrome-devtools MCP 实测：
- 目标页：https://www.google.com/recaptcha/api2/demo
- 复选框在 anchor iframe 内
- 挑战图在 bframe iframe 内
- 类别来自 POST .../api2/reload 响应
- 初始整图来自 GET .../api2/payload
- dynamic 位置来自 POST .../api2/replaceimage 的 ds，单格图片来自紧随其后的 /payload
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Callable
import webbrowser
from urllib.parse import parse_qs, urlparse

from ..grid.grid_engine import resolve_challenge_grid
from .capture_service import DEMO_URL
from .click_geometry import Rect, TileLayout, build_tile_layout, cell_center, grid_from_pmeta


ReloadCallback = Callable[[str], None]
PayloadCallback = Callable[[bytes, str | None], None]
StatusCallback = Callable[[str], None]


AUTOMATED_QUERY_RESTRICTION_MARKERS = (
    "您的计算机或网络可能在发送自动查询内容",
    "为了保护我们的用户，我们目前无法处理您的请求",
    "our systems have detected unusual traffic from your computer network",
    "your computer or network may be sending automated queries",
)


class AutomatedQueryRestrictionError(RuntimeError):
    """页面已显示自动查询限制提示。"""


def detect_automated_query_restriction(text: str | None) -> bool:
    """检查页面文本是否包含 Google 自动查询限制提示。"""
    normalized = "".join(str(text or "").lower().split())
    return any(
        "".join(marker.lower().split()) in normalized
        for marker in AUTOMATED_QUERY_RESTRICTION_MARKERS
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summarize_playwright_error(error: Exception, *, action: str = "操作") -> str:
    """把 Playwright 多行 Call log 压缩成一条中文提示。"""
    message = str(error or "").strip()
    lowered = message.lower()
    if "intercepts pointer events" in lowered or "outside of the viewport" in lowered:
        return "图块被其他页面元素遮挡或已离开可视区域"
    if "timeout" in lowered:
        return f"{action}超时"
    if "not visible" in lowered:
        return "目标元素当前不可见"
    first_line = next((line.strip() for line in message.splitlines() if line.strip()), "未知错误")
    if len(first_line) > 160:
        first_line = first_line[:157] + "…"
    return first_line


def parse_replaceimage_indices(post_data: str | bytes | None) -> list[int]:
    """从 ``replaceimage`` 表单请求中解析 ``ds`` 格子索引。

    真实请求示例：``ds=%5B2%5D``，URL 解码后为 ``ds=[2]``。
    """
    if post_data is None:
        return []
    if isinstance(post_data, bytes):
        text = post_data.decode("utf-8", errors="replace")
    else:
        text = str(post_data)
    values = parse_qs(text, keep_blank_values=True).get("ds") or []
    indices: list[int] = []
    for value in values:
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = value
        items = decoded if isinstance(decoded, list) else [decoded]
        for item in items:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if index >= 0 and index not in indices:
                indices.append(index)
    return indices

MACOS_CHROME_EXECUTABLE = Path(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)


@dataclass
class CapturedChallenge:
    """一轮挑战捕获结果。"""

    reload_text: str
    payload_bytes: bytes
    payload_content_type: str | None = None
    challenge_type: str | None = None
    category_label: str | None = None
    pmeta: Any = None
    grid_rows: int = 3
    grid_cols: int = 3
    # replaceimage 请求参数 ds 中的原始动态图块 ID。
    source_tile_id: int | None = None
    # 原始动态图块 ID 解析后对应的固定 GUI 网格位置。
    source_tile_index: int | None = None


@dataclass
class BrowserSession:
    """浏览器会话。

    - Playwright 可用时：强制本机 Google Chrome（不用内置 Chromium）打开并监听网络
    - Playwright 不可用时：用系统 Google Chrome 打开页面
    """

    url: str = DEMO_URL
    headless: bool = False
    opened: bool = False
    # 开始会话后默认点「我不是机器人」，触发图片挑战
    auto_click_checkbox: bool = True
    timeout_ms: int = 45_000

    _playwright: Any = field(default=None, repr=False)
    _browser: Any = field(default=None, repr=False)
    _context: Any = field(default=None, repr=False)
    _page: Any = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    _latest_reload: str | None = field(default=None, repr=False)
    _latest_payload: bytes | None = field(default=None, repr=False)
    _latest_payload_type: str | None = field(default=None, repr=False)
    _capture_started_at: float = field(default=0.0, repr=False)
    _reload_received_at: float = field(default=0.0, repr=False)
    _payload_received_at: float = field(default=0.0, repr=False)
    _reload_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _payload_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # 已交给上层归档/展示的 payload 哈希；用于持续监控时去重
    _consumed_payload_sha: str | None = field(default=None, repr=False)
    _consumed_reload_sha: str | None = field(default=None, repr=False)
    # 最近一次接口变化摘要，便于日志与调试
    _last_interface_note: str | None = field(default=None, repr=False)
    # 是否启用被动监控（点击后、会话空闲时继续收新图）
    continuous_monitor: bool = True
    # 每个 replaceimage 响应与其 ds 索引在响应回调内直接绑定。
    # 使用队列保留快速连续点击产生的所有单格新图。
    _pending_replace_indices: deque[int] = field(default_factory=deque, repr=False)
    _replacement_queue: deque[CapturedChallenge] = field(default_factory=deque, repr=False)
    # dynamic 挑战的 ds 是“动态图块 ID”，不是始终为 0–8 的固定位置。
    # 例如固定位置 4 第一次替换后可能变为 ID=10，后续 ds=[10]
    # 仍应回填 GUI 格子 4。
    _dynamic_tile_positions: dict[int, int] = field(default_factory=dict, repr=False)
    _next_dynamic_tile_id: int = field(default=0, repr=False)
    _dynamic_tile_count: int = field(default=0, repr=False)

    on_status: StatusCallback | None = None
    on_reload: ReloadCallback | None = None
    on_payload: PayloadCallback | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def _emit(self, message: str) -> None:
        if self.on_status:
            self.on_status(message)

    @staticmethod
    def playwright_available() -> bool:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def chrome_available() -> bool:
        """检查本机是否存在可由 Playwright channel=chrome 启动的 Chrome。"""
        if MACOS_CHROME_EXECUTABLE.is_file():
            return True
        return any(
            shutil.which(name)
            for name in ("google-chrome", "google-chrome-stable", "chrome")
        )

    def open(self, *, use_playwright: bool = True) -> bool:
        """打开目标页。优先受控 Google Chrome，失败后打开系统 Chrome。"""
        with self._lock:
            if use_playwright and self.playwright_available() and self.chrome_available():
                try:
                    self._open_playwright()
                    self.opened = True
                    return True
                except Exception as exc:
                    self._emit(f"受控 Google Chrome 启动失败，改用系统 Chrome：{exc}")
                    self.close()
            self.opened = self._open_system_chrome()
            self._emit("系统 Google Chrome 已打开，请在页面中人工触发图片挑战")
            return self.opened

    def _open_system_chrome(self) -> bool:
        """在 macOS 上明确使用 Google Chrome，其他系统使用默认浏览器。"""
        if MACOS_CHROME_EXECUTABLE.is_file():
            result = subprocess.run(
                ["open", "-a", "Google Chrome", self.url],
                check=False,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        return bool(webbrowser.open(self.url, new=2))

    def _chrome_launch_options(self) -> dict[str, Any]:
        """构造只指向本机 Google Chrome 的启动参数，避免落到内置 Chromium。"""
        options: dict[str, Any] = {
            "headless": bool(self.headless),
            # 使用可见窗口和稳定的 Chrome 启动参数，不修改页面指纹。
            "args": [
                "--no-first-run",
                "--no-default-browser-check",
            ],
        }
        if MACOS_CHROME_EXECUTABLE.is_file():
            options["executable_path"] = str(MACOS_CHROME_EXECUTABLE)
            self._emit(f"使用本机 Chrome：{MACOS_CHROME_EXECUTABLE}")
            return options
        for name in ("google-chrome-stable", "google-chrome", "chrome"):
            path = shutil.which(name)
            if path:
                options["executable_path"] = path
                self._emit(f"使用本机 Chrome：{path}")
                return options
        # 最后才用 channel=chrome（仍指向已安装 Chrome，不是 bundled Chromium）
        options["channel"] = "chrome"
        self._emit("使用 Playwright channel=chrome 启动本机 Google Chrome")
        return options

    def _open_playwright(self) -> None:
        from playwright.sync_api import sync_playwright

        self.close()
        self._playwright = sync_playwright().start()
        self._emit("正在启动本机 Google Chrome（非 Chromium）")
        launch_options = self._chrome_launch_options()
        self._browser = self._playwright.chromium.launch(**launch_options)
        version = ""
        try:
            version = str(self._browser.version or "")
        except Exception:
            version = ""
        self._emit(f"Google Chrome 已启动{f'，版本 {version}' if version else ''}")
        self._context = self._browser.new_context(
            locale="zh-CN",
            viewport={"width": 1100, "height": 900},
        )
        self._page = self._context.new_page()
        self._page.on("request", self._on_request)
        self._page.on("response", self._on_response)
        self._emit(f"正在打开：{self.url}")
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=self.timeout_ms)
        self._emit("Google Chrome 页面已加载")
        if self.auto_click_checkbox:
            # 等 anchor iframe 就绪后再点，避免空白页上抢点失败
            try:
                self._page.wait_for_selector(
                    'iframe[src*="anchor"]',
                    state="attached",
                    timeout=15_000,
                )
                self._page.wait_for_timeout(600)
            except Exception:
                pass
            self.click_checkbox()
            self._emit("已点击复选框，等待 /reload 与 /payload…")
        else:
            self._emit("等待人工点击复选框以触发图片挑战…")

    def close(self) -> None:
        with self._lock:
            for name in ("_page", "_context", "_browser"):
                obj = getattr(self, name, None)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
                    setattr(self, name, None)
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            self.opened = False
            self._latest_reload = None
            self._latest_payload = None
            self._latest_payload_type = None
            self._capture_started_at = 0.0
            self._reload_received_at = 0.0
            self._payload_received_at = 0.0
            self._reload_event.clear()
            self._payload_event.clear()
            self._consumed_payload_sha = None
            self._consumed_reload_sha = None
            self._last_interface_note = None
            self._pending_replace_indices.clear()
            self._replacement_queue.clear()
            self._clear_dynamic_tile_map()

    def clear_site_data(self) -> None:
        """清空当前 Playwright 上下文的网站数据并刷新页面。

        ``clear_cookies`` 会清空当前上下文中的第一方和第三方 Cookie；
        页面脚本再清理 localStorage/sessionStorage。上下文不是持久化用户目录，
        因此不触碰用户 Chrome 的其他站点数据。
        """
        page = self._require_page()
        context = self._context
        if context is not None:
            clear_cookies = getattr(context, "clear_cookies", None)
            if callable(clear_cookies):
                clear_cookies()
            new_cdp_session = getattr(context, "new_cdp_session", None)
            if callable(new_cdp_session):
                try:
                    cdp = new_cdp_session(page)
                    cdp.send("Network.clearBrowserCookies")
                    cdp.send("Network.clearBrowserCache")
                    parsed_url = urlparse(str(getattr(page, "url", "") or self.url))
                    if parsed_url.scheme and parsed_url.netloc:
                        cdp.send(
                            "Storage.clearDataForOrigin",
                            {
                                "origin": f"{parsed_url.scheme}://{parsed_url.netloc}",
                                "storageTypes": "all",
                            },
                        )
                    cdp.detach()
                except Exception as exc:
                    self._emit(f"Chrome 缓存清理提示：{exc}")
        try:
            page.evaluate(
                """() => {
                    try { localStorage.clear(); } catch (_) {}
                    try { sessionStorage.clear(); } catch (_) {}
                }"""
            )
        except Exception as exc:
            self._emit(f"页面存储清理提示：{exc}")
        self.reset_capture()
        try:
            page.reload(wait_until="domcontentloaded", timeout=self.timeout_ms)
        except Exception as exc:
            self._emit(f"清理后刷新页面失败：{exc}")
            raise
        self._emit("已清空当前会话网站数据（包含第三方 Cookie），页面已刷新")

    def is_graphic_challenge_open(self) -> bool:
        """判断图形挑战 iframe 是否仍在显示。"""
        page = self._page
        if page is None:
            return False
        try:
            frames = page.locator('iframe[src*="bframe"]')
            if frames.count() <= 0:
                return False
            frame = frames.first
            if not frame.is_visible():
                return False
            box = frame.bounding_box()
            return bool(box and box.get("width", 0) >= 100 and box.get("height", 0) >= 100)
        except Exception:
            return False

    def checkbox_needs_click(self) -> bool:
        """判断 anchor 复选框是否可见且尚未勾选。"""
        page = self._page
        if page is None or self.is_graphic_challenge_open():
            return False
        frame_selectors = (
            'iframe[src*="anchor"]',
            'iframe[title*="reCAPTCHA"]',
            'iframe[src*="recaptcha"]',
        )
        for frame_sel in frame_selectors:
            try:
                anchor = page.frame_locator(frame_sel)
                checkbox = anchor.locator("#recaptcha-anchor").first
                if not checkbox.is_visible():
                    continue
                checked = checkbox.get_attribute("aria-checked")
                return checked != "true"
            except Exception:
                continue
        return False

    def checkbox_is_checked(self) -> bool:
        """anchor 复选框是否已勾选，即本轮验证是否已通过。"""
        page = self._page
        if page is None:
            return False
        for frame_sel in (
            'iframe[src*="anchor"]',
            'iframe[title*="reCAPTCHA"]',
            'iframe[src*="recaptcha"]',
        ):
            try:
                checkbox = page.frame_locator(frame_sel).locator("#recaptcha-anchor").first
                if not checkbox.is_visible():
                    continue
                return checkbox.get_attribute("aria-checked") == "true"
            except Exception:
                continue
        return False

    def detect_solve_outcome(self, *, settle_sec: float = 1.5) -> str:
        """点击验证后判定本轮结果。

        判定依据只读 DOM，不发任何额外请求：

        - 图形挑战已关闭且复选框已勾选 → 通过
        - 图形挑战仍然打开 → 未通过（或多步骤挑战进入下一步）
        - 其余情况无法确定，返回未知，调用方不应据此标注

        返回值取自 ``solve_feedback`` 中的 OUTCOME_* 常量。
        """
        from .solve_feedback import OUTCOME_FAILED, OUTCOME_PASSED, OUTCOME_UNKNOWN

        if self._page is None:
            return OUTCOME_UNKNOWN
        # 验证结果需要一点时间落到 DOM 上。
        deadline = time.time() + max(0.0, settle_sec)
        while time.time() < deadline:
            if self.checkbox_is_checked():
                return OUTCOME_PASSED
            time.sleep(0.15)
        try:
            if self.is_graphic_challenge_open():
                return OUTCOME_FAILED
        except Exception:
            return OUTCOME_UNKNOWN
        return OUTCOME_PASSED if self.checkbox_is_checked() else OUTCOME_UNKNOWN

    def automated_query_restriction_message(self) -> str | None:
        """检查主页和所有 iframe 是否出现自动查询限制提示。"""
        page = self._page
        if page is None:
            return None
        try:
            frames = list(page.frames)
        except Exception:
            frames = []
        for frame in frames:
            try:
                text = frame.locator("body").inner_text(timeout=300)
            except Exception:
                continue
            if detect_automated_query_restriction(text):
                return (
                    "检测到页面提示‘计算机或网络可能在发送自动查询’，"
                    "已停止全部在线操作并关闭浏览器会话"
                )
        return None

    def raise_if_automated_query_restricted(self) -> None:
        """发现自动查询限制后立即中断当前等待或点击流程。"""
        message = self.automated_query_restriction_message()
        if message:
            raise AutomatedQueryRestrictionError(message)

    def is_automation_ready(self) -> bool:
        return self._page is not None

    # ------------------------------------------------------------------
    # 网络拦截
    # ------------------------------------------------------------------

    @staticmethod
    def _response_kind(url: str) -> str | None:
        """根据路径识别挑战响应，兼容 api2 与 enterprise。"""
        try:
            path = urlparse(url).path.rstrip("/").lower()
        except Exception:
            path = url.split("?", 1)[0].rstrip("/").lower()
        if path.endswith("/reload"):
            return "reload"
        if path.endswith("/payload"):
            return "payload"
        if path.endswith("/replaceimage"):
            return "replaceimage"
        return None

    def _build_replacement_challenge(
        self,
        body: bytes,
        content_type: str | None,
        tile_id: int,
    ) -> CapturedChallenge | None:
        """把 /payload 图片与 replaceimage 的动态图块 ID 绑定。"""
        if not self._latest_reload:
            self._emit("replaceimage 后续 payload 已返回，但当前没有 reload 元数据")
            return None
        from .capture_service import parse_reload_response

        parsed = parse_reload_response(self._latest_reload)
        categories = parsed.get("categories") or []
        pmeta = parsed.get("pmeta")
        spec = resolve_challenge_grid(
            str(parsed.get("challenge_type") or "dynamic"),
            grid_from_pmeta(pmeta),
        )
        labels = [str(item.get("label") or item.get("id")) for item in categories]
        category_label = ", ".join(labels) if labels else None
        self._ensure_dynamic_tile_map(spec.count)
        position = self._dynamic_tile_positions.get(tile_id)
        if position is None:
            self._emit(
                f"replaceimage 遇到未知动态图块 ID：ds=[{tile_id}]，"
                f"当前已记录 ID 范围=0–{max(self._next_dynamic_tile_id - 1, 0)}"
            )
            return None

        # 每个 replaceimage + payload 成功后，网页会为新图片分配
        # 下一个递增图块 ID。新 ID 仍指向同一个固定网格位置。
        new_tile_id = self._next_dynamic_tile_id
        self._dynamic_tile_positions[new_tile_id] = position
        self._next_dynamic_tile_id += 1
        self._emit(
            f"replaceimage 动态 ID 映射：{tile_id} → 格子 {position}；"
            f"新图块 ID {new_tile_id} → 格子 {position}"
        )
        return CapturedChallenge(
            reload_text=self._latest_reload,
            payload_bytes=body,
            payload_content_type=content_type,
            challenge_type=str(parsed.get("challenge_type") or "dynamic"),
            category_label=category_label,
            pmeta=pmeta,
            grid_rows=spec.rows,
            grid_cols=spec.columns,
            source_tile_id=tile_id,
            source_tile_index=position,
        )

    def _clear_dynamic_tile_map(self) -> None:
        """清空当前 dynamic 挑战的图块 ID 映射。"""
        self._dynamic_tile_positions.clear()
        self._next_dynamic_tile_id = 0
        self._dynamic_tile_count = 0

    def _reset_dynamic_tile_map(self, count: int) -> None:
        """为一轮新挑战建立初始 ID 到固定位置的映射。"""
        tile_count = max(0, int(count))
        self._dynamic_tile_positions = {index: index for index in range(tile_count)}
        self._next_dynamic_tile_id = tile_count
        self._dynamic_tile_count = tile_count
        if tile_count:
            self._emit(
                f"已初始化动态图块映射：ID 0–{tile_count - 1} "
                f"对应固定格子 0–{tile_count - 1}"
            )

    def _ensure_dynamic_tile_map(self, count: int) -> None:
        """确保当前挑战已建立动态图块映射。"""
        tile_count = max(0, int(count))
        if not self._dynamic_tile_positions or self._dynamic_tile_count != tile_count:
            self._reset_dynamic_tile_map(tile_count)

    def _on_request(self, request: Any) -> None:
        """在 replaceimage 请求发出时立即登记 ds。"""
        try:
            url = str(getattr(request, "url", "") or "")
            if self._response_kind(url) != "replaceimage":
                return
            method = str(getattr(request, "method", "") or "").upper()
            if method not in {"POST", "GET"}:
                return
            post_data = getattr(request, "post_data", None)
            indices = parse_replaceimage_indices(post_data)
            if not indices:
                self._emit("replaceimage 请求缺少有效 ds 索引，忽略")
                return
            self._pending_replace_indices.extend(indices)
            self._emit(
                f"发现 replaceimage 请求：ds={indices}，"
                f"等待后续 /payload 图片，"
                f"待配对={len(self._pending_replace_indices)}"
            )
        except Exception as exc:
            self._emit(f"replaceimage 请求参数解析异常：{exc}")

    def _on_response(self, response: Any) -> None:
        try:
            url = response.url or ""
            kind = self._response_kind(url)
            method = str(getattr(response.request, "method", "")).upper()
            if kind is None:
                return
            status = getattr(response, "status", None)
            self._emit(f"发现 {kind} 响应：{method} {urlparse(url).path}（状态 {status or '未知'}）")
            # 只接收本轮点击后的响应，避免把页面初始化旧响应配到新挑战。
            if self._capture_started_at and time.monotonic() < self._capture_started_at:
                return
            if kind == "replaceimage" and method in {"POST", "GET"}:
                # 该响应体不是图片；ds 已在 request 事件中登记。
                self._emit("replaceimage 响应已完成，等待配对的 /payload 图片")
                return
            if kind == "reload" and method in {"POST", "GET"}:
                text = response.text()
                if not text or len(text.strip()) < 8:
                    self._emit("reload 响应为空，忽略本次响应")
                    return
                # 先交给解析器判断，不再用固定的 rresp 字符串过滤。
                try:
                    from .capture_service import parse_reload_response

                    parsed = parse_reload_response(text)
                except Exception as exc:
                    self._emit(f"reload 响应解析失败：{exc}")
                    return
                self._latest_reload = text
                self._reload_received_at = time.monotonic()
                self._reload_event.set()
                categories = parsed.get("categories") or []
                raw_type = parsed.get("challenge_type") or "未知"
                labels = [str(item.get("label") or item.get("id")) for item in categories]
                self._emit(f"reload 已解析：类型={raw_type}，类别={', '.join(labels) or '未知'}")
                if self.on_reload:
                    self.on_reload(text)
            elif kind == "payload" and method in {"GET", "POST"}:
                body = response.body()
                if body and len(body) > 100 and status not in {204, 404}:
                    ctype = None
                    try:
                        ctype = response.headers.get("content-type")
                    except Exception:
                        ctype = None
                    if self._pending_replace_indices:
                        # replaceimage 只给出 ds，真正的单格图在随后
                        # /payload 响应中。该分支不改动初始完整图。
                        tile_id = self._pending_replace_indices.popleft()
                        replacement = self._build_replacement_challenge(body, ctype, tile_id)
                        if replacement is not None:
                            self._replacement_queue.append(replacement)
                            self._emit(
                                f"payload 单格图已与 replaceimage ds=[{tile_id}] 配对："
                                f"{len(body)} 字节，待处理={len(self._replacement_queue)}"
                            )
                    else:
                        # 没有待配对 ds 时，/payload 才是新的完整挑战图。
                        self._latest_payload = body
                        self._latest_payload_type = ctype
                        self._payload_received_at = time.monotonic()
                        self._payload_event.set()
                        self._emit(
                            f"payload 完整图已读取：{len(body)} 字节，"
                            f"类型={ctype or '未知'}"
                        )
                        if self.on_payload:
                            self.on_payload(body, ctype)
                else:
                    self._emit(f"{kind} 响应没有有效图片字节，忽略")
        except Exception as exc:
            self._emit(f"网络拦截异常：{exc}")

    def reset_capture(self) -> None:
        self._capture_started_at = time.monotonic()
        self._latest_reload = None
        self._latest_payload = None
        self._latest_payload_type = None
        self._reload_received_at = 0.0
        self._payload_received_at = 0.0
        self._reload_event.clear()
        self._payload_event.clear()
        # 主动重捕时清掉已消费标记，下一轮新图可再次交付
        self._consumed_payload_sha = None
        self._consumed_reload_sha = None
        self._last_interface_note = None
        self._pending_replace_indices.clear()
        self._replacement_queue.clear()
        self._clear_dynamic_tile_map()

    def _current_payload_sha(self) -> str | None:
        if not self._latest_payload:
            return None
        return _sha256_bytes(self._latest_payload)

    def _current_reload_sha(self) -> str | None:
        if not self._latest_reload:
            return None
        return _sha256_text(self._latest_reload)

    def _mark_challenge_consumed(self) -> None:
        self._consumed_payload_sha = self._current_payload_sha()
        self._consumed_reload_sha = self._current_reload_sha()

    def _has_unconsumed_payload(self) -> bool:
        sha = self._current_payload_sha()
        return bool(sha) and sha != self._consumed_payload_sha

    def _has_paired_challenge(self, *, require_since_capture: bool = True) -> bool:
        if not self._latest_reload or not self._latest_payload:
            return False
        if require_since_capture and self._capture_started_at:
            if self._reload_received_at < self._capture_started_at:
                return False
            if self._payload_received_at < self._capture_started_at:
                return False
        return True

    def wait_for_challenge(self, timeout_sec: float = 30.0) -> CapturedChallenge:
        """等待一轮 reload + payload，并持续驱动 Playwright 事件循环。"""
        if not self.is_automation_ready():
            raise RuntimeError("浏览器自动化未启动")
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.raise_if_automated_query_restricted()
            if self._has_paired_challenge(require_since_capture=True):
                # 两个响应必须来自本轮捕获窗口，且不接受明显跨轮的旧缓存。
                self._emit("reload 与 payload 已配对，开始构建挑战")
                challenge = self._build_challenge(mark_consumed=True)
                return challenge
            # Playwright 同步 API 的 response 回调在当前线程派发。这里调用
            # wait_for_timeout 主动让出浏览器事件循环，不能只用 Event.wait 阻塞。
            remaining_ms = max(10, int((deadline - time.monotonic()) * 1000))
            self._page.wait_for_timeout(min(100, remaining_ms))
        if not self._latest_reload:
            raise TimeoutError("等待 /reload 超时")
        if not self._latest_payload:
            raise TimeoutError("等待 /payload 超时")
        return self._build_challenge(mark_consumed=True)

    def poll_new_challenge(self) -> CapturedChallenge | None:
        """被动轮询 replaceimage 队列或新的完整 payload。

        用于会话空闲持续监控：点击后换图、dynamic 局部刷新、下一轮 multicaptcha。
        需要驱动 Playwright 事件循环，调用方应在 worker 线程周期调用本方法。
        """
        if not self.is_automation_ready() or not self.continuous_monitor:
            return None
        self.raise_if_automated_query_restricted()
        # 短暂让出事件循环，便于 response 回调写入最新字节
        try:
            self._page.wait_for_timeout(20)
        except Exception:
            return None
        if self._replacement_queue:
            challenge = self._replacement_queue.popleft()
            self._emit(
                f"交付 replaceimage 单格图：ds=[{challenge.source_tile_id}] "
                f"→ 固定格子 {challenge.source_tile_index}，"
                f"队列剩余={len(self._replacement_queue)}"
            )
            return challenge
        if not self._latest_payload or not self._has_unconsumed_payload():
            return None
        # 至少要有一份 reload 元数据（可来自本轮或上一轮）
        if not self._latest_reload:
            return None
        reload_changed = self._current_reload_sha() != self._consumed_reload_sha
        payload_sha = self._current_payload_sha() or ""
        note = (
            f"接口变化：payload 新图 sha={payload_sha[:10]}…"
            + ("，reload 同步更新" if reload_changed else "（沿用上一轮 reload 元数据）")
        )
        self._last_interface_note = note
        self._emit(note)
        return self._build_challenge(mark_consumed=True)

    def wait_for_new_challenge(self, timeout_sec: float = 10.0) -> CapturedChallenge | None:
        """主动等待点击后 replaceimage，其次才是新整图。"""
        if not self.is_automation_ready():
            return None
        baseline = self._consumed_payload_sha
        deadline = time.monotonic() + max(0.2, timeout_sec)
        while time.monotonic() < deadline:
            self.raise_if_automated_query_restricted()
            if self._replacement_queue:
                challenge = self._replacement_queue.popleft()
                self._emit(
                    f"检测到点击后 replaceimage："
                    f"ds=[{challenge.source_tile_id}] "
                    f"→ 固定格子 {challenge.source_tile_index}"
                )
                return challenge
            if (
                self._latest_payload
                and self._latest_reload
                and self._current_payload_sha() != baseline
            ):
                self._emit("检测到点击后的新 payload/reload，准备归档")
                return self._build_challenge(mark_consumed=True)
            remaining_ms = max(10, int((deadline - time.monotonic()) * 1000))
            self._page.wait_for_timeout(min(100, remaining_ms))
        return None

    def _build_challenge(self, *, mark_consumed: bool = False) -> CapturedChallenge:
        from .capture_service import parse_reload_response

        assert self._latest_reload is not None
        assert self._latest_payload is not None
        parsed = parse_reload_response(self._latest_reload)
        categories = parsed.get("categories") or []
        label = None
        if categories:
            label = categories[0].get("label") or categories[0].get("id")
        pmeta = parsed.get("pmeta")
        spec = resolve_challenge_grid(
            str(parsed.get("challenge_type") or "imageselect"),
            grid_from_pmeta(pmeta),
        )
        grid = (spec.rows, spec.columns)
        labels = [str(item.get("label") or item.get("id")) for item in categories]
        self._emit(
            f"挑战数据已生成：类型={parsed.get('challenge_type') or '未知'}，"
            f"类别={', '.join(labels) or '未知'}，网格={grid[0]}×{grid[1]}"
        )
        challenge_type = str(parsed.get("challenge_type") or "imageselect")
        # 整图代表一轮新挑战；dynamic 图块 ID 从 0 开始重建。
        # 局部 replaceimage payload 不会进入此方法，因此不会误重置。
        if challenge_type == "dynamic":
            self._reset_dynamic_tile_map(spec.count)
        else:
            self._clear_dynamic_tile_map()
        challenge = CapturedChallenge(
            reload_text=self._latest_reload,
            payload_bytes=self._latest_payload,
            payload_content_type=self._latest_payload_type,
            challenge_type=challenge_type,
            category_label=", ".join(labels) if labels else (str(label) if label else None),
            pmeta=pmeta,
            grid_rows=grid[0],
            grid_cols=grid[1],
        )
        if mark_consumed:
            self._mark_challenge_consumed()
        return challenge

    # ------------------------------------------------------------------
    # 点击
    # ------------------------------------------------------------------

    def click_checkbox(self) -> None:
        """点击 anchor 内“我不是机器人”复选框。"""
        page = self._require_page()
        self.raise_if_automated_query_restricted()
        self.reset_capture()
        self._emit("正在点击复选框…")
        # 常见结构：iframe[src*="anchor"] 内 #recaptcha-anchor / .recaptcha-checkbox-border
        frame_selectors = (
            'iframe[src*="anchor"]',
            'iframe[title*="reCAPTCHA"]',
            'iframe[src*="recaptcha"]',
        )
        candidates = [
            "#recaptcha-anchor",
            ".recaptcha-checkbox-border",
            ".recaptcha-checkbox",
            'div[role="checkbox"]',
            "#recaptcha-anchor-label",
        ]
        last_error: Exception | None = None
        for frame_sel in frame_selectors:
            anchor = page.frame_locator(frame_sel)
            for selector in candidates:
                try:
                    locator = anchor.locator(selector).first
                    locator.wait_for(state="visible", timeout=6_000)
                    locator.scroll_into_view_if_needed(timeout=3_000)
                    locator.click(timeout=5_000, force=False)
                    self._emit(f"复选框已点击（{frame_sel} → {selector}）")
                    return
                except Exception as exc:
                    last_error = exc
                    continue
        # 兜底：点整个 anchor iframe 中心偏左（复选框区域）
        try:
            box = page.locator('iframe[src*="anchor"], iframe[title*="reCAPTCHA"]').first.bounding_box()
            if box:
                page.mouse.click(box["x"] + min(28.0, box["width"] * 0.2), box["y"] + box["height"] / 2)
                self._emit("复选框已点击（坐标兜底）")
                return
        except Exception as exc:
            last_error = exc
        summary = summarize_playwright_error(
            last_error or RuntimeError("未知错误"),
            action="点击",
        )
        raise RuntimeError(f"点击复选框失败：{summary}")

    def refresh_challenge(self) -> None:
        """点击 bframe 内“换一个新的验证码”。"""
        page = self._require_page()
        self.raise_if_automated_query_restricted()
        self.reset_capture()
        bframe = page.frame_locator('iframe[src*="bframe"]')
        for selector in ("#recaptcha-reload-button", "button#recaptcha-reload-button"):
            try:
                btn = bframe.locator(selector).first
                btn.wait_for(state="visible", timeout=5_000)
                btn.click(timeout=5_000)
                self._emit("已点击换图")
                return
            except Exception:
                continue
        raise RuntimeError("未找到换图按钮")

    def click_tiles(self, indices: list[int], *, delay_ms: int = 220) -> list[int]:
        """按格子编号点击。优先 DOM 元素，失败则用几何估算。"""
        if not indices:
            return []
        page = self._require_page()
        self.raise_if_automated_query_restricted()
        unique = sorted({int(i) for i in indices if int(i) >= 0})
        clicked: list[int] = []

        # 1) DOM 优先：.rc-imageselect-tile 与索引一一对应
        dom_clicked = self._click_tiles_dom(unique, delay_ms=delay_ms)
        if dom_clicked is not None:
            return dom_clicked

        # 2) 坐标估算（页面坐标系）
        layout = self.measure_tile_layout(rows=None, columns=None)
        for index in unique:
            if index >= len(layout.cells):
                self._emit(f"跳过越界格子 {index}")
                continue
            x, y = cell_center(layout, index)
            page.mouse.click(x, y)
            clicked.append(index)
            self._emit(f"已点击格子 {index} @ ({x:.0f},{y:.0f})")
            page.wait_for_timeout(delay_ms)
        return clicked

    def _click_tiles_dom(self, indices: list[int], *, delay_ms: int) -> list[int] | None:
        """DOM 点击成功返回已点编号；不可用返回 None。"""
        page = self._require_page()
        bframe = page.frame_locator('iframe[src*="bframe"]')
        # 等挑战图出现
        try:
            bframe.locator(
                ".rc-imageselect-tile, td.rc-imageselect-tile, .rc-image-tile-target"
            ).first.wait_for(state="visible", timeout=8_000)
        except Exception:
            pass

        tiles = bframe.locator(".rc-imageselect-tile")
        try:
            count = tiles.count()
        except Exception:
            count = 0
        if count <= 0:
            tiles = bframe.locator("td.rc-imageselect-tile, .rc-image-tile-target")
            try:
                count = tiles.count()
            except Exception:
                return None
        if count <= 0:
            return None

        clicked: list[int] = []
        for index in indices:
            if index >= count:
                self._emit(f"跳过 DOM 越界格子 {index}/{count}")
                continue
            try:
                tile = tiles.nth(index)
                tile.scroll_into_view_if_needed(timeout=3_000)
                tile.click(timeout=5_000)
                clicked.append(index)
                self._emit(f"已点击 DOM 格子 {index}/{count}")
                page.wait_for_timeout(delay_ms)
            except Exception as exc:
                self._emit(
                    f"DOM 点击格子 {index} 失败："
                    f"{summarize_playwright_error(exc, action='点击')}"
                )
        return clicked

    def click_verify(self) -> None:
        """点击 bframe 验证按钮。"""
        page = self._require_page()
        bframe = page.frame_locator('iframe[src*="bframe"]')
        for selector in ("#recaptcha-verify-button", "button#recaptcha-verify-button"):
            try:
                btn = bframe.locator(selector).first
                btn.wait_for(state="visible", timeout=5_000)
                btn.click(timeout=5_000)
                self._emit("已点击验证")
                return
            except Exception:
                continue
        raise RuntimeError("未找到验证按钮")

    def measure_bframe_rect(self) -> Rect:
        page = self._require_page()
        box = page.locator('iframe[src*="bframe"]').first.bounding_box()
        if not box or box["width"] < 50 or box["height"] < 50:
            raise RuntimeError("bframe 不可见或尺寸异常")
        return Rect(box["x"], box["y"], box["width"], box["height"])

    def measure_tile_layout(
        self,
        rows: int | None = None,
        columns: int | None = None,
    ) -> TileLayout:
        """测量当前挑战图块布局。

        优先读 DOM 格子外框；失败时用 MCP 标定的 header/footer 估算。
        """
        page = self._require_page()
        bframe_rect = self.measure_bframe_rect()

        # 尝试从 pmeta / 最新 reload 推断网格
        if rows is None or columns is None:
            rows, columns = 3, 3
            if self._latest_reload:
                try:
                    from .capture_service import parse_reload_response

                    parsed = parse_reload_response(self._latest_reload)
                    spec = resolve_challenge_grid(
                        str(parsed.get("challenge_type") or "imageselect"),
                        grid_from_pmeta(parsed.get("pmeta")),
                    )
                    rows, columns = spec.rows, spec.columns
                except Exception:
                    pass

        # DOM 格子包围盒
        try:
            frame = None
            for f in page.frames:
                if "bframe" in (f.url or ""):
                    frame = f
                    break
            if frame is not None:
                boxes = frame.eval_on_selector_all(
                    ".rc-imageselect-tile, td.rc-imageselect-tile, .rc-image-tile-target",
                    """els => els.map(el => {
                        const r = el.getBoundingClientRect();
                        return {x:r.x, y:r.y, w:r.width, h:r.height};
                    })""",
                )
                if boxes and len(boxes) >= rows * columns:
                    # frame 内坐标 → 页面坐标
                    cells = []
                    for item in boxes[: rows * columns]:
                        cells.append(
                            Rect(
                                bframe_rect.x + float(item["x"]),
                                bframe_rect.y + float(item["y"]),
                                float(item["w"]),
                                float(item["h"]),
                            )
                        )
                    table = Rect(
                        min(c.x for c in cells),
                        min(c.y for c in cells),
                        max(c.x + c.width for c in cells) - min(c.x for c in cells),
                        max(c.y + c.height for c in cells) - min(c.y for c in cells),
                    )
                    return TileLayout(
                        table=table,
                        rows=rows,
                        columns=columns,
                        cells=tuple(cells),
                    )
        except Exception:
            pass

        return build_tile_layout(bframe_rect, rows, columns)

    def _require_page(self) -> Any:
        if self._page is None:
            raise RuntimeError("浏览器自动化未启动，请先“开始在线会话”")
        return self._page

    @staticmethod
    def instructions() -> str:
        return (
            "【Google Chrome 在线采集】\n"
            "1. 点击“开始在线采集”，程序用本机 Google Chrome 打开目标站并点击复选框。\n"
            "2. 程序监听 /reload 与 /payload 取得完整挑战图。\n"
            "3. 点击后监听 /replaceimage，根据请求参数 ds 把新图回填到指定格子。\n"
            "4. 新图归档后 GUI 画布自动刷新；开启“在线识别验证”后会自动跑模型。\n"
            "5. 再开启“自动点击网页图块”时，按识别坐标点击 Chrome 内图块（不点验证按钮）。\n"
            "\n"
            "【手工导入兜底】\n"
            "如果 Playwright 或 Chrome 监听不可用，可用“手动导入样本”选择图片和 reload 响应。\n"
        )


def detect_image_ext(content: bytes, content_type: str | None = None) -> str:
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    if content_type:
        lowered = content_type.lower()
        if "png" in lowered:
            return ".png"
        if "gif" in lowered:
            return ".gif"
        if "webp" in lowered:
            return ".webp"
        if "jpeg" in lowered or "jpg" in lowered:
            return ".jpg"
    return ".jpg"


def strip_xsrf_prefix(text: str) -> str:
    raw = text.strip()
    if raw.startswith(")]}'"):
        newline = raw.find("\n")
        return raw[newline + 1 :] if newline >= 0 else raw[4:]
    return raw


def safe_json_loads(text: str) -> Any:
    return json.loads(strip_xsrf_prefix(text))
