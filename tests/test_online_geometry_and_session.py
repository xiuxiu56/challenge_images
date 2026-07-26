"""在线点击几何与样本归档单测。"""

import json
from pathlib import Path

from PIL import Image

from challenge_images.online.capture_service import (
    OnlineCaptureService,
    parse_reload_response,
)
from challenge_images.online.browser_session import (
    BrowserSession,
    CapturedChallenge,
    parse_replaceimage_indices,
    summarize_playwright_error,
)
from challenge_images.online.click_geometry import (
    Rect,
    build_tile_layout,
    cell_center,
    grid_from_pmeta,
)
from challenge_images.online.online_session import OnlineSolveSession


def test_build_tile_layout_matches_mcp_measurement():
    # MCP 实测：bframe ≈ (85,1,400,580)，3x3 中心约 (160,184) 等
    bframe = Rect(85, 1, 400, 580)
    layout = build_tile_layout(bframe, 3, 3)
    assert layout.rows == 3 and layout.columns == 3
    assert len(layout.cells) == 9
    c0 = cell_center(layout, 0)
    assert abs(c0[0] - 160) < 3
    assert abs(c0[1] - 184) < 5
    c8 = cell_center(layout, 8)
    assert abs(c8[0] - 410) < 3
    assert abs(c8[1] - 434) < 5


def test_online_browser_defaults_to_visible_chrome_with_checkbox_action():
    session = BrowserSession()
    assert session.headless is False
    assert session.auto_click_checkbox is True
    opts = session._chrome_launch_options()
    # 不得依赖默认 bundled Chromium：必须带 executable_path 或 channel=chrome
    assert "executable_path" in opts or opts.get("channel") == "chrome"
    assert opts.get("headless") is False


def test_clear_site_data_clears_context_cookies_cache_and_page_storage():
    class Cdp:
        def __init__(self):
            self.commands = []

        def send(self, command, params=None):
            self.commands.append((command, params))

        def detach(self):
            self.commands.append(("detach", None))

    class Context:
        def __init__(self):
            self.cleared = 0
            self.cdp = Cdp()

        def clear_cookies(self):
            self.cleared += 1

        def new_cdp_session(self, _page):
            return self.cdp

    class Page:
        def __init__(self):
            self.url = "https://www.google.com/recaptcha/api2/demo"
            self.evaluated = []
            self.reloads = 0

        def evaluate(self, script):
            self.evaluated.append(script)

        def reload(self, **_kwargs):
            self.reloads += 1

    session = BrowserSession()
    session._page = Page()
    session._context = Context()
    session.clear_site_data()

    assert session._context.cleared == 1
    assert session._context.cdp.commands == [
        ("Network.clearBrowserCookies", None),
        ("Network.clearBrowserCache", None),
        (
            "Storage.clearDataForOrigin",
            {"origin": "https://www.google.com", "storageTypes": "all"},
        ),
        ("detach", None),
    ]
    assert len(session._page.evaluated) == 1
    assert session._page.reloads == 1


def test_grid_from_pmeta_reads_rows_cols():
    pmeta = ["pmeta", ["/m/09d_r", None, 3, 3, 3, None, "Mountain"]]
    assert grid_from_pmeta(pmeta) == (3, 3)


def test_parse_reload_from_mcp_sample():
    sample = Path(__file__).resolve().parents[1] / "data" / "online_capture" / "_mcp_reload_sample.network-response"
    if not sample.is_file():
        # CI/本地无样本时跳过
        return
    parsed = parse_reload_response(sample.read_text(encoding="utf-8"))
    assert parsed["challenge_type"] == "imageselect"
    assert parsed["categories"][0]["id"] == "/m/09d_r"
    assert parsed["grid"] == {"rows": 3, "columns": 3}


def test_import_bytes_archives_payload(tmp_path: Path):
    image_bytes = Path(__file__).resolve().parents[1] / "data" / "online_capture" / "_mcp_payload_sample.network-response"
    reload_path = Path(__file__).resolve().parents[1] / "data" / "online_capture" / "_mcp_reload_sample.network-response"
    if not image_bytes.is_file() or not reload_path.is_file():
        # 无 MCP 样本时用合成图
        from io import BytesIO

        buf = BytesIO()
        Image.new("RGB", (300, 300), "red").save(buf, format="JPEG")
        payload = buf.getvalue()
        reload_text = ")]}'\n" + json.dumps(
            [
                "rresp",
                "token",
                None,
                None,
                ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
                "imageselect",
                None,
                None,
                None,
                "payload-token",
            ]
        )
    else:
        payload = image_bytes.read_bytes()
        reload_text = reload_path.read_text(encoding="utf-8")

    service = OnlineCaptureService(tmp_path / "online")
    sample = service.import_bytes(payload, reload_text)
    assert sample.path.is_file()
    assert sample.metadata_path.is_file()
    assert sample.challenge_type in {"imageselect", "dynamic"}
    assert sample.sha256


def test_browser_response_listener_accepts_enterprise_paths_and_keeps_metadata():
    """企业版路径也应进入同一轮 reload/payload 捕获。"""

    class Request:
        def __init__(self, method: str):
            self.method = method

    class Response:
        def __init__(self, url: str, method: str, *, text: str = "", body: bytes = b""):
            self.url = url
            self.request = Request(method)
            self.status = 200
            self.headers = {"content-type": "image/jpeg"}
            self._text = text
            self._body = body

        def text(self):
            return self._text

        def body(self):
            return self._body

    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "imageselect",
            None,
            None,
            None,
            "payload-token",
        ]
    )
    payload = b"\xff\xd8\xff" + b"0" * 200
    session = BrowserSession()
    session.reset_capture()
    session._on_response(
        Response(
            "https://www.google.com/recaptcha/enterprise/reload?x=1",
            "POST",
            text=reload_text,
        )
    )
    session._on_response(
        Response(
            "https://www.google.com/recaptcha/enterprise/payload?x=1",
            "GET",
            body=payload,
        )
    )
    assert session._latest_reload == reload_text
    assert session._latest_payload == payload
    assert session._latest_payload_type == "image/jpeg"


def test_parse_replaceimage_ds_indices():
    """附件真实请求 ``ds=%5B2%5D`` 应解析为格子 2。"""
    assert parse_replaceimage_indices("v=VERSION&c=TOKEN&ds=%5B2%5D") == [2]
    assert parse_replaceimage_indices("ds=%5B2%2C5%5D") == [2, 5]
    assert parse_replaceimage_indices("v=VERSION") == []


def test_summarize_playwright_error_removes_multiline_call_log():
    error = RuntimeError(
        "Locator.click: Timeout 5000ms exceeded\n"
        "Call log:\n"
        "- element is outside of the viewport\n"
        "- iframe intercepts pointer events"
    )
    summary = summarize_playwright_error(error, action="点击")
    assert summary == "图块被其他页面元素遮挡或已离开可视区域"
    assert "Call log" not in summary
    assert "Timeout" not in summary


def test_replaceimage_ds_is_bound_to_following_payload_image():
    """replaceimage 只登记 ds，随后 payload 才是该位置的单格图。"""
    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "dynamic",
        ]
    )

    class Request:
        def __init__(self, method: str, post_data: str | None = None):
            self.method = method
            self.post_data = post_data

    class Response:
        def __init__(self, url: str, method: str, body: bytes, post_data=None):
            self.url = url
            self.request = Request(method, post_data)
            self.request.url = url
            self.status = 200
            self.headers = {"content-type": "image/jpeg"}
            self._body = body

        def body(self):
            return self._body

    session = BrowserSession()
    session._latest_reload = reload_text
    initial = b"\xff\xd8\xff" + b"I" * 300
    session._latest_payload = initial
    session._consumed_payload_sha = session._current_payload_sha()
    # replaceimage 响应体不是图片，它只让程序记住 ds=[7]。
    replace_response = Response(
        "https://www.google.com/recaptcha/api2/replaceimage?k=KEY",
        "POST",
        b"NOT_AN_IMAGE",
        "v=VERSION&c=TOKEN&ds=%5B7%5D",
    )
    session._on_request(replace_response.request)
    session._on_response(replace_response)
    assert list(session._pending_replace_indices) == [7]
    assert len(session._replacement_queue) == 0

    tile = b"\xff\xd8\xff" + b"M" * 200
    session._on_response(
        Response(
            "https://www.google.com/recaptcha/api2/payload?p=TOKEN",
            "GET",
            tile,
        )
    )

    assert session._latest_payload == initial
    assert list(session._pending_replace_indices) == []
    assert len(session._replacement_queue) == 1
    challenge = session._replacement_queue.popleft()
    assert challenge.source_tile_id == 7
    assert challenge.source_tile_index == 7
    assert challenge.payload_bytes == tile


def test_repeated_dynamic_tile_ids_map_to_fixed_grid_position():
    """同一固定格子重复替换时，ds 递增仍应回填原位置。"""
    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "dynamic",
        ]
    )

    class Request:
        def __init__(self, method: str, post_data: str | None = None):
            self.method = method
            self.post_data = post_data
            self.url = "https://www.google.com/recaptcha/api2/replaceimage?k=KEY"

    class Response:
        def __init__(self, url: str, method: str, body: bytes, post_data=None):
            self.url = url
            self.request = Request(method, post_data)
            self.request.url = url
            self.status = 200
            self.headers = {"content-type": "image/jpeg"}
            self._body = body

        def body(self):
            return self._body

    session = BrowserSession()
    session._latest_reload = reload_text
    session._reset_dynamic_tile_map(9)

    # 真实时序：0 替换后生成 ID=9，4 替换后生成 ID=10；
    # 之后同一固定位置 4 会以 ds=10、11 继续请求。
    raw_ids = [0, 4, 10, 11]
    for order, raw_id in enumerate(raw_ids):
        request = Request("POST", f"v=VERSION&ds=%5B{raw_id}%5D")
        session._on_request(request)
        session._on_response(
            Response(
                f"https://www.google.com/recaptcha/api2/payload?p={order}",
                "GET",
                b"\xff\xd8\xff" + bytes([65 + order]) * 200,
            )
        )

    challenges = list(session._replacement_queue)
    assert [item.source_tile_id for item in challenges] == [0, 4, 10, 11]
    assert [item.source_tile_index for item in challenges] == [0, 4, 4, 4]
    assert session._dynamic_tile_positions[9] == 0
    assert session._dynamic_tile_positions[10] == 4
    assert session._dynamic_tile_positions[11] == 4
    assert session._dynamic_tile_positions[12] == 4


def test_wait_for_challenge_drives_playwright_event_loop():
    """等待网络响应期间需要让 Playwright 有机会派发 response 回调。"""

    class Page:
        def __init__(self, session: BrowserSession):
            self.session = session
            self.calls = 0

        def wait_for_timeout(self, _milliseconds: int):
            self.calls += 1
            if self.calls == 1:
                reload_text = ")]}'\n" + json.dumps(
                    [
                        "rresp", "token", None, 120,
                        ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
                        "imageselect", None, None, None, "payload-token",
                    ]
                )
                self.session._latest_reload = reload_text
                self.session._latest_payload = b"\xff\xd8\xff" + b"0" * 200
                now = self.session._capture_started_at + 0.001
                self.session._reload_received_at = now
                self.session._payload_received_at = now

    session = BrowserSession()
    session.reset_capture()
    page = Page(session)
    session._page = page
    challenge = session.wait_for_challenge(timeout_sec=0.5)
    assert page.calls >= 1
    assert challenge.challenge_type == "imageselect"
    assert challenge.category_label == "Crosswalk"
    # 交付后应标记已消费，避免空闲监控重复推送同一张图
    assert session._consumed_payload_sha is not None
    assert session.poll_new_challenge() is None


def test_poll_and_wait_for_new_challenge_after_payload_change():
    """持续监控 / 点击后监控：payload 哈希变化时交付新挑战。"""

    class Page:
        def wait_for_timeout(self, _milliseconds: int):
            return None

    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "imageselect",
            None,
            None,
            None,
            "payload-token",
        ]
    )
    first_payload = b"\xff\xd8\xff" + b"A" * 200
    second_payload = b"\xff\xd8\xff" + b"B" * 200

    session = BrowserSession()
    session._page = Page()
    session.continuous_monitor = True
    session.reset_capture()
    session._latest_reload = reload_text
    session._latest_payload = first_payload
    session._reload_received_at = session._capture_started_at + 0.001
    session._payload_received_at = session._capture_started_at + 0.001
    first = session._build_challenge(mark_consumed=True)
    assert first.payload_bytes == first_payload
    assert session.poll_new_challenge() is None

    session._latest_payload = second_payload
    session._payload_received_at = session._capture_started_at + 0.002
    polled = session.poll_new_challenge()
    assert polled is not None
    assert polled.payload_bytes == second_payload
    assert session._consumed_payload_sha is not None

    # 再次变化时，wait_for_new_challenge 也能拿到
    third_payload = b"\xff\xd8\xff" + b"C" * 200
    session._latest_payload = third_payload

    class Page2:
        def __init__(self, browser: BrowserSession):
            self.browser = browser
            self.calls = 0

        def wait_for_timeout(self, _milliseconds: int):
            self.calls += 1
            # 第一次循环时 payload 已是 third，应立即返回

    session._page = Page2(session)
    waited = session.wait_for_new_challenge(timeout_sec=0.5)
    assert waited is not None
    assert waited.payload_bytes == third_payload


def test_dynamic_clicks_keep_payload_and_tile_index_paired(tmp_path: Path):
    """动态多格点击应逐格等待新图，保留每张图的目标索引。"""
    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "dynamic",
        ]
    )

    class DynamicBrowser:
        def __init__(self):
            self._latest_reload = reload_text
            self.clicked = []
            self.sequence = 0

        def is_automation_ready(self):
            return True

        def click_tiles(self, indices, *, delay_ms):
            del delay_ms
            self.clicked.extend(indices)
            return list(indices)

        def wait_for_new_challenge(self, timeout_sec):
            del timeout_sec
            self.sequence += 1
            image = Image.new("RGB", (100, 100), (self.sequence * 40, 0, 0))
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return CapturedChallenge(
                reload_text=reload_text,
                payload_bytes=buffer.getvalue(),
                payload_content_type="image/png",
                challenge_type="dynamic",
                category_label="Crosswalk",
                grid_rows=3,
                grid_cols=3,
            )

    browser = DynamicBrowser()
    capture = OnlineCaptureService(tmp_path / "online")
    session = OnlineSolveSession(capture=capture, browser=browser)

    clicked, follow_ups = session.apply_clicks([5, 2], watch_after_ms=500)

    assert clicked == [2, 5]
    assert browser.clicked == [2, 5]
    assert [item.replacement_index for item in follow_ups] == [2, 5]
    assert [item.replacement_order for item in follow_ups] == [1, 2]
    assert all(item.replacement_total == 2 for item in follow_ups)
    assert all("replacements" in item.sample.path.parts for item in follow_ups)


def test_online_session_archives_replaceimage_payload_with_ds_metadata(tmp_path: Path):
    """会话归档应把 replaceimage 单格图写入独立目录并保留双重索引。"""
    reload_text = ")]}'\n" + json.dumps(
        [
            "rresp",
            "token",
            None,
            120,
            ["pmeta", ["/m/014xcs", None, 3, 3, 3, None, "Crosswalk"]],
            "dynamic",
        ]
    )
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (100, 100), "yellow").save(buffer, format="JPEG")
    challenge = CapturedChallenge(
        reload_text=reload_text,
        payload_bytes=buffer.getvalue(),
        payload_content_type="image/jpeg",
        challenge_type="dynamic",
        category_label="Crosswalk",
        pmeta=json.loads(reload_text.split("\n", 1)[1])[4],
        grid_rows=3,
        grid_cols=3,
        source_tile_id=10,
        source_tile_index=4,
    )
    session = OnlineSolveSession(
        capture=OnlineCaptureService(tmp_path / "online"),
        browser=BrowserSession(),
    )

    sample = session.archive_challenge(challenge)

    assert sample.path.parent == tmp_path / "online" / "replacements" / "dynamic" / "人行横道"
    assert sample.source_tile_id == 10
    assert sample.source_tile_index == 4
    sidecar = json.loads(sample.metadata_path.read_text(encoding="utf-8"))
    assert sidecar["pmeta"] == challenge.pmeta
    assert sidecar["source_tile_id"] == 10
    assert sidecar["source_tile_index"] == 4


def test_late_dynamic_payload_keeps_pending_tile_index(tmp_path: Path):
    """主动等待超时后，迟到 payload 仍应带回原点击格子。"""
    reload_text = ")]}'\n" + json.dumps(
        ["rresp", "token", None, 120, ["pmeta", ["/m/014xcs", None, 3, 3, 3]], "dynamic"]
    )
    image = Image.new("RGB", (100, 100), "green")
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    challenge = CapturedChallenge(
        reload_text=reload_text,
        payload_bytes=buffer.getvalue(),
        payload_content_type="image/png",
        challenge_type="dynamic",
        grid_rows=3,
        grid_cols=3,
        source_tile_id=10,
        source_tile_index=4,
    )

    class LateBrowser:
        _latest_reload = reload_text

        def is_automation_ready(self):
            return True

        def click_tiles(self, indices, *, delay_ms):
            del delay_ms
            return list(indices)

        def wait_for_new_challenge(self, timeout_sec):
            del timeout_sec
            return None

        def poll_new_challenge(self):
            return challenge

    session = OnlineSolveSession(
        capture=OnlineCaptureService(tmp_path / "online"),
        browser=LateBrowser(),
    )
    clicked, follow_ups = session.apply_clicks([4, 6], watch_after_ms=200)
    assert clicked == [4]
    assert follow_ups == []

    late = session.poll_new_challenge()
    assert late is not None
    assert late.source == "post_click"
    assert late.replacement_index == 4
    assert late.sample.source_tile_id == 10
    assert "replacements" in late.sample.path.parts
    assert session._pending_replacements == []


def test_pmeta_grid_indices_cover_all_challenge_types():
    """行列在 pmeta 分类行的下标 3,4，不是 2,3。

    旧实现读 [2],[3]，只在 3×3 题型偶然命中（该位置恰好也是 3），
    对全部 4×4 挑战返回 None。实测 8626 条在线记录，下标 [3],[4]
    命中率 100%。
    """
    from challenge_images.online.click_geometry import grid_from_pmeta

    # dynamic：真实样本，[3],[4] = 3,3
    dynamic = ["pmeta", ["/m/09d_r", None, 3, 3, 3, None, "Mountain"]]
    assert grid_from_pmeta(dynamic) == (3, 3)

    # tileselect / multicaptcha：真实样本，[3],[4] = 4,4，而 [2] = 2
    four_by_four = ["pmeta", ["/m/0199g", None, 2, 4, 4]]
    assert grid_from_pmeta(four_by_four) == (4, 4)

    # multicaptcha 多步骤：分类行嵌套在更深的层级里
    nested = ["pmeta", None, None, None, None, [[
        ["/m/0199g", None, 2, 4, 4],
        ["/m/04_sv", None, 2, 4, 4],
    ]]]
    assert grid_from_pmeta(nested) == (4, 4)


def test_pmeta_grid_rejects_invalid_values():
    from challenge_images.online.click_geometry import grid_from_pmeta

    assert grid_from_pmeta(None) is None
    assert grid_from_pmeta([]) is None
    assert grid_from_pmeta(["pmeta"]) is None
    # 行列不在 3/4 范围内时不采信。
    assert grid_from_pmeta(["pmeta", ["/m/0199g", None, 2, 9, 9]]) is None
    # 分类行过短。
    assert grid_from_pmeta(["pmeta", ["/m/0199g", None]]) is None
