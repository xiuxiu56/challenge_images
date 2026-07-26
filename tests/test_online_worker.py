"""在不启动真实浏览器的情况下测试在线自动刷新调度。
"""

from challenge_images.online.online_worker import (
    AUTO_REFRESH_INTERVAL_SEC,
    CHECKBOX_MONITOR_INTERVAL_SEC,
    SITE_DATA_CLEAR_INTERVAL_SEC,
    _OnlineEngine,
)
from challenge_images.online.browser_session import detect_automated_query_restriction


class _FakeBrowser:
    continuous_monitor = True

    def __init__(self) -> None:
        self.refresh_count = 0
        self.clear_site_data_count = 0
        self.click_checkbox_count = 0
        self.challenge_open = True
        self.checkbox_available = False
        self.restriction_message = None

    def is_automation_ready(self) -> bool:
        return True

    def refresh_challenge(self) -> None:
        self.refresh_count += 1

    def clear_site_data(self) -> None:
        self.clear_site_data_count += 1

    def is_graphic_challenge_open(self) -> bool:
        return self.challenge_open

    def checkbox_needs_click(self) -> bool:
        return self.checkbox_available and not self.challenge_open

    def click_checkbox(self) -> None:
        self.click_checkbox_count += 1

    def automated_query_restriction_message(self):
        return self.restriction_message


class _FakeSession:
    def __init__(self) -> None:
        self.browser = _FakeBrowser()
        self.stop_count = 0

    def poll_new_challenge(self):
        return None

    def stop(self) -> None:
        self.stop_count += 1


def test_detects_chinese_and_english_automated_query_messages():
    assert detect_automated_query_restriction(
        "您的计算机或网络可能在发送自动查询内容。"
    )
    assert detect_automated_query_restriction(
        "Your computer or network may be sending automated queries."
    )
    assert not detect_automated_query_restriction("请选择包含公共汽车的所有图片")


def test_query_restriction_stops_all_online_schedulers_and_session():
    engine = _OnlineEngine()
    fake = _FakeSession()
    fake.browser.restriction_message = "检测到自动查询限制"
    engine.session = fake
    engine._configure_auto_refresh(True, emit_status=False)
    engine._configure_site_data_clear(True, emit_status=False)
    engine._configure_checkbox_monitor(True, emit_status=False)
    messages = []
    engine.query_restricted.connect(messages.append)

    engine._poll_idle()

    assert fake.stop_count == 1
    assert engine._auto_refresh_enabled is False
    assert engine._site_data_clear_enabled is False
    assert engine._checkbox_monitor_enabled is False
    assert messages == ["检测到自动查询限制"]


def test_auto_refresh_waits_three_seconds_and_clicks_once():
    engine = _OnlineEngine()
    fake = _FakeSession()
    engine.session = fake
    engine._configure_auto_refresh(True, interval_sec=AUTO_REFRESH_INTERVAL_SEC, emit_status=False)
    first_deadline = engine._next_auto_refresh_at

    assert engine._maybe_auto_refresh(now=first_deadline - 0.01) is False
    assert fake.browser.refresh_count == 0
    assert engine._maybe_auto_refresh(now=first_deadline) is True
    assert fake.browser.refresh_count == 1
    assert engine._maybe_auto_refresh(now=first_deadline + 0.01) is False
    assert fake.browser.refresh_count == 1


def test_auto_refresh_disabled_does_not_click_browser():
    engine = _OnlineEngine()
    fake = _FakeSession()
    engine.session = fake
    engine._configure_auto_refresh(False, emit_status=False)

    assert engine._maybe_auto_refresh(now=10_000.0) is False
    assert fake.browser.refresh_count == 0


def test_site_data_cleanup_runs_at_three_minute_interval():
    engine = _OnlineEngine()
    fake = _FakeSession()
    engine.session = fake
    engine._configure_site_data_clear(
        True,
        interval_sec=SITE_DATA_CLEAR_INTERVAL_SEC,
        emit_status=False,
    )
    deadline = engine._next_site_data_clear_at

    assert engine._maybe_clear_site_data(now=deadline - 0.01) is False
    assert fake.browser.clear_site_data_count == 0
    assert engine._maybe_clear_site_data(now=deadline) is True
    assert fake.browser.clear_site_data_count == 1


def test_checkbox_monitor_requires_five_seconds_of_continuous_closure():
    engine = _OnlineEngine()
    fake = _FakeSession()
    fake.browser.challenge_open = False
    fake.browser.checkbox_available = True
    engine.session = fake
    engine._configure_checkbox_monitor(
        True,
        interval_sec=CHECKBOX_MONITOR_INTERVAL_SEC,
        emit_status=False,
    )
    deadline = engine._next_checkbox_check_at

    assert engine._maybe_monitor_checkbox(now=deadline - 0.01) is False
    assert fake.browser.click_checkbox_count == 0
    # 第一次发现关闭只开始计时，不立即点击。
    assert engine._maybe_monitor_checkbox(now=deadline) is False
    assert fake.browser.click_checkbox_count == 0
    second_deadline = engine._next_checkbox_check_at
    assert engine._maybe_monitor_checkbox(now=second_deadline) is True
    assert fake.browser.click_checkbox_count == 1

    fake.browser.challenge_open = True
    next_deadline = engine._next_checkbox_check_at
    assert engine._maybe_monitor_checkbox(now=next_deadline) is False
    assert fake.browser.click_checkbox_count == 1


def test_checkbox_monitor_cancels_retry_when_challenge_recovers():
    engine = _OnlineEngine()
    fake = _FakeSession()
    fake.browser.challenge_open = False
    fake.browser.checkbox_available = True
    engine.session = fake
    engine._configure_checkbox_monitor(
        True,
        interval_sec=CHECKBOX_MONITOR_INTERVAL_SEC,
        emit_status=False,
    )

    first_deadline = engine._next_checkbox_check_at
    assert engine._maybe_monitor_checkbox(now=first_deadline) is False
    assert engine._checkbox_missing_since == first_deadline

    # 模拟动态图块更新期间 iframe 短暂消失后恢复。
    fake.browser.challenge_open = True
    fake.browser.checkbox_available = False
    second_deadline = engine._next_checkbox_check_at
    assert engine._maybe_monitor_checkbox(now=second_deadline) is False
    assert engine._checkbox_missing_since is None
    assert fake.browser.click_checkbox_count == 0
