"""在线采集页。

从 ``qt_gui.py`` 抽出：这一页管理 Playwright 会话、接收捕获到的挑战图、
执行识别并把结果点击回网页。与工作线程的信号交互也集中在这里。
"""

from __future__ import annotations

import hashlib

from typing import TYPE_CHECKING, Any

from ..config import ASSETS_DIR, DEFAULT_DEVICE, ROOT
from ..grid.grid_engine import draw_grid, parse_grid, replace_grid_tile, resolve_challenge_grid
from ..online import BrowserSession
from ..online.online_worker import AUTO_REFRESH_INTERVAL_SEC, CHECKBOX_MONITOR_INTERVAL_SEC, SITE_DATA_CLEAR_INTERVAL_SEC
from ..recognition import PARAMETER_PRESET_LABELS, build_click_plan
from .state import RecognitionOutcome, RecognitionRequest, SOURCE_ONLINE
from .theme import set_button_role
from .widgets import _preview_pixmap
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QGroupBox, QHBoxLayout, QMessageBox, QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget
from datetime import datetime
from io import BytesIO
from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover
    # mixin 总是被混入 QMainWindow；声明基类让需要 QWidget 的调用
    # 通过类型检查，运行时仍是普通 mixin。
    from PySide6.QtWidgets import QWidget as _MixinBase
else:
    _MixinBase = object


class OnlineTabMixin(_MixinBase):
    """在线采集页：浏览器会话、样本归档、识别与图块点击。"""

    if TYPE_CHECKING:  # pragma: no cover
        # 以下属性与方法由宿主窗口 QtChallengeGUI 提供。
        # 显式声明让类型检查通过，同时把隐式耦合变成可读的契约。
        online_image: Any
        online_sample: Any
        online_image_sha256: Any
        online_predictions: Any
        online_all_predictions: Any
        auto_click_tiles: Any
        auto_refresh_challenge: Any
        browser_session: Any
        clear_site_data: Any
        monitor_checkbox: Any
        online_canvas: Any
        online_capture: Any
        online_category: Any
        online_data: Any
        online_detail: Any
        online_enabled: Any
        online_grid: Any
        online_image_info: Any
        online_status: Any
        online_status_summary: Any
        online_target: Any
        online_type: Any
        online_worker: Any
        parameter_preset: Any
        recognition_mode: Any
        service: Any
        status: Any
        tabs: Any
        def _choose_data_directory(self, *args: Any, **kwargs: Any) -> Any: ...
        def _click_settings(self, *args: Any, **kwargs: Any) -> Any: ...
        def _compose_report(self, *args: Any, **kwargs: Any) -> Any: ...
        def _display_path(self, *args: Any, **kwargs: Any) -> Any: ...
        def _parameter_preset_changed(self, *args: Any, **kwargs: Any) -> Any: ...
        def _recognition_parameters(self, *args: Any, **kwargs: Any) -> Any: ...
        def _selected_weight(self, *args: Any, **kwargs: Any) -> Any: ...
        def _show_actual_recognition_route(self, *args: Any, **kwargs: Any) -> Any: ...
        def _submit_recognition(self, *args: Any, **kwargs: Any) -> Any: ...

    def _build_online_tab(self) -> None:
        """构建在线采集工作台，配置集中放在“设置配置”页。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        preview_box = QGroupBox("最新在线图片")
        preview_box.setMinimumWidth(480)
        preview = QVBoxLayout(preview_box)
        preview.setContentsMargins(10, 14, 10, 10)
        preview.addWidget(self.online_image_info)
        preview.addWidget(self.online_canvas, 1)
        right_tabs = QTabWidget()
        right_tabs.setMinimumWidth(480)
        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(5, 5, 5, 5)
        result_layout.addWidget(self.online_detail, 1)
        log_page = QWidget()
        log = QVBoxLayout(log_page)
        log.setContentsMargins(5, 5, 5, 5)
        log.addWidget(self.online_status_summary)
        log.addWidget(self.online_status, 1)
        clear_button = QPushButton("清空状态日志")
        clear_button.clicked.connect(self._clear_online_status)
        log.addWidget(clear_button)
        right_tabs.addTab(result_page, "识别结果")
        right_tabs.addTab(log_page, "在线状态")
        self.online_right_tabs = right_tabs
        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(preview_box)
        body.addWidget(right_tabs)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        body.setSizes([660, 660])
        self.online_body_splitter = body
        layout.addWidget(body, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        for text, handler in (
            ("开始在线采集", self._open_online_demo),
            ("关闭浏览器会话", self._stop_online_session),
            ("手动导入样本", self._import_online_sample),
            ("加载在线模型", self._load_online_model),
            ("识别当前图片", self._recognize_online),
            ("点击识别图块", self._click_online_predictions),
            ("复制在线结果", self._copy_online_result),
        ):
            button = QPushButton(text)
            if text in {"开始在线采集", "识别当前图片"}:
                set_button_role(button, "primary")
            if text == "识别当前图片":
                self.online_recognize_button = button
            button.clicked.connect(handler)
            actions.addWidget(button)
        layout.insertLayout(0, actions)
        self.tabs.addTab(tab, "在线采集")

    def _open_online_demo(self) -> None:
        """启动 Google Chrome：打开目标站 → 点复选框 → 拦截并归档。"""
        self.tabs.setCurrentIndex(1)
        self.online_right_tabs.setCurrentIndex(1)
        if not (BrowserSession.playwright_available() and BrowserSession.chrome_available()):
            # 无 Playwright 时退回系统 Google Chrome + 人工导入说明
            opened = self.browser_session.open(use_playwright=False)
            message = "在线状态：系统 Google Chrome 已打开（半自动）" if opened else "在线状态：浏览器启动未确认"
            self._set_online_status(message)
            QMessageBox.information(self, "在线采集步骤", self.browser_session.instructions())
            return
        checkbox_mode = "自动点击复选框" if self.monitor_checkbox.isChecked() else "等待人工点击复选框"
        self._set_online_status(f"正在启动 Google Chrome，{checkbox_mode}并等待挑战图片…")
        self.online_worker.request_start(
            headless=False,
            auto_click_checkbox=self.monitor_checkbox.isChecked(),
            capture_timeout=120.0,
            auto_refresh=self.auto_refresh_challenge.isChecked(),
            refresh_interval=AUTO_REFRESH_INTERVAL_SEC,
            clear_site_data=self.clear_site_data.isChecked(),
            monitor_checkbox=self.monitor_checkbox.isChecked(),
            capture_root=self.online_data.text(),
        )

    def _online_auto_refresh_changed(self, checked: bool) -> None:
        """切换每3秒点击一次 Chrome 挑战刷新按钮。"""
        self.online_worker.request_auto_refresh(
            bool(checked),
            interval=AUTO_REFRESH_INTERVAL_SEC,
        )

    def _online_checkbox_monitor_changed(self, checked: bool) -> None:
        """切换挑战关闭后的复选框重试监控。"""
        self.online_worker.request_checkbox_monitor(
            bool(checked),
            interval=CHECKBOX_MONITOR_INTERVAL_SEC,
        )

    def _online_site_data_clear_changed(self, checked: bool) -> None:
        """切换每三分钟清理当前在线上下文网站数据。"""
        self.online_worker.request_site_data_clear(
            bool(checked),
            interval=SITE_DATA_CLEAR_INTERVAL_SEC,
        )

    def _stop_online_session(self) -> None:
        self._set_online_status("正在关闭会话…")
        self.online_worker.request_stop()

    def _set_online_status(self, message: str) -> None:
        """追加在线阶段日志，同时更新当前阶段摘要。"""
        clean = str(message).strip()
        if clean.startswith("在线状态："):
            clean = clean[len("在线状态：") :].strip()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.online_status.appendPlainText(f"[{timestamp}] {clean}")
        self.online_status_summary.setText(f"在线状态：{clean}")

    def _clear_online_status(self) -> None:
        """清空在线日志，并保留一条新的初始状态。"""
        self.online_status.clear()
        self._set_online_status("状态日志已清空")

    def _on_online_status(self, message: str) -> None:
        self._set_online_status(message)
        self.status.setText(message)

    def _on_online_failed(self, message: str) -> None:
        self.tabs.setCurrentIndex(1)
        self.online_right_tabs.setCurrentIndex(1)
        self._set_online_status(f"失败 - {message}")
        self.status.setText(f"在线失败：{message}")
        QMessageBox.warning(self, "在线会话", message)

    def _on_online_started(self, info: dict) -> None:
        mode = "Google Chrome 自动化" if info.get("automation") else "半自动"
        self._set_online_status(f"已启动（{mode}），归档目录={info.get('capture_root')}")

    def _on_online_challenge(self, result) -> None:
        """后台捕获/持续监控完成：自动载入样本；识别/点图块由开关控制。"""
        sample = result.sample
        challenge = result.challenge
        source = str(getattr(result, "source", "initial") or "initial")
        source_label = {
            "initial": "首轮捕获",
            "manual": "手动捕获",
            "poll": "持续监控",
            "post_click": "点击后新图",
            "refresh": "换图",
        }.get(source, source)
        if challenge.challenge_type:
            self.online_type.setCurrentText(str(challenge.challenge_type))
        if challenge.category_label:
            self.online_category.setText(str(challenge.category_label))
        # 根据 pmeta 网格同步 GUI 网格
        detected = (
            int(getattr(challenge, "grid_rows", 0) or 0),
            int(getattr(challenge, "grid_cols", 0) or 0),
        )
        # 图片尺寸是最可靠的网格信号，优先于挑战类型与 pmeta。
        # Image.open 只解析文件头即可拿到尺寸，不会读入像素。
        image_size: tuple[int, int] | None = None
        try:
            with Image.open(sample.path) as probe:
                image_size = probe.size
        except (OSError, AttributeError):
            image_size = None
        spec = resolve_challenge_grid(
            challenge.challenge_type,
            detected,
            image_size=image_size,
        )
        rows, cols = spec.rows, spec.columns
        self.online_grid.setCurrentText(spec.text)
        self._set_online_status(
            f"[{source_label}] 挑战信息已到达：类型={challenge.challenge_type or '未知'}，"
            f"类别={challenge.category_label or '未知'}，网格={rows}×{cols}"
        )
        replacement_index = getattr(result, "replacement_index", None)
        if source == "post_click" and replacement_index is not None and self.online_image is not None:
            self._apply_online_replacement(result, source_label)
            return
        # 新图（含点击后/空闲监控）一律重新载入 GUI 画布
        self._show_online_sample(sample, source=source_label)

    def _apply_online_replacement(self, result, source_label: str) -> None:
        """把 dynamic 点击后的单格 payload 回填到当前整图。"""
        index = int(result.replacement_index)
        index_text = str(index) if index >= 0 else "未读取"
        try:
            replacement = Image.open(BytesIO(result.challenge.payload_bytes)).convert("RGB")
            spec = parse_grid(self.online_grid.currentText())
            # 某些轮次会直接返回新整图；只有明显小于整图时才按单格合成。
            base_image = self.online_image
            is_full_image = base_image is not None and (
                replacement.width >= base_image.width * 0.75
                and replacement.height >= base_image.height * 0.75
            )
            if is_full_image:
                self.online_image = replacement
                update_text = f"已加载点击后整图（触发格子 {index}）"
            elif self.online_image is not None:
                self.online_image = replace_grid_tile(
                    self.online_image,
                    replacement,
                    spec,
                    index,
                )
                update_text = f"已根据 replaceimage ds 将新图回填到格子 {index}"
            if self.online_image is None:
                return
            self.online_image_sha256 = hashlib.sha256(self.online_image.tobytes()).hexdigest()
            self.online_predictions = []
            self.online_all_predictions = []
            self._render_online([])
            order = int(getattr(result, "replacement_order", 0) or 0)
            total = int(getattr(result, "replacement_total", 0) or 0)
            progress = f"（{order}/{total}）" if total else ""
            raw_path = self._display_path(result.sample.path)
            self._set_online_status(
                f"[{source_label}] {update_text}{progress}；单格原图已保存：{raw_path}"
            )
            self.status.setText(f"在线图片已更新：格子 {index_text}{progress}")

            # 一次点多格时，等全部替换图合成完成后再识别一次。
            is_last = total <= 0 or order >= total
            if is_last and self.online_enabled.isChecked():
                if self.service.loaded:
                    self._set_online_status(f"[{source_label}] 合成完成，正在识别更新后的整图")
                    self._recognize_online()
                else:
                    self._set_online_status(f"[{source_label}] 整图已更新，在线模型尚未加载")
        except Exception as exc:
            self._set_online_status(f"[{source_label}] 格子 {index_text} 新图合成失败：{exc}")
            QMessageBox.warning(self, "在线图片更新", str(exc))

    def _on_online_stopped(self) -> None:
        self._set_online_status("会话已关闭")
        self.status.setText("在线会话已关闭")

    def _on_online_query_restricted(self, message: str) -> None:
        """出现自动查询限制提示时关闭所有在线自动功能。"""
        for checkbox in (
            self.online_enabled,
            self.auto_click_tiles,
            self.auto_refresh_challenge,
            self.monitor_checkbox,
            self.clear_site_data,
        ):
            previous = checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(previous)
        self.tabs.setCurrentIndex(1)
        self.online_right_tabs.setCurrentIndex(1)
        reason = str(message).strip() or "检测到自动查询限制提示"
        self._set_online_status(f"已紧急停止：{reason}")
        self.status.setText("在线采集已停止：页面出现自动查询限制提示")
        QMessageBox.warning(self, "在线采集已停止", reason)

    def _on_online_clicks_done(self, indices: list[int]) -> None:
        """统一显示手动或自动网页图块点击结果；新图由 challenge_ready 刷新。"""
        clicked = sorted({int(index) for index in indices})
        if clicked:
            message = (
                f"网页图块点击完成：{clicked}；"
                "若随后有新图加载，会自动归档并刷新 GUI"
            )
        else:
            message = "网页图块未点击，请确认挑战页和图块仍在显示"
        self._set_online_status(message)
        self.status.setText(message)

    def _import_online_sample(self) -> None:
        """导入人工导出的 payload 图片和可选 reload 响应。"""
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 payload 挑战图片",
            str(ROOT),
            "图片 (*.jpg *.jpeg *.png *.webp *.bmp);;所有文件 (*)",
        )
        if not image_path:
            return
        reload_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 reload 响应正文（可取消）",
            str(ROOT),
            "响应文本 (*.txt *.json);;所有文件 (*)",
        )
        try:
            self._set_online_status("正在解析并归档样本")
            sample = self.online_capture.import_sample(
                image_path=image_path,
                reload_response_path=reload_path or None,
                challenge_type=self.online_type.currentText(),
                category=self.online_category.text().strip() or None,
            )
            self._show_online_sample(sample)
        except Exception as exc:
            self._set_online_status(f"导入失败 - {exc}")
            QMessageBox.critical(self, "在线样本导入失败", str(exc))

    def _show_online_sample(self, sample, *, source: str = "归档") -> None:
        """把归档样本载入在线页面，不改动离线页面状态。

        同一路径重复推送时跳过重载；不同文件（点击后新图）会刷新画布。
        """
        previous_path = None
        if self.online_sample is not None:
            previous_path = str(getattr(self.online_sample, "path", "") or "")
        new_path = str(sample.path)
        if previous_path and previous_path == new_path and self.online_image is not None:
            self._set_online_status(f"[{source}] 图片未变化，跳过重复载入：{sample.path.name}")
            return

        self._set_online_status(f"[{source}] 正在载入 GUI 图片和在线元数据")
        self.online_sample = sample
        self.online_image = Image.open(sample.path).convert("RGB")
        self.online_image_sha256 = hashlib.sha256(self.online_image.tobytes()).hexdigest()
        self.online_predictions = []
        self.online_all_predictions = []
        self.online_type.setCurrentText(sample.challenge_type)
        # 手动导入没有 challenge 对象时，仍按类型设置正确网格。
        self.online_grid.setCurrentText(resolve_challenge_grid(sample.challenge_type).text)
        self.online_target.setText(sample.target_class)
        self._apply_online_profile()
        online_preview = draw_grid(
            self.online_image,
            parse_grid(self.online_grid.currentText()),
            [],
            [],
            ASSETS_DIR / "image.png",
        )
        self.online_canvas.setPixmap(_preview_pixmap(online_preview))
        self.online_image_info.setText(
            f"图片：{self._display_path(sample.path)}\n"
            f"挑战类型：{sample.challenge_type}  |  网格：{self.online_grid.currentText()}\n"
            f"类别：{sample.category_zh} / {sample.category_en}  |  mid：{sample.category_mid or '未知'}\n"
            f"来源：{source}\n"
            f"元数据：{self._display_path(sample.metadata_path)}"
        )
        self.online_detail.setPlainText(
            f"图片: {self._display_path(sample.path)}\n"
            f"来源: {source}\n"
            f"挑战类型: {sample.challenge_type}\n"
            f"网格: {self.online_grid.currentText()}\n"
            f"类别: {sample.category_zh} / {sample.category_en}\n"
            f"mid: {sample.category_mid or '未知'}\n"
            f"SHA-256: {sample.sha256}\n"
            f"元数据: {self._display_path(sample.metadata_path)}\n\n"
            "识别结果待生成。"
        )
        self.status.setText(f"在线样本已归档（{source}）：{self._display_path(sample.path)}")
        self._set_online_status(
            f"[{source}] GUI 图片已更新：类型={sample.challenge_type}，类别={sample.category_zh} / "
            f"{sample.category_en}，mid={sample.category_mid or '未知'}，文件={sample.path.name}"
        )
        if self.online_enabled.isChecked():
            if self.service.loaded:
                self._set_online_status(f"[{source}] 正在运行模型识别")
                self._recognize_online()
            else:
                self._set_online_status(f"[{source}] 图片已载入，但在线模型尚未加载")
                self.status.setText("在线图片已载入；请在在线页加载模型。")

    def _maybe_apply_online_clicks(self, indices: list[int]) -> None:
        """识别完成后按统一 3×3 / 4×4 点击计划应用到浏览器。"""
        if not self.auto_click_tiles.isChecked():
            return
        if not indices:
            self._set_online_status("识别无命中格子，跳过网页点击")
            return
        if not self.online_worker.session.browser.is_automation_ready():
            self._set_online_status("浏览器未就绪，无法点击网页图块")
            return
        plan = self._online_click_plan(indices)
        self._submit_online_click_plan(plan, prefix="正在自动应用识别格子")

    def _click_online_predictions(self) -> None:
        """把当前在线识别结果手动应用到浏览器图块。"""
        if self.online_image is None or self.online_sample is None:
            QMessageBox.information(self, "在线图片", "当前没有已加载的在线挑战图片。")
            return
        if not self.online_predictions:
            QMessageBox.information(self, "在线结果", "请先点击“识别当前在线图片”。")
            return
        if not self.online_worker.session.browser.is_automation_ready():
            QMessageBox.information(self, "在线浏览器", "Google Chrome 自动化会话尚未就绪。")
            return

        indices = sorted({int(item.index) for item in self.online_predictions})
        if not indices:
            QMessageBox.information(self, "在线结果", "当前识别结果没有命中格子。")
            return
        self._render_online(indices)
        plan = self._online_click_plan(indices)
        if self._submit_online_click_plan(plan, prefix="正在手动应用识别格子"):
            self.status.setText(
                f"正在点击网页图块：{indices}（{plan.strategy_label}）"
            )

    def _online_type_changed(self, value: str) -> None:
        """在线类型手动切换时同步网格默认值与识别参数。"""
        self.online_grid.setCurrentText(resolve_challenge_grid(value).text)
        self._parameter_preset_changed()

    def _render_online(self, selected: list[int]) -> None:
        if self.online_image is None:
            return
        rendered = draw_grid(
            self.online_image,
            parse_grid(self.online_grid.currentText()),
            selected,
            [],
            ASSETS_DIR / "image.png",
        )
        self.online_canvas.setPixmap(_preview_pixmap(rendered))

    def _recognize_online(self) -> None:
        """使用公共模型、统一策略和公共参数识别当前在线图片。"""
        if self.online_image is None or self.online_sample is None:
            QMessageBox.information(self, "在线图片", "请先捕获或导入一张在线挑战图片。")
            return
        if not self.service.loaded:
            QMessageBox.information(self, "分类模型", "请先在识别方案页加载公共分类模型。")
            return
        target = self.online_target.text().strip() or self.online_sample.target_class
        try:
            self._apply_online_profile()
            spec = parse_grid(self.online_grid.currentText())
        except ValueError as exc:
            self._set_online_status(f"在线识别失败：{exc}")
            QMessageBox.critical(self, "在线识别失败", str(exc))
            return
        preset = str(self.parameter_preset.currentData() or "balanced")
        header = (
            f"图片: {self._display_path(self.online_sample.path)}\n"
            f"原始类别: {self.online_sample.category_zh} / {self.online_sample.category_en}\n"
            f"mid: {self.online_sample.category_mid or '未知'}\n"
            f"SHA-256: {self.online_sample.sha256}\n"
            f"元数据: {self._display_path(self.online_sample.metadata_path)}\n"
            f"参数方案: {PARAMETER_PRESET_LABELS.get(preset, preset)}\n"
        )
        self._set_online_status(f"正在识别：目标={target}，网格={spec.text}")
        self._submit_recognition(
            RecognitionRequest(
                source=SOURCE_ONLINE,
                image=self.online_image,
                challenge_type=self.online_type.currentText(),
                spec=spec,
                target_class=target,
                requested_mode=str(self.recognition_mode.currentData() or "smart"),
                parameters=self._recognition_parameters(),
                image_key=self.online_image_sha256 or self.online_sample.sha256,
                header=header,
            )
        )

    def _apply_online_result(self, outcome: RecognitionOutcome) -> None:
        result = outcome.result
        self.online_all_predictions = list(result.all_predictions)
        self.online_predictions = list(result.selected_predictions)
        indices = list(result.indices)
        self._render_online(indices)
        self.online_detail.setPlainText(self._compose_report(outcome))
        self._show_actual_recognition_route(result)
        self._set_online_status(
            f"在线识别完成（{result.route.label}）：识别到格子 {indices}"
        )
        self.status.setText(f"在线识别完成（{result.route.label}）：{indices}")
        self._maybe_apply_online_clicks(indices)

    def _copy_online_result(self) -> None:
        text = self.online_detail.toPlainText().strip()
        if not text:
            self.status.setText("当前没有在线识别结果")
            return
        QApplication.clipboard().setText(text)
        self.status.setText("在线识别结果已复制到剪贴板")

    def _load_online_model(self) -> None:
        selected = self._selected_weight()
        if not selected:
            QMessageBox.information(self, "分类模型", "请在识别方案页选择公共分类权重。")
            return
        try:
            info = self.service.load(selected, DEFAULT_DEVICE)
            message = f"公共分类模型已加载：{Path(info['weights']).name}，设备={info['device']}"
            self._set_online_status(message)
            self.status.setText(message)
        except Exception as exc:
            QMessageBox.critical(self, "分类模型加载失败", str(exc))

    def _try_auto_load_online_model(self) -> None:
        if self.service.loaded:
            self._set_online_status(
                f"在线识别使用公共分类模型：{Path(str(self.service.weights)).name}"
            )
            return
        value = self._selected_weight()
        path = Path(value) if value else None
        if path is None or not path.is_file():
            self._set_online_status("当前没有可自动加载的在线模型")
            return
        try:
            info = self.service.load(path, DEFAULT_DEVICE)
            self._set_online_status(
                f"公共分类模型已自动加载：{Path(info['weights']).name}，设备={info['device']}"
            )
        except Exception as exc:
            self._set_online_status(f"在线模型自动加载失败：{exc}")

    def _apply_online_profile(self) -> None:
        """在线图片切换类别后刷新同一组公共参数。"""
        if str(self.parameter_preset.currentData() or "balanced") != "custom":
            self._parameter_preset_changed(
                target_override=self.online_target.text().strip()
            )

    def _online_click_plan(self, indices: list[int]):
        """按当前在线挑战生成 3×3 / 4×4 点击计划。"""
        return build_click_plan(
            self.online_type.currentText(),
            parse_grid(self.online_grid.currentText()),
            indices,
            self._click_settings(),
        )

    def _submit_online_click_plan(self, plan, *, prefix: str) -> bool:
        """校验保护规则后将点击计划提交给现有后台队列。"""
        if plan.blocked:
            message = f"{prefix}已暂停：{plan.reason}"
            self._set_online_status(message)
            self.status.setText(message)
            QMessageBox.warning(self, "点击保护", message)
            return False
        self._set_online_status(
            f"{prefix}：{plan.indices}；方案={plan.strategy_label}"
        )
        self.online_worker.request_apply_clicks(
            plan.indices,
            click_verify=plan.click_verify,
            watch_after_ms=plan.watch_after_ms,
            delay_ms=plan.delay_ms,
        )
        return True

    def _choose_online_data(self) -> None:
        """选择在线图片归档根目录。"""
        self._choose_data_directory(self.online_data, "选择在线图片目录", "online")
