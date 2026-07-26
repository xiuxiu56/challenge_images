"""PySide6 主 GUI：大图网格识别、人工标注和在线样本归档。"""

from __future__ import annotations

import sys
import hashlib
from io import BytesIO
from datetime import datetime
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QLabel, QLineEdit,
    QGroupBox, QMainWindow, QMessageBox, QPushButton,
    QHBoxLayout, QPlainTextEdit, QSplitter, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..config import (
    ANNOTATIONS_DIR,
    ASSETS_DIR,
    CHALLENGE_DIR,
    DEFAULT_DEVICE,
    ONLINE_CAPTURE_DIR,
    REPORTS_DIR,
    ROOT,
    RECOMMENDED_SEGMENTATION_MODEL,
    available_model_choices,
    available_segmentation_model_choices,
    model_display_name,
    resolve_default_segmentation_weight,
    resolve_default_weight,
)
from ..annotation_store import AnnotationStore
from ..grid.grid_engine import draw_grid, grid_for_challenge, grid_index_from_point, parse_grid, replace_grid_tile, resolve_challenge_grid
from ..training.model_service import ModelService, TilePrediction
from ..segmentation.model_service import SegmentationModelService
from ..segmentation.result_fusion import FUSION_MODE_LABELS
from ..recognition.engine import RecognitionResult
from ..recognition import (
    ENGINE_MODE_LABELS,
    PARAMETER_PRESET_LABELS,
    ClickSettings,
    RecognitionEngine,
    RecognitionParameters,
    build_click_plan,
    format_recognition_report,
    parameters_for,
)
from ..data.sample_manager import SampleManager, write_jsonl
from ..online import BrowserSession, OnlineCaptureService
from ..online.online_worker import (
    AUTO_REFRESH_INTERVAL_SEC,
    CHECKBOX_MONITOR_INTERVAL_SEC,
    SITE_DATA_CLEAR_INTERVAL_SEC,
    OnlineWorker,
)
from .state import (
    SOURCE_FUSION,
    SOURCE_OFFLINE,
    SOURCE_ONLINE,
    RecognitionOutcome,
    RecognitionRequest,
)
from .theme import APP_STYLESHEET, set_button_role
from .online_data_tab import OnlineDataTabMixin
from .settings_tab import SettingsTabMixin
from .widgets import NumericLineEdit, _preview_pixmap
from .workers import RecognitionWorker


class QtChallengeGUI(QMainWindow, OnlineDataTabMixin, SettingsTabMixin):
    def __init__(self, project_root: Path = ROOT) -> None:
        super().__init__()
        self.project_root = project_root
        self.gui_settings = QSettings(
            str(self.project_root / "reports" / "gui_settings.ini"),
            QSettings.IniFormat,
        )
        saved_offline = Path(
            str(self.gui_settings.value("directories/offline", str(CHALLENGE_DIR)))
        ).expanduser()
        saved_online = Path(
            str(self.gui_settings.value("directories/online", str(ONLINE_CAPTURE_DIR)))
        ).expanduser()
        self.initial_offline_data = saved_offline if saved_offline.is_dir() else CHALLENGE_DIR
        self.initial_online_data = saved_online if saved_online.is_dir() else ONLINE_CAPTURE_DIR
        self.setWindowTitle("挑战图片识别与标注工作台")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.service = ModelService()
        self.segmentation_service = SegmentationModelService()
        self.recognition_service = RecognitionEngine(
            self.service,
            self.segmentation_service,
        )
        # 识别在独立线程执行；4×4 图要跑 16 格分类加整图分割，
        # 留在 UI 线程会让窗口冻结数秒。
        self.recognition_worker = RecognitionWorker(self.recognition_service, self)
        self.recognition_worker.finished.connect(self._on_recognition_finished)
        self.recognition_worker.failed.connect(self._on_recognition_failed)
        self.recognition_worker.busy_changed.connect(self._on_recognition_busy_changed)
        self.manager: SampleManager | None = None
        self.current: dict | None = None
        self.image: Image.Image | None = None
        self.predictions: list[TilePrediction] = []
        self.all_predictions: list[TilePrediction] = []
        self.last_offline_result: RecognitionResult | None = None
        self._offline_render_indices: list[int] = []
        self.annotation_indices: set[int] = set()
        self.annotation_mode = False
        self.annotations = AnnotationStore(ANNOTATIONS_DIR / "grid_annotations.json")
        self.online_capture = OnlineCaptureService(self.initial_online_data)
        self.online_sample = None
        self.online_image: Image.Image | None = None
        self.online_image_sha256 = ""
        self.online_predictions: list[TilePrediction] = []
        self.online_all_predictions: list[TilePrediction] = []
        self.fusion_image: Image.Image | None = None
        self.fusion_image_key = ""
        self.fusion_source = ""
        self.fusion_sample: dict | None = None
        self.fusion_manager: SampleManager | None = None
        self.fusion_indices: list[int] = []
        self.fusion_preview_image: Image.Image | None = None
        self.browser_session = BrowserSession()
        self.online_worker = OnlineWorker(self)
        self.online_worker.status.connect(self._on_online_status)
        self.online_worker.failed.connect(self._on_online_failed)
        self.online_worker.started_ok.connect(self._on_online_started)
        self.online_worker.challenge_ready.connect(self._on_online_challenge)
        self.online_worker.clicks_done.connect(self._on_online_clicks_done)
        self.online_worker.stopped.connect(self._on_online_stopped)
        self.online_worker.query_restricted.connect(self._on_online_query_restricted)
        self.challenge = QComboBox()
        self.challenge.addItems(["dynamic", "imageselect", "tileselect", "multicaptcha"])
        self.grid = QComboBox()
        self.grid.addItems(["3×3", "4×4"])
        self.data_source = QComboBox()
        self.data_source.addItem("离线图片", "offline")
        self.data_source.addItem("在线图片", "online")
        saved_source = str(self.gui_settings.value("sources/recognition", "offline"))
        self.data_source.setCurrentIndex(max(0, self.data_source.findData(saved_source)))
        self.threshold = NumericLineEdit(0.25, 0, 1)
        self.top1_threshold = NumericLineEdit(0.80, 0, 1)
        self.top_k = NumericLineEdit(3, 1, 5, integer=True)
        self.multiview = QCheckBox("允许人行横道受控多视角")
        self.multiview.setChecked(True)
        self.multiview_threshold = NumericLineEdit(0.80, 0, 1)
        self.imgsz = QComboBox()
        self.imgsz.addItems(["224", "320", "640"])
        self.recognition_mode = QComboBox()
        for mode, label in ENGINE_MODE_LABELS.items():
            self.recognition_mode.addItem(label, mode)
        self.parameter_preset = QComboBox()
        for preset, label in PARAMETER_PRESET_LABELS.items():
            self.parameter_preset.addItem(label, preset)
        self.show_advanced_parameters = QCheckBox("显示高级参数")
        self.show_advanced_parameters.setChecked(False)
        self.active_recognition_label = QLabel("当前实际方案：等待图片")
        self.active_recognition_label.setProperty("role", "muted")
        self.active_recognition_label.setWordWrap(True)
        self.click_delay = NumericLineEdit(220, 0, 2000, integer=True)
        self.dynamic_wait_seconds = NumericLineEdit(8, 1, 30, integer=True)
        self.auto_verify = QCheckBox("点击后自动验证")
        self.auto_verify.setChecked(False)
        self.maximum_selected_ratio = NumericLineEdit(0.90, 0.50, 1.0, decimals=2)
        self._applying_recognition_preset = False
        self.status_filter = QComboBox()
        self.status_filter.addItems(["全部", "未处理", "成功", "失败"])
        self.deduplicate = QCheckBox("精确去重")
        self.deduplicate.setChecked(True)
        self.online_enabled = QCheckBox("在线识别验证（拿到图后自动跑模型）")
        self.online_enabled.setChecked(False)
        self.auto_click_tiles = QCheckBox("自动点击网页图块")
        self.auto_click_tiles.setChecked(False)
        self.auto_refresh_challenge = QCheckBox("自动刷新挑战（每3秒）")
        self.auto_refresh_challenge.setChecked(False)
        self.monitor_checkbox = QCheckBox("自动点击并监控复选框（关闭后5秒重试）")
        self.monitor_checkbox.setChecked(True)
        self.clear_site_data = QCheckBox("每3分钟清理站点数据（含第三方Cookie）")
        self.clear_site_data.setChecked(False)
        self.online_category = QLineEdit()
        self.online_category.setPlaceholderText("reload 缺失时填写类别，例如 Crosswalk")
        self.online_category.setMinimumWidth(180)
        self.online_type = QComboBox()
        self.online_type.addItems(["dynamic", "imageselect", "tileselect", "multicaptcha"])
        self.online_type.setMinimumWidth(130)
        self.online_grid = QComboBox()
        self.online_grid.addItems(["3×3", "4×4"])
        self.online_target = QLineEdit()
        self.online_target.setPlaceholderText("由 reload 自动填写，例如 Crosswalk")
        self.online_status_summary = QLabel("在线状态：未启动")
        self.online_status_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.online_status = QPlainTextEdit()
        self.online_status.setReadOnly(True)
        self.online_status.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.online_status.setMaximumBlockCount(300)
        self.online_status.setMinimumSize(480, 300)
        self.online_status.setPlaceholderText("浏览器、网络捕获、解析、归档和识别事件会按时间显示在这里。")
        self._set_online_status("未启动")
        self.online_canvas = QLabel("等待在线挑战图片…")
        self.online_canvas.setAlignment(Qt.AlignCenter)
        self.online_canvas.setMinimumSize(420, 250)
        # QLabel 默认会把当前文字或 QPixmap 尺寸当作 sizeHint。
        # 忽略这个动态 sizeHint，避免“等待→加载图片→标记结果”时
        # 左右分栏被图片像素尺寸重新推挤。
        self.online_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.online_canvas.setScaledContents(False)
        self.online_canvas.setStyleSheet(
            "QLabel { color: #94a3b8; background: #0f172a; border: 1px solid #334155; border-radius: 10px; }"
        )
        self.online_image_info = QLabel("尚未捕获在线图片")
        self.online_image_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.online_image_info.setWordWrap(True)
        self.online_detail = QPlainTextEdit()
        self.online_detail.setReadOnly(True)
        self.online_detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.online_detail.setFont(QFont("Menlo", 11))
        self.online_detail.setObjectName("reportEditor")
        self.online_detail.setPlaceholderText("在线模型识别结果会显示在这里。")
        self.weights = QComboBox()
        self.weights.setEditable(False)
        self.weights.setMaxVisibleItems(12)
        self.fusion_seg_weights = QComboBox()
        self.fusion_seg_weights.setEditable(False)
        self.fusion_seg_weights.setMaxVisibleItems(12)
        self.fusion_seg_weights.setMinimumWidth(240)
        self._populate_models()
        self._populate_segmentation_model_combo()
        self.offline_data = QLineEdit(str(self.initial_offline_data))
        self.offline_data.setReadOnly(True)
        self.offline_data.setCursorPosition(0)
        self.offline_data.setToolTip("离线图片根目录，下级直接包含 dynamic、imageselect 等挑战类型")
        self.online_data = QLineEdit(str(self.initial_online_data))
        self.online_data.setReadOnly(True)
        self.online_data.setCursorPosition(0)
        self.online_data.setToolTip("在线采集图片的归档根目录")
        self.target = QLineEdit("自动读取文件夹类别")
        self.canvas = QLabel("正在加载首张图片…")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(560, 300)
        # 离线数据中 3×3 常为 300×300，4×4 常为 450×450。
        # 忽略 QPixmap 原始尺寸的布局建议，避免 multicaptcha
        # 载入后把左侧画布和整个分栏推大。
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas.setScaledContents(False)
        self.canvas.setStyleSheet("QLabel { color: #94a3b8; background: #0f172a; border: 1px solid #334155; border-radius: 10px; }")
        self.canvas.setMouseTracking(True)
        # Qt 允许运行时替换事件处理器；mypy 无法表达这种模式。
        self.canvas.mousePressEvent = self._canvas_click  # type: ignore[method-assign]
        self.image_info = QLabel("图片信息待加载")
        self.image_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.image_info.setWordWrap(True)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.detail.setFont(QFont("Menlo", 12))
        self.detail.setObjectName("reportEditor")
        self.detail.setPlaceholderText("识别结果会显示在这里，可直接选择复制。")
        self.fusion_challenge = QComboBox()
        self.fusion_challenge.addItems(["dynamic", "imageselect", "tileselect", "multicaptcha"])
        self.fusion_data_source = QComboBox()
        self.fusion_data_source.addItem("离线图片", "offline")
        self.fusion_data_source.addItem("在线图片", "online")
        saved_fusion_source = str(self.gui_settings.value("sources/fusion", "offline"))
        self.fusion_data_source.setCurrentIndex(
            max(0, self.fusion_data_source.findData(saved_fusion_source))
        )
        self.fusion_grid_label = QLabel("3×3")
        self.fusion_target = QLineEdit("Car")
        self.fusion_target.setMinimumWidth(110)
        self.fusion_cls_imgsz = QComboBox()
        self.fusion_cls_imgsz.addItems(["224", "320", "640"])
        self.fusion_cls_imgsz.setCurrentText("224")
        self.fusion_seg_imgsz = QComboBox()
        self.fusion_seg_imgsz.addItems(["320", "640"])
        self.fusion_seg_imgsz.setCurrentText("640")
        self.fusion_seg_confidence = NumericLineEdit(0.25, 0, 1)
        self.fusion_min_cell_ratio = NumericLineEdit(0.002, 0, 1, decimals=4)
        self.fusion_min_mask_ratio = NumericLineEdit(0.10, 0, 1)
        self.fusion_instance_cls_threshold = NumericLineEdit(0.80, 0, 1)
        self.fusion_instance_confidence = NumericLineEdit(0.60, 0, 1)
        self.fusion_mode = QComboBox()
        for mode, label in FUSION_MODE_LABELS.items():
            self.fusion_mode.addItem(label, mode)
        self.fusion_canvas = QLabel("请先载入当前离线或在线图片")
        self.fusion_canvas.setAlignment(Qt.AlignCenter)
        self.fusion_canvas.setMinimumSize(460, 300)
        self.fusion_canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.fusion_canvas.setScaledContents(False)
        self.fusion_canvas.setStyleSheet("QLabel { color: #94a3b8; background: #0f172a; border: 1px solid #334155; border-radius: 10px; }")
        self.fusion_detail = QPlainTextEdit()
        self.fusion_detail.setReadOnly(True)
        self.fusion_detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.fusion_detail.setFont(QFont("Menlo", 11))
        self.fusion_detail.setObjectName("reportEditor")
        self.fusion_detail.setPlaceholderText("分类、分割 mask 和融合结果会显示在这里。")
        self.fusion_source_label = QLabel("尚未载入融合验证图片")
        self.fusion_source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.fusion_source_label.setWordWrap(True)
        self.status = QLabel("准备就绪")
        self.status.setProperty("role", "status")
        self.counts = QLabel("成功 0 / 失败 0 / 总计 0 / 成功率 0.00%")
        self.counts.setProperty("role", "muted")
        self._build()
        self._refresh_manager()
        self.challenge.currentTextChanged.connect(self._challenge_changed)
        self._configure_selector_popups()
        self.online_target.editingFinished.connect(self._apply_online_profile)
        self.online_type.currentTextChanged.connect(self._online_type_changed)
        self.auto_refresh_challenge.toggled.connect(self._online_auto_refresh_changed)
        self.monitor_checkbox.toggled.connect(self._online_checkbox_monitor_changed)
        self.clear_site_data.toggled.connect(self._online_site_data_clear_changed)
        self.fusion_challenge.currentTextChanged.connect(self._sync_fusion_grid)
        self.fusion_challenge.currentTextChanged.connect(self._fusion_challenge_changed)
        self.data_source.currentIndexChanged.connect(self._data_source_changed)
        self.fusion_data_source.currentIndexChanged.connect(self._fusion_data_source_changed)
        self.parameter_preset.currentIndexChanged.connect(self._parameter_preset_changed)
        self.recognition_mode.currentIndexChanged.connect(self._update_recognition_summary)
        self.show_advanced_parameters.toggled.connect(self._set_advanced_parameters_visible)
        self._connect_advanced_parameter_changes()
        self._parameter_preset_changed()
        self._next()
        self._try_auto_load_model()
        self._try_auto_load_online_model()

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(6)
        self.setStyleSheet(APP_STYLESHEET)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        outer.addWidget(self.tabs, 1)
        self._build_offline_tab()
        self._build_online_tab()
        self._build_online_data_tab()
        self._build_segmentation_tab()
        self._build_settings_tab()

        outer.addWidget(self.status)
        outer.addWidget(self.counts)
        self.status_filter.currentTextChanged.connect(lambda _value: self._refresh_and_load())
        self.deduplicate.toggled.connect(lambda _checked: self._refresh_and_load())
        self.target.editingFinished.connect(self._apply_inference_profile)

    def _build_offline_tab(self) -> None:
        """构建默认识别工作台，只保留图片、结果和高频操作。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        image_box = QGroupBox("图片与网格")
        image_box.setMinimumWidth(480)
        image_layout = QVBoxLayout(image_box)
        image_layout.addWidget(self.image_info)
        image_layout.addWidget(self.canvas, 1)
        result_box = QGroupBox("识别结果")
        result_box.setMinimumWidth(480)
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.detail, 1)
        copy_button = QPushButton("复制全部结果")
        copy_button.clicked.connect(self._copy_result)
        result_layout.addWidget(copy_button)
        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(image_box)
        body.addWidget(result_box)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        body.setSizes([660, 660])
        self.offline_body_splitter = body
        layout.addWidget(body, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        for text, handler in (
            ("加载模型", self._load_model), ("开始识别", self._recognize),
            ("标注模式", self._toggle_annotation), ("保存标注", self._save_annotation),
            ("记录成功", lambda: self._mark("success")), ("记录失败", lambda: self._mark("failed")),
            ("随机图片", self._random), ("下一张", self._next), ("扫描重复", self._scan),
        ):
            button = QPushButton(text)
            if text in {"开始识别", "记录成功"}:
                set_button_role(button, "primary")
            elif text == "记录失败":
                set_button_role(button, "danger")
            if text == "开始识别":
                self.recognize_button = button
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.insertLayout(0, buttons)
        self.tabs.addTab(tab, "识别与标注")

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

    def _build_segmentation_tab(self) -> None:
        """构建分类 Top-K、整图 mask 和格子融合工作台。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        preview_box = QGroupBox("目标 mask 与融合格子")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.fusion_source_label)
        preview_layout.addWidget(self.fusion_canvas, 1)
        result_box = QGroupBox("融合识别结果")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.fusion_detail, 1)
        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(preview_box)
        body.addWidget(result_box)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        body.setSizes([660, 660])
        body.setMinimumHeight(300)
        layout.addWidget(body, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        load_class = QPushButton("加载分类模型")
        load_class.clicked.connect(self._load_fusion_class_model)
        load_seg = QPushButton("加载分割模型")
        load_seg.clicked.connect(self._load_fusion_seg_model)
        load_offline = QPushButton("载入识别页图片")
        load_offline.clicked.connect(lambda: self._load_fusion_source("recognition"))
        load_online = QPushButton("载入采集页图片")
        load_online.clicked.connect(lambda: self._load_fusion_source("online"))
        recognize = QPushButton("按当前方案识别")
        set_button_role(recognize, "primary")
        self.fusion_recognize_button = recognize
        recognize.clicked.connect(self._recognize_fusion)
        random_image = QPushButton("随机图片")
        random_image.clicked.connect(self._fusion_random)
        next_image = QPushButton("下一张")
        next_image.clicked.connect(self._fusion_next)
        scan_duplicates = QPushButton("扫描重复")
        scan_duplicates.clicked.connect(self._scan_fusion_duplicates)
        copy_result = QPushButton("复制结果")
        copy_result.clicked.connect(self._copy_fusion_result)
        apply_online = QPushButton("应用到网页")
        apply_online.clicked.connect(self._apply_fusion_to_online)
        for button in (
            load_class,
            load_seg,
            load_offline,
            load_online,
            recognize,
            random_image,
            next_image,
            scan_duplicates,
            apply_online,
            copy_result,
        ):
            actions.addWidget(button)
        layout.insertLayout(0, actions)
        self.tabs.addTab(tab, "分割与融合")

    def _configure_selector_popups(self) -> None:
        """给窄选择器设置更宽的弹出列表，避免选项被裁成小卡片。"""
        popup_widths = {
            self.challenge: 190,
            self.grid: 130,
            self.data_source: 180,
            self.imgsz: 150,
            self.status_filter: 170,
            self.weights: 420,
            self.online_type: 190,
            self.online_grid: 130,
            self.online_stats_kind: 190,
            self.online_stats_type: 190,
            self.fusion_seg_weights: 380,
            self.fusion_challenge: 190,
            self.fusion_data_source: 180,
            self.fusion_cls_imgsz: 150,
            self.fusion_seg_imgsz: 150,
            self.fusion_mode: 230,
            self.recognition_mode: 240,
            self.parameter_preset: 210,
        }
        for combo, width in popup_widths.items():
            combo.view().setMinimumWidth(width)
            combo.view().setStyleSheet(
                "QAbstractItemView { min-width: %dpx; padding: 6px; }"
                "QAbstractItemView::item { min-height: 30px; padding: 4px 10px; }" % width
            )

    def _connect_advanced_parameter_changes(self) -> None:
        """用户手工调整任一高级参数后自动切换到自定义方案。"""
        for editor in (
            self.top1_threshold,
            self.threshold,
            self.top_k,
            self.multiview_threshold,
            self.fusion_seg_confidence,
            self.fusion_min_cell_ratio,
            self.fusion_min_mask_ratio,
            self.fusion_instance_cls_threshold,
            self.fusion_instance_confidence,
        ):
            editor.textEdited.connect(self._advanced_parameter_edited)
        for combo in (
            self.imgsz,
            self.fusion_cls_imgsz,
            self.fusion_seg_imgsz,
            self.fusion_mode,
        ):
            combo.activated.connect(self._advanced_parameter_edited)
        self.multiview.clicked.connect(self._advanced_parameter_edited)

    def _advanced_parameter_edited(self, *_args) -> None:
        """标记当前参数已脱离预设。"""
        if self._applying_recognition_preset:
            return
        index = self.parameter_preset.findData("custom")
        if index >= 0 and self.parameter_preset.currentIndex() != index:
            self.parameter_preset.setCurrentIndex(index)

    def _parameter_preset_changed(self, *_args, target_override: str | None = None) -> None:
        """把所选参数方案填入唯一一组公共输入框。"""
        preset = str(self.parameter_preset.currentData() or "balanced")
        if preset != "custom":
            target = target_override or self._current_recognition_target()
            # 已加载权重时按其训练分辨率推理，避免训练/推理尺寸错配。
            # 挑战类型决定 3×3 独立图还是 4×4 连续图，两者阈值策略相反。
            parameters = parameters_for(
                preset,
                target,
                self.service.training_imgsz,
                self._current_challenge_type(),
            )
            self._applying_recognition_preset = True
            try:
                self.imgsz.setCurrentText(str(parameters.classification_imgsz))
                self.top1_threshold.setValue(parameters.classification_top1)
                self.threshold.setValue(parameters.classification_candidate)
                self.multiview_threshold.setValue(parameters.classification_local)
                self.top_k.setValue(parameters.classification_top_k)
                self.multiview.setChecked(parameters.allow_multiview)
                self.fusion_cls_imgsz.setCurrentText(
                    str(parameters.fusion_classification_imgsz)
                )
                self.fusion_seg_imgsz.setCurrentText(str(parameters.segmentation_imgsz))
                self.fusion_seg_confidence.setValue(parameters.segmentation_confidence)
                self.fusion_min_cell_ratio.setValue(parameters.segmentation_min_cell_ratio)
                self.fusion_min_mask_ratio.setValue(parameters.segmentation_min_mask_ratio)
                self.fusion_instance_cls_threshold.setValue(
                    parameters.instance_classification_threshold
                )
                self.fusion_instance_confidence.setValue(
                    parameters.instance_confidence_threshold
                )
                mode_index = self.fusion_mode.findData(parameters.fusion_mode)
                if mode_index >= 0:
                    self.fusion_mode.setCurrentIndex(mode_index)
            finally:
                self._applying_recognition_preset = False
        self._set_advanced_parameters_visible(
            self.show_advanced_parameters.isChecked()
        )
        self._update_recognition_summary()

    def _current_recognition_target(self) -> str:
        """取得当前功能页最相关的目标类别。"""
        if getattr(self, "tabs", None) is not None:
            current_tab = self.tabs.currentIndex()
            if current_tab == 1 and self.online_target.text().strip():
                return self.online_target.text().strip()
            if current_tab == 3 and self.fusion_target.text().strip():
                return self.fusion_target.text().strip()
        target = self.target.text().strip()
        if target == "自动读取文件夹类别" and self.current:
            return str(self.current.get("target_class", ""))
        return target

    def _current_challenge_type(self) -> str:
        """取得当前功能页的挑战类型，用于选择 3×3 / 4×4 参数策略。"""
        if getattr(self, "tabs", None) is not None:
            current_tab = self.tabs.currentIndex()
            if current_tab == 1:
                return self.online_type.currentText()
            if current_tab == 3:
                return self.fusion_challenge.currentText()
        return self.challenge.currentText()

    def _set_advanced_parameters_visible(self, visible: bool) -> None:
        """折叠或展开低频识别阈值，保持小窗口不被表单压扁。"""
        if hasattr(self, "advanced_parameters_box"):
            self.advanced_parameters_box.setVisible(bool(visible))

    def _recognition_parameters(self) -> RecognitionParameters:
        """从公共设置读取一次完整且不可变的识别参数。"""
        return RecognitionParameters(
            classification_imgsz=int(self.imgsz.currentText()),
            classification_top1=float(self.top1_threshold.value()),
            classification_candidate=float(self.threshold.value()),
            classification_local=float(self.multiview_threshold.value()),
            classification_top_k=int(self.top_k.value()),
            allow_multiview=self.multiview.isChecked(),
            fusion_classification_imgsz=int(self.fusion_cls_imgsz.currentText()),
            segmentation_imgsz=int(self.fusion_seg_imgsz.currentText()),
            segmentation_confidence=float(self.fusion_seg_confidence.value()),
            segmentation_min_cell_ratio=float(self.fusion_min_cell_ratio.value()),
            segmentation_min_mask_ratio=float(self.fusion_min_mask_ratio.value()),
            instance_classification_threshold=float(
                self.fusion_instance_cls_threshold.value()
            ),
            instance_confidence_threshold=float(
                self.fusion_instance_confidence.value()
            ),
            fusion_mode=str(self.fusion_mode.currentData() or "balanced"),
        )

    def _update_recognition_summary(self, *_args) -> None:
        """显示用户当前选择；实际路由会在拿到题型后更新。"""
        mode = str(self.recognition_mode.currentData() or "smart")
        preset = str(self.parameter_preset.currentData() or "balanced")
        self.active_recognition_label.setText(
            f"当前选择：{ENGINE_MODE_LABELS.get(mode, mode)} / "
            f"{PARAMETER_PRESET_LABELS.get(preset, preset)}；"
            "实际方案会按题型、网格和分割类别覆盖情况决定。"
        )

    def _show_actual_recognition_route(self, result) -> None:
        """识别后把实际执行链路显示在公共设置页。"""
        self.active_recognition_label.setText(
            f"当前实际方案：{result.route.label}；{result.route.reason}；"
            f"网格={result.spec.text}，目标={result.target_class}"
        )

    def _click_settings(self) -> ClickSettings:
        """读取公共网页点击参数。"""
        return ClickSettings(
            delay_ms=int(self.click_delay.value()),
            dynamic_wait_ms=int(self.dynamic_wait_seconds.value()) * 1000,
            auto_verify=self.auto_verify.isChecked(),
            maximum_selected_ratio=float(self.maximum_selected_ratio.value()),
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
        spec = resolve_challenge_grid(challenge.challenge_type, detected)
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

    def closeEvent(self, event) -> None:  # noqa: N802
        for worker in (
            getattr(self, "online_worker", None),
            getattr(self, "recognition_worker", None),
        ):
            if worker is None:
                continue
            try:
                worker.shutdown()
            except Exception:
                pass
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """窗口尺寸变化后按新画布重绘离线图片。"""
        super().resizeEvent(event)
        if getattr(self, "image", None) is not None and hasattr(self, "canvas"):
            self._render(list(self._offline_render_indices))

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

    def _challenge_changed(self, value: str) -> None:
        self.grid.setCurrentText(grid_for_challenge(value).text)
        self._refresh_manager()
        # 3×3 与 4×4 的阈值策略不同，切换题型后重算参数。
        self._parameter_preset_changed()

    def _online_type_changed(self, value: str) -> None:
        """在线类型手动切换时同步网格默认值与识别参数。"""
        self.online_grid.setCurrentText(resolve_challenge_grid(value).text)
        self._parameter_preset_changed()

    @staticmethod
    def _selected_source(combo: QComboBox) -> str:
        """返回数据来源选择器的稳定标识。"""
        return str(combo.currentData() or "offline")

    def _active_data_root(self) -> Path:
        """返回识别与标注当前选择的根目录。"""
        field = self.online_data if self._selected_source(self.data_source) == "online" else self.offline_data
        return Path(field.text()).expanduser()

    def _active_fusion_root(self) -> Path:
        """返回分割与融合当前选择的根目录。"""
        field = (
            self.online_data
            if self._selected_source(self.fusion_data_source) == "online"
            else self.offline_data
        )
        return Path(field.text()).expanduser()

    def _data_source_changed(self, _index: int) -> None:
        """切换识别与标注的数据源并立即载入首张。"""
        source = self._selected_source(self.data_source)
        self.gui_settings.setValue("sources/recognition", source)
        self.gui_settings.sync()
        source_name = "在线图片" if source == "online" else "离线图片"
        self.data_source_hint.setText(f"当前使用{source_name}目录：{self._display_path(self._active_data_root())}")
        self._refresh_and_load()

    def _fusion_data_source_changed(self, _index: int) -> None:
        """切换融合样本来源，重置随机/下一张的样本游标。"""
        self.fusion_manager = None
        source = self._selected_source(self.fusion_data_source)
        self.gui_settings.setValue("sources/fusion", source)
        self.gui_settings.sync()
        source_name = "在线图片" if source == "online" else "离线图片"
        self.fusion_data_source_hint.setText(
            f"随机图片和下一张当前使用{source_name}目录："
            f"{self._display_path(self._active_fusion_root())}"
        )
        self.status.setText(f"分割与融合数据源已切换为：{source_name}")

    def _refresh_manager(self) -> None:
        root = self._active_data_root()
        self.manager = SampleManager(
            root,
            self.challenge.currentText(),
            self.deduplicate.isChecked(),
            status_filter=self.status_filter.currentText(),
        )
        source_name = "在线" if self._selected_source(self.data_source) == "online" else "离线"
        self.status.setText(
            f"{source_name} / {self.challenge.currentText()}：去重后 {len(self.manager)} 张"
        )

    def _refresh_and_load(self) -> None:
        self._refresh_manager()
        self._next()

    def _populate_models(self) -> None:
        self._populate_model_combo(self.weights)

    def _populate_segmentation_model_combo(self) -> None:
        """填充分割权重；本地为空时保留推荐模型名称作为待加载项。"""
        self.fusion_seg_weights.clear()
        paths = available_segmentation_model_choices()
        for value in paths:
            path = Path(value)
            experiment = path.parent.name if path.name == "best.pt" else path.stem
            self.fusion_seg_weights.addItem(
                f"{experiment}｜{path.name}",
                str(path),
            )
        default = resolve_default_segmentation_weight()
        if default is not None:
            index = next(
                (
                    item
                    for item in range(self.fusion_seg_weights.count())
                    if self.fusion_seg_weights.itemData(item) == str(default)
                ),
                -1,
            )
            if index >= 0:
                self.fusion_seg_weights.setCurrentIndex(index)
        if not self.fusion_seg_weights.count():
            self.fusion_seg_weights.addItem(
                f"{RECOMMENDED_SEGMENTATION_MODEL}｜预训练分割模型",
                RECOMMENDED_SEGMENTATION_MODEL,
            )

    def _populate_model_combo(self, combo: QComboBox) -> None:
        """把项目本地模型填充到指定下拉框。"""
        paths = available_model_choices()
        combo.clear()
        for path in paths:
            model_path = Path(path)
            combo.addItem(model_display_name(model_path), str(model_path))
        default = resolve_default_weight()
        if default is not None and default.is_file():
            index = next(
                (i for i in range(combo.count()) if combo.itemData(i) == str(default)),
                -1,
            )
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.addItem(model_display_name(default), str(default))
                combo.setCurrentIndex(combo.count() - 1)
        elif combo.count():
            combo.setCurrentIndex(0)
        else:
            combo.setEditText("")

    def _selected_weight(self) -> str:
        """取得下拉项保存的完整路径；手工输入时返回输入内容。"""
        return self._selected_combo_weight(self.weights)

    @staticmethod
    def _selected_combo_weight(combo: QComboBox) -> str:
        index = combo.currentIndex()
        if index >= 0:
            stored = combo.itemData(index)
            if stored:
                return str(stored)
        return combo.currentText().strip()

    def _choose_weights(self) -> None:
        self._choose_model_for_combo(self.weights, "选择离线模型权重")

    def _choose_fusion_seg_weights(self) -> None:
        self._choose_model_for_combo(self.fusion_seg_weights, "选择 YOLO 分割模型权重")

    def _choose_model_for_combo(self, combo: QComboBox, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, str(ROOT), "权重 (*.pt);;所有文件 (*)")
        if path:
            existing = next(
                (i for i in range(combo.count()) if combo.itemData(i) == path),
                -1,
            )
            if existing < 0:
                combo.addItem(Path(path).name, path)
                existing = combo.count() - 1
            combo.setCurrentIndex(existing)

    def _choose_offline_data(self) -> None:
        """选择离线挑战图片根目录。"""
        self._choose_data_directory(self.offline_data, "选择离线图片目录", "offline")

    def _choose_online_data(self) -> None:
        """选择在线图片归档根目录。"""
        self._choose_data_directory(self.online_data, "选择在线图片目录", "online")

    def _choose_data_directory(
        self,
        field: QLineEdit,
        title: str,
        source: str,
    ) -> None:
        """通过文件夹选择器更新公共图片目录。"""
        current = Path(field.text()).expanduser()
        start = current if current.is_dir() else ROOT
        path = QFileDialog.getExistingDirectory(self, title, str(start))
        if not path:
            return
        field.setText(path)
        field.setCursorPosition(0)
        self.gui_settings.setValue(f"directories/{source}", path)
        self.gui_settings.sync()
        self.manager = None
        self.fusion_manager = None
        if source == "online":
            self.online_capture = OnlineCaptureService(path)
            self.online_stats_message.setText("在线图片目录已更新，点击“刷新统计”重新扫描。")
        if self._selected_source(self.data_source) == source:
            self._data_source_changed(self.data_source.currentIndex())
        if self._selected_source(self.fusion_data_source) == source:
            self._fusion_data_source_changed(self.fusion_data_source.currentIndex())
        self.status.setText(f"{title}已更新：{self._display_path(path)}")

    def _load_model(self) -> None:
        selected = self._selected_weight()
        if not selected:
            QMessageBox.information(self, "模型", "当前没有本地训练模型，请点击“选择权重”加载 .pt 文件。")
            return
        try:
            info = self.service.load(selected, DEFAULT_DEVICE)
            self.status.setText(f"模型已加载：{Path(info['weights']).name}，设备={info['device']}")
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))

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

    def _load_fusion_class_model(self) -> None:
        selected = self._selected_weight()
        if not selected:
            QMessageBox.information(self, "分类模型", "请在识别方案页选择公共分类权重。")
            return
        try:
            info = self.service.load(selected, DEFAULT_DEVICE)
            self.status.setText(
                f"公共分类模型已加载：{Path(info['weights']).name}，设备={info['device']}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "分类模型加载失败", str(exc))

    def _load_fusion_seg_model(self) -> None:
        selected = self._selected_combo_weight(self.fusion_seg_weights)
        if not selected:
            QMessageBox.information(self, "分割模型", "请选择 YOLO segmentation 权重。")
            return
        try:
            info = self.segmentation_service.load(selected, DEFAULT_DEVICE)
            self.status.setText(
                f"分割模型已加载：{Path(info['weights']).name}，设备={info['device']}，"
                f"类别数={len(info['classes'])}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "分割模型加载失败", str(exc))

    def _sync_fusion_grid(self, challenge_type: str) -> None:
        """仅根据 /reload 挑战类型同步3×3/4×4，不读取网页 CSS 类。"""
        self.fusion_grid_label.setText(grid_for_challenge(challenge_type).text)

    def _fusion_challenge_changed(self, _challenge_type: str) -> None:
        """融合页切换挑战类型后重置独立样本游标并重算识别参数。"""
        self.fusion_manager = None
        self._parameter_preset_changed()
        if self.fusion_image is not None:
            self.fusion_indices = []
            self.fusion_preview_image = None
            self._render_fusion_preview()

    def _load_fusion_source(self, source: str) -> None:
        """把识别页或在线采集页的当前图片复制到融合标签。"""
        if source == "online":
            if self.online_image is None or self.online_sample is None:
                QMessageBox.information(self, "在线图片", "当前没有已载入的在线完整挑战图片。")
                return
            self.fusion_image = self.online_image.copy()
            self.fusion_image_key = self.online_image_sha256 or self.online_sample.sha256
            self.fusion_source = f"在线：{self._display_path(self.online_sample.path)}"
            self.fusion_sample = None
            self.fusion_challenge.setCurrentText(self.online_type.currentText())
            self.fusion_target.setText(
                self.online_target.text().strip() or self.online_sample.target_class
            )
        else:
            if self.image is None or self.current is None:
                QMessageBox.information(self, "识别页图片", "识别与标注页当前没有已载入的挑战图片。")
                return
            self.fusion_image = self.image.copy()
            self.fusion_image_key = str(self.current.get("sha256", ""))
            source_name = "在线目录" if self._selected_source(self.data_source) == "online" else "离线目录"
            self.fusion_source = f"识别页（{source_name}）：{self._display_path(self.current['path'])}"
            self.fusion_sample = self.current
            self.fusion_challenge.setCurrentText(self.challenge.currentText())
            self.fusion_target.setText(str(self.current.get("target_class", "")))
        self._finish_loading_fusion_source()

    def _finish_loading_fusion_source(self) -> None:
        """统一初始化融合图片、真值标注和结果区。"""
        if self.fusion_image is None:
            return
        spec = grid_for_challenge(self.fusion_challenge.currentText())
        self.fusion_grid_label.setText(spec.text)
        self.fusion_indices = []
        self.fusion_preview_image = None
        self._render_fusion_preview()
        self.fusion_source_label.setText(
            f"来源：{self.fusion_source}  |  挑战类型：{self.fusion_challenge.currentText()}  |  "
            f"接口网格：{spec.text}  |  目标：{self.fusion_target.text()}"
        )
        self.fusion_detail.setPlainText(
            f"来源: {self.fusion_source}\n"
            f"挑战类型: {self.fusion_challenge.currentText()}\n"
            f"接口确定网格: {spec.text}\n"
            f"目标类别: {self.fusion_target.text()}\n\n"
            "等待分类 + 分割融合识别。"
        )
        self.status.setText(f"融合验证图片已载入：{self.fusion_source}")

    def _ensure_fusion_manager(self) -> SampleManager:
        """创建或复用融合页自己的去重样本管理器。"""
        root = self._active_fusion_root()
        challenge_type = self.fusion_challenge.currentText()
        if (
            self.fusion_manager is None
            or self.fusion_manager.root != root
            or self.fusion_manager.challenge_type != challenge_type
            or self.fusion_manager.deduplicate != self.deduplicate.isChecked()
        ):
            self.fusion_manager = SampleManager(
                root,
                challenge_type,
                self.deduplicate.isChecked(),
            )
        return self.fusion_manager

    def _load_fusion_sample(self, sample: dict | None) -> None:
        """直接在融合页载入样本，不需切换到离线标签。"""
        if not sample:
            QMessageBox.information(self, "融合样本", "当前挑战类型没有可用图片。")
            return
        try:
            self.fusion_image = Image.open(sample["path"]).convert("RGB")
        except Exception as exc:
            QMessageBox.warning(self, "图片加载失败", str(exc))
            return
        self.fusion_sample = sample
        self.fusion_image_key = str(sample.get("sha256", ""))
        source_name = "在线" if self._selected_source(self.fusion_data_source) == "online" else "离线"
        self.fusion_source = f"{source_name}：{self._display_path(sample['path'])}"
        self.fusion_target.setText(str(sample.get("target_class", "")))
        self._finish_loading_fusion_source()

    def _fusion_next(self) -> None:
        manager = self._ensure_fusion_manager()
        self._load_fusion_sample(manager.next_sample())

    def _fusion_random(self) -> None:
        manager = self._ensure_fusion_manager()
        self._load_fusion_sample(manager.random_sample())

    def _render_fusion_preview(self) -> None:
        """按融合结果或人工真值稳定重绘预览。"""
        if self.fusion_image is None:
            return
        spec = grid_for_challenge(self.fusion_challenge.currentText())
        base = self.fusion_preview_image or self.fusion_image
        rendered = draw_grid(
            base,
            spec,
            self.fusion_indices,
            icon_path=ASSETS_DIR / "image.png",
        )
        self.fusion_canvas.setPixmap(_preview_pixmap(rendered))

    def _recognize_fusion(self) -> None:
        """在诊断页执行公共识别方案，并展示分类或融合明细。"""
        if self.fusion_image is None:
            QMessageBox.information(self, "融合图片", "请先载入当前离线或在线图片。")
            return
        if not self.service.loaded:
            QMessageBox.information(self, "分类模型", "请先在识别方案页加载公共分类模型。")
            return
        target = self.fusion_target.text().strip()
        if not target:
            QMessageBox.information(self, "目标类别", "请填写 /reload 解析得到的目标类别。")
            return
        spec = grid_for_challenge(self.fusion_challenge.currentText())
        self._parameter_preset_changed(target_override=target)
        preset = str(self.parameter_preset.currentData() or "balanced")
        segmentation_name = (
            Path(str(self.segmentation_service.weights)).name
            if self.segmentation_service.loaded
            else "未加载"
        )
        header = (
            f"来源: {self.fusion_source}\n"
            f"参数方案: {PARAMETER_PRESET_LABELS.get(preset, preset)}\n"
            f"分类模型: {Path(str(self.service.weights)).name}\n"
            f"分割模型: {segmentation_name}\n"
        )
        # 识别改由工作线程执行，不再需要 WaitCursor + processEvents 硬扛。
        self._submit_recognition(
            RecognitionRequest(
                source=SOURCE_FUSION,
                image=self.fusion_image,
                challenge_type=self.fusion_challenge.currentText(),
                spec=spec,
                target_class=target,
                requested_mode=str(self.recognition_mode.currentData() or "smart"),
                parameters=self._recognition_parameters(),
                image_key=self.fusion_image_key or None,
                header=header,
            )
        )

    def _apply_fusion_result(self, outcome: RecognitionOutcome) -> None:
        result = outcome.result
        self.fusion_indices = list(result.indices)
        self.fusion_preview_image = result.preview
        self._render_fusion_preview()
        self.fusion_detail.setPlainText(self._compose_report(outcome))
        self._show_actual_recognition_route(result)
        self.status.setText(f"识别完成（{result.route.label}）：{result.indices}")

    def _scan_fusion_duplicates(self) -> None:
        """按融合页当前挑战类型扫描精确重复图片。"""
        from ..data.sample_manager import scan_duplicates

        groups = scan_duplicates(
            self._active_fusion_root(),
            self.fusion_challenge.currentText(),
        )
        extra = sum(len(paths) - 1 for paths in groups.values())
        QMessageBox.information(
            self,
            "融合图片重复扫描",
            f"挑战类型：{self.fusion_challenge.currentText()}\n"
            f"重复组：{len(groups)}\n多余副本：{extra}\n"
            "GUI 只跳过重复图片，不直接删除文件。",
        )

    def _copy_fusion_result(self) -> None:
        text = self.fusion_detail.toPlainText().strip()
        if not text:
            self.status.setText("当前没有可复制的融合结果")
            return
        QApplication.clipboard().setText(text)
        self.status.setText("融合结果已复制到剪贴板")

    def _apply_fusion_to_online(self) -> None:
        """将在线图片的融合结果交给现有 Playwright 点击队列。"""
        if not self.fusion_source.startswith("在线："):
            QMessageBox.information(self, "融合来源", "请先载入当前在线图片并完成融合识别。")
            return
        if not self.fusion_indices:
            QMessageBox.information(self, "融合结果", "当前融合结果没有命中格子。")
            return
        if not self.online_worker.session.browser.is_automation_ready():
            QMessageBox.information(self, "在线浏览器", "Google Chrome 自动化会话尚未就绪。")
            return
        plan = self._online_click_plan(self.fusion_indices)
        if self._submit_online_click_plan(plan, prefix="正在应用方案格子到网页"):
            self.status.setText(
                f"正在点击方案格子：{self.fusion_indices}（{plan.strategy_label}）"
            )

    def _try_auto_load_model(self) -> None:
        value = self._selected_weight()
        if not value:
            self.status.setText("首张图片已加载；当前没有本地训练模型")
            return
        path = Path(value)
        if not path.is_file():
            self.status.setText("首张图片已加载；请选择模型权重后点击“加载模型”")
            return
        try:
            info = self.service.load(path, DEFAULT_DEVICE)
            self.status.setText(f"首张图片已加载，模型已加载：{Path(info['weights']).name}，设备={info['device']}")
        except Exception as exc:
            self.status.setText(f"首张图片已加载；模型待加载：{exc}")

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

    def _load(self, sample: dict | None) -> None:
        if not sample:
            QMessageBox.information(self, "样本", "当前目录没有样本")
            return
        self.current = sample
        self.image = Image.open(sample["path"]).convert("RGB")
        self.predictions = []
        self.all_predictions = []
        self.last_offline_result = None
        self.annotation_indices = set((self.annotations.get(sample["path"]) or {}).get("真实格子", []))
        self.target.setText(str(sample["target_class"]))
        self._apply_inference_profile()
        self._render(sorted(self.annotation_indices))
        self.image_info.setText(
            f"图片：{self._display_path(sample['path'])}\n"
            f"目标类别：{sample['raw_class']}（{sample['target_class']}）  |  "
            f"SHA-256：{sample['sha256'][:16]}…"
        )
        self.detail.setPlainText(
            f"图片: {self._display_path(sample['path'])}\n"
            f"目标类别: {sample['raw_class']}（{sample['target_class']}）\n"
            f"SHA-256: {sample['sha256']}\n\n"
            "识别结果待生成。"
        )

    def _display_path(self, path: str | Path) -> str:
        """界面使用项目相对路径，内部仍保留绝对路径。"""
        source = Path(path)
        try:
            return str(source.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return source.name

    def _apply_inference_profile(self) -> None:
        """按当前类别刷新非自定义公共参数预设。"""
        if str(self.parameter_preset.currentData() or "balanced") != "custom":
            self._parameter_preset_changed()

    def _apply_online_profile(self) -> None:
        """在线图片切换类别后刷新同一组公共参数。"""
        if str(self.parameter_preset.currentData() or "balanced") != "custom":
            self._parameter_preset_changed(
                target_override=self.online_target.text().strip()
            )

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

    def _next(self) -> None:
        if self.manager is None:
            self._refresh_manager()
        self._load(self.manager.next_sample() if self.manager else None)

    def _random(self) -> None:
        if self.manager is None:
            self._refresh_manager()
        self._load(self.manager.random_sample() if self.manager else None)

    def _render(self, selected: list[int]) -> None:
        if self.image is None:
            return
        effective_selected = selected or sorted(self.annotation_indices)
        self._offline_render_indices = list(effective_selected)
        rendered = draw_grid(self.image, parse_grid(self.grid.currentText()), effective_selected, [], ASSETS_DIR / "image.png")
        # 与在线采集使用完全相同的预览规格，不受原图、
        # 画布、窗口尺寸以及初始化/随机/下一张操作影响。
        self.canvas.setPixmap(_preview_pixmap(rendered))

    # ------------------------------------------------------------------
    # 识别：三个功能页共用同一条异步链路
    # ------------------------------------------------------------------

    def _submit_recognition(self, request: RecognitionRequest) -> None:
        """把识别请求交给工作线程，UI 立即返回。"""
        self.status.setText(
            f"正在识别（{request.source_label}）：目标={request.target_class}，"
            f"网格={request.spec.text}，"
            f"引擎={ENGINE_MODE_LABELS.get(request.requested_mode, request.requested_mode)}"
        )
        self.recognition_worker.submit(request)

    def _on_recognition_busy_changed(self, busy: bool) -> None:
        """识别期间禁用三个入口，避免重复提交。"""
        for button in (
            getattr(self, "recognize_button", None),
            getattr(self, "online_recognize_button", None),
            getattr(self, "fusion_recognize_button", None),
        ):
            if button is not None:
                button.setEnabled(not busy)

    def _on_recognition_failed(self, source: str, message: str) -> None:
        if source == SOURCE_ONLINE:
            self._set_online_status(f"在线识别失败：{message}")
        self.status.setText(f"识别失败：{message}")
        QMessageBox.critical(self, "识别失败", message)

    def _on_recognition_finished(self, outcome: RecognitionOutcome) -> None:
        """按来源把结果分发回对应页面。"""
        handlers = {
            SOURCE_OFFLINE: self._apply_offline_result,
            SOURCE_ONLINE: self._apply_online_result,
            SOURCE_FUSION: self._apply_fusion_result,
        }
        handler = handlers.get(outcome.source)
        if handler is None:
            return
        try:
            handler(outcome)
        except Exception as exc:  # 渲染失败不应让工作线程停摆
            QMessageBox.critical(self, "结果显示失败", str(exc))

    @staticmethod
    def _compose_report(outcome: RecognitionOutcome) -> str:
        """公共报告体：头部由各页面提供，主体三页一致。"""
        return f"{outcome.request.header}\n{format_recognition_report(outcome.result)}"

    def _recognize(self) -> None:
        if self.image is None:
            self._next()
        if self.image is None or self.current is None:
            return
        if not self.service.loaded:
            QMessageBox.information(self, "模型", "请先加载模型")
            return
        target = self.target.text()
        if target == "自动读取文件夹类别" and self.current:
            target = self.current["target_class"]
        try:
            self._parameter_preset_changed(target_override=target)
            spec = parse_grid(self.grid.currentText())
        except ValueError as exc:
            QMessageBox.critical(self, "识别失败", str(exc))
            return
        preset = str(self.parameter_preset.currentData() or "balanced")
        header = (
            f"图片: {self._display_path(self.current['path'])}\n"
            f"原始类别: {self.current['raw_class']}（{self.current['target_class']}）\n"
            f"SHA-256: {self.current['sha256']}\n"
            f"参数方案: {PARAMETER_PRESET_LABELS.get(preset, preset)}\n"
        )
        self._submit_recognition(
            RecognitionRequest(
                source=SOURCE_OFFLINE,
                image=self.image,
                challenge_type=self.challenge.currentText(),
                spec=spec,
                target_class=target,
                requested_mode=str(self.recognition_mode.currentData() or "smart"),
                parameters=self._recognition_parameters(),
                image_key=self.current["sha256"],
                header=header,
            )
        )

    def _apply_offline_result(self, outcome: RecognitionOutcome) -> None:
        result = outcome.result
        self.all_predictions = list(result.all_predictions)
        self.predictions = list(result.selected_predictions)
        self.last_offline_result = result
        indices = list(result.indices)
        self._render(indices)
        truth = sorted(self.annotation_indices)
        metrics = (
            f"真实格子: {truth}\n"
            f"完全匹配: {bool(truth and truth == indices)}\n"
            f"漏选格子: {sorted(set(truth) - set(indices))}\n"
            f"误选格子: {sorted(set(indices) - set(truth))}\n"
            if truth
            else ""
        )
        self.detail.setPlainText(
            f"{outcome.request.header}{metrics}\n{format_recognition_report(result)}"
        )
        self._show_actual_recognition_route(result)
        self.status.setText(f"识别完成（{result.route.label}）：{indices}")

    def _copy_result(self) -> None:
        """把右侧完整纯文本结果复制到系统剪贴板。"""
        text = self.detail.toPlainText().strip()
        if not text:
            self.status.setText("当前没有可复制的识别结果")
            return
        QApplication.clipboard().setText(text)
        self.status.setText("识别结果已复制到剪贴板")

    def _mark(self, status: str) -> None:
        if not self.current:
            QMessageBox.information(self, "样本", "请先加载图片")
            return
        existing = self._read_results()
        if any(item.get("sha256") == self.current["sha256"] and item.get("status") == status for item in existing):
            self.status.setText("该样本已经记录过相同状态")
            return
        truth = sorted(self.annotation_indices)
        predicted = sorted(x.index for x in self.predictions)
        preset = str(self.parameter_preset.currentData() or "balanced")
        requested_mode = str(self.recognition_mode.currentData() or "smart")
        actual_mode = (
            self.last_offline_result.route.actual_mode
            if self.last_offline_result is not None
            else "未识别"
        )
        parameters = self._recognition_parameters()
        write_jsonl(
            REPORTS_DIR / "gui_results.jsonl",
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "challenge_type": self.challenge.currentText(),
                "grid": self.grid.currentText(),
                "path": str(self.current["path"]),
                "sha256": self.current["sha256"],
                "target_class": self.current["target_class"],
                "recognition_engine": requested_mode,
                "actual_engine": actual_mode,
                "parameter_preset": preset,
                "policy_name": PARAMETER_PRESET_LABELS.get(preset, preset),
                "threshold": parameters.classification_candidate,
                "top1_threshold": parameters.classification_top1,
                "multiview_threshold": parameters.classification_local,
                "top_k": parameters.classification_top_k,
                "multiview": parameters.allow_multiview,
                "imgsz": parameters.classification_imgsz,
                "fusion_classification_imgsz": parameters.fusion_classification_imgsz,
                "segmentation_imgsz": parameters.segmentation_imgsz,
                "fusion_mode": parameters.fusion_mode,
                "predicted_indices": predicted,
                "selected_predictions": [x.__dict__ for x in self.predictions],
                "all_predictions": [x.__dict__ for x in self.all_predictions],
                "真实格子": truth,
                "整图完全匹配": bool(truth and truth == predicted),
                "漏选格子": sorted(set(truth) - set(predicted)),
                "误选格子": sorted(set(predicted) - set(truth)),
            },
        )
        self._update_counts()
        self.status.setText("已记录：" + ("成功" if status == "success" else "失败"))

    def _update_counts(self) -> None:
        import json
        path = REPORTS_DIR / "gui_results.jsonl"
        ok = fail = 0
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    ok += item.get("status") == "success"
                    fail += item.get("status") == "failed"
                except json.JSONDecodeError:
                    pass
        total = ok + fail
        self.counts.setText(f"成功 {ok} / 失败 {fail} / 总计 {total} / 成功率 {(ok / total * 100 if total else 0):.2f}%")

    def _read_results(self) -> list[dict]:
        import json
        path = REPORTS_DIR / "gui_results.jsonl"
        if not path.is_file():
            return []
        rows=[]
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def _toggle_annotation(self) -> None:
        self.annotation_mode = not self.annotation_mode
        self.status.setText("标注模式已开启：点击格子切换真实目标" if self.annotation_mode else "标注模式已关闭")

    def _canvas_click(self, event) -> None:
        if not self.annotation_mode or self.image is None:
            return
        spec = parse_grid(self.grid.currentText())
        pixmap = self.canvas.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        # QLabel 使用居中显示，图片缩放后四周可能有留白；先扣除留白，
        # 再按实际 QPixmap 尺寸换算，避免点击位置与网格编号错位。
        image_x = event.position().x() - (self.canvas.width() - pixmap.width()) / 2
        image_y = event.position().y() - (self.canvas.height() - pixmap.height()) / 2
        index = grid_index_from_point(image_x, image_y, pixmap.width(), pixmap.height(), spec)
        if index is None:
            return
        if index in self.annotation_indices:
            self.annotation_indices.remove(index)
        else:
            self.annotation_indices.add(index)
        self._render(sorted(self.annotation_indices))

    def _save_annotation(self) -> None:
        if not self.current:
            return
        self.annotations.set(self.current["path"], challenge_type=self.challenge.currentText(), grid=self.grid.currentText(), target_class=self.current["target_class"], indices=list(self.annotation_indices))
        self.status.setText(f"已保存真实格子：{sorted(self.annotation_indices)}")

    def _scan(self) -> None:
        from ..data.sample_manager import scan_duplicates
        groups = scan_duplicates(self._active_data_root(), self.challenge.currentText())
        count = sum(len(x) - 1 for x in groups.values())
        QMessageBox.information(self, "重复扫描", f"重复组：{len(groups)}，重复文件：{count}\nGUI 默认跳过重复，不直接删除。")


def launch_qt_gui(project_root: str | Path = ROOT) -> None:
    # instance() 的返回类型是 QCoreApplication；这里始终是 QApplication。
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setFont(QFont("PingFang SC", 13))
    window = QtChallengeGUI(Path(project_root))
    window.show()
    app.exec()
