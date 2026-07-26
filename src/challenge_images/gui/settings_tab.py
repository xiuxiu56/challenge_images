"""设置配置页的控件搭建。

从 ``qt_gui.py`` 抽出：这一组方法只做界面装配，不含识别逻辑，
占据了主窗口约 350 行。
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import model_display_name
from ..recognition import ENGINE_MODE_LABELS, PARAMETER_PRESET_LABELS
from ..segmentation.result_fusion import FUSION_MODE_LABELS
from .widgets import NumericLineEdit


class SettingsTabMixin:
    """设置配置页：把低频参数集中到一处，避免铺满主界面。

    只负责搭建控件；参数的读取与应用仍在主窗口，
    因为它们被三个功能页共用。
    """

    def _build_settings_tab(self) -> None:
        """构建公共设置入口，集中管理四个功能页的参数。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        header = QWidget()
        header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 8, 18, 8)
        header_layout.setSpacing(16)
        title = QLabel("公共设置配置")
        title.setProperty("role", "pageTitle")
        description = QLabel(
            "挑战类型、网格、目标类别、模型权重、数据目录和识别参数在此统一管理；"
            "修改后立即对应用于相关功能页。"
        )
        description.setProperty("role", "muted")
        description.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(description, 1)
        layout.addWidget(header)

        settings_tabs = QTabWidget()
        settings_tabs.setObjectName("settingsTabs")
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_recognition_strategy_settings()),
            "识别方案",
        )
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_data_source_settings()), "公共数据源"
        )
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_offline_settings()), "识别与标注"
        )
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_online_settings()), "在线采集"
        )
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_online_data_settings()), "在线图片数据"
        )
        settings_tabs.addTab(
            self._wrap_settings_page(self._build_fusion_settings()), "分割与融合"
        )
        layout.addWidget(settings_tabs, 1)
        self.settings_tabs = settings_tabs
        self.tabs.addTab(tab, "设置配置")

    @staticmethod
    def _wrap_settings_page(page: QWidget) -> QScrollArea:
        """用垂直滚动区承载设置分页，避免小窗口将表单行压缩重叠。"""
        page.setMinimumHeight(page.sizeHint().height())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(0)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_data_source_settings(self) -> QWidget:
        """将两个公共根目录放进独立分页，避免长期挤压识别参数。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        directory_box = QGroupBox("公共图片目录")
        directories = QGridLayout(directory_box)
        directories.setHorizontalSpacing(10)
        directories.setVerticalSpacing(10)
        directories.addWidget(QLabel("离线图片位置"), 0, 0)
        directories.addWidget(self.offline_data, 0, 1)
        choose_offline = QPushButton("选择离线目录")
        choose_offline.clicked.connect(self._choose_offline_data)
        directories.addWidget(choose_offline, 0, 2)
        directories.addWidget(QLabel("在线图片位置"), 1, 0)
        directories.addWidget(self.online_data, 1, 1)
        choose_online = QPushButton("选择在线目录")
        choose_online.clicked.connect(self._choose_online_data)
        directories.addWidget(choose_online, 1, 2)
        directories.setColumnStretch(1, 1)
        layout.addWidget(directory_box)
        note = QLabel(
            "识别与标注、分割与融合可以各自选择离线或在线公共数据源；"
            "目录只在此维护一次。"
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _build_recognition_strategy_settings(self) -> QWidget:
        """构建三个识别入口共享的引擎、模型、参数和点击方案。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scheme_box = QGroupBox("识别方案")
        scheme = QGridLayout(scheme_box)
        scheme.setHorizontalSpacing(10)
        scheme.setVerticalSpacing(8)
        scheme.addWidget(QLabel("识别引擎"), 0, 0)
        scheme.addWidget(self.recognition_mode, 0, 1)
        scheme.addWidget(QLabel("参数方案"), 0, 2)
        scheme.addWidget(self.parameter_preset, 0, 3)
        scheme.addWidget(self.show_advanced_parameters, 0, 4)
        scheme.addWidget(self.active_recognition_label, 1, 0, 1, 5)
        note = QLabel(
            "智能推荐会让 3×3 独立图和动态题走逐格分类；"
            "4×4 连续照片在分割模型覆盖目标类别时走分类 + 整图分割。"
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        scheme.addWidget(note, 2, 0, 1, 5)
        scheme.setColumnStretch(1, 1)
        scheme.setColumnStretch(3, 1)
        layout.addWidget(scheme_box)

        model_box = QGroupBox("公共模型")
        model = QGridLayout(model_box)
        model.setHorizontalSpacing(8)
        model.setVerticalSpacing(6)
        model.addWidget(QLabel("分类模型"), 0, 0)
        model.addWidget(self.weights, 0, 1)
        choose_class = QPushButton("选择分类权重")
        choose_class.clicked.connect(self._choose_weights)
        load_class = QPushButton("加载分类模型")
        load_class.clicked.connect(self._load_model)
        model.addWidget(choose_class, 0, 2)
        model.addWidget(load_class, 0, 3)
        model.addWidget(QLabel("分割模型"), 1, 0)
        model.addWidget(self.fusion_seg_weights, 1, 1)
        choose_seg = QPushButton("选择分割权重")
        choose_seg.clicked.connect(self._choose_fusion_seg_weights)
        load_seg = QPushButton("加载分割模型")
        load_seg.clicked.connect(self._load_fusion_seg_model)
        model.addWidget(choose_seg, 1, 2)
        model.addWidget(load_seg, 1, 3)
        model_note = QLabel(
            "默认分类权重使用 recaptcha_v2_m2_320：已通过当前 4×4 摩托车和红绿灯困难样本回归；"
            "模型切换后可在识别结果中核对实际方案。"
        )
        model_note.setProperty("role", "muted")
        model_note.setWordWrap(True)
        model.addWidget(model_note, 2, 0, 1, 4)
        model.setColumnStretch(1, 1)
        layout.addWidget(model_box)

        parameter_box = QGroupBox("高级识别参数")
        parameters = QGridLayout(parameter_box)
        parameters.setHorizontalSpacing(8)
        parameters.setVerticalSpacing(6)
        parameters.addWidget(QLabel("分类尺寸"), 0, 0)
        parameters.addWidget(self.imgsz, 0, 1)
        parameters.addWidget(QLabel("Top-1 阈值"), 0, 2)
        parameters.addWidget(self.top1_threshold, 0, 3)
        parameters.addWidget(QLabel("候选阈值"), 0, 4)
        parameters.addWidget(self.threshold, 0, 5)
        parameters.addWidget(QLabel("Top-K"), 0, 6)
        parameters.addWidget(self.top_k, 0, 7)
        parameters.addWidget(self.multiview, 1, 0, 1, 2)
        parameters.addWidget(QLabel("局部阈值"), 1, 2)
        parameters.addWidget(self.multiview_threshold, 1, 3)
        parameters.addWidget(QLabel("分割尺寸"), 1, 4)
        parameters.addWidget(self.fusion_seg_imgsz, 1, 5)
        parameters.addWidget(QLabel("mask 置信度"), 1, 6)
        parameters.addWidget(self.fusion_seg_confidence, 1, 7)
        parameters.addWidget(QLabel("格子覆盖率"), 2, 0)
        parameters.addWidget(self.fusion_min_cell_ratio, 2, 1)
        parameters.addWidget(QLabel("mask 占比"), 2, 2)
        parameters.addWidget(self.fusion_min_mask_ratio, 2, 3)
        parameters.addWidget(QLabel("实例分类复核"), 2, 4)
        parameters.addWidget(self.fusion_instance_cls_threshold, 2, 5)
        parameters.addWidget(QLabel("实例最低置信度"), 2, 6)
        parameters.addWidget(self.fusion_instance_confidence, 2, 7)
        parameters.addWidget(QLabel("融合策略"), 3, 0)
        parameters.addWidget(self.fusion_mode, 3, 1, 1, 2)
        parameters.addWidget(QLabel("融合分类尺寸"), 3, 3)
        parameters.addWidget(self.fusion_cls_imgsz, 3, 4)
        for widget in (
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
            widget.setMinimumWidth(82)
        self.advanced_parameters_box = parameter_box
        layout.addWidget(parameter_box)

        click_box = QGroupBox("3×3 / 4×4 点击方案")
        click = QGridLayout(click_box)
        click.setHorizontalSpacing(10)
        click.setVerticalSpacing(8)
        click.addWidget(QLabel("点击间隔（毫秒）"), 0, 0)
        click.addWidget(self.click_delay, 0, 1)
        click.addWidget(QLabel("动态换图等待（秒）"), 0, 2)
        click.addWidget(self.dynamic_wait_seconds, 0, 3)
        click.addWidget(QLabel("异常命中保护"), 0, 4)
        click.addWidget(self.maximum_selected_ratio, 0, 5)
        click.addWidget(self.auto_verify, 0, 6)
        click_note = QLabel(
            "3×3 静态题批量点击；3×3 dynamic 按格点击并等待对应替换图；"
            "4×4 按连续照片的投影结果批量点击。DOM 图块优先，坐标只作兜底。"
        )
        click_note.setProperty("role", "muted")
        click_note.setWordWrap(True)
        click.addWidget(click_note, 1, 0, 1, 7)
        layout.addWidget(click_box)
        layout.addStretch(1)
        return page

    def _build_offline_settings(self) -> QWidget:
        """构建识别与标注的数据和样本筛选设置。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        source_box = QGroupBox("图片与标注")
        source = QGridLayout(source_box)
        source.setHorizontalSpacing(8)
        source.setVerticalSpacing(6)
        source.addWidget(QLabel("挑战类型"), 0, 0)
        source.addWidget(self.challenge, 0, 1)
        source.addWidget(QLabel("网格"), 0, 2)
        source.addWidget(self.grid, 0, 3)
        source.addWidget(QLabel("目标类别"), 0, 4)
        source.addWidget(self.target, 0, 5)
        source.addWidget(QLabel("数据来源"), 1, 0)
        source.addWidget(self.data_source, 1, 1)
        recognition_source_name = (
            "在线图片" if self._selected_source(self.data_source) == "online" else "离线图片"
        )
        self.data_source_hint = QLabel(f"当前使用{recognition_source_name}目录")
        self.data_source_hint.setProperty("role", "muted")
        source.addWidget(self.data_source_hint, 1, 2, 1, 4)
        source.addWidget(QLabel("样本状态"), 2, 0)
        source.addWidget(self.status_filter, 2, 1)
        source.addWidget(self.deduplicate, 2, 2, 1, 2)
        source.setColumnStretch(1, 1)
        source.setColumnStretch(5, 2)
        layout.addWidget(source_box)
        layout.addStretch(1)
        return page

    def _build_online_settings(self) -> QWidget:
        """构建在线采集和在线识别设置。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        capture_box = QGroupBox("浏览器与采集")
        capture = QGridLayout(capture_box)
        capture.setHorizontalSpacing(10)
        capture.setVerticalSpacing(10)
        capture.addWidget(QLabel("解析类型"), 0, 0)
        capture.addWidget(self.online_type, 0, 1)
        capture.addWidget(QLabel("解析类别 / 归档兜底"), 0, 2)
        capture.addWidget(self.online_category, 0, 3, 1, 2)
        capture.addWidget(self.online_enabled, 1, 0, 1, 2)
        capture.addWidget(self.auto_click_tiles, 1, 2)
        capture.addWidget(self.auto_refresh_challenge, 1, 3)
        capture.addWidget(self.monitor_checkbox, 2, 0, 1, 2)
        capture.addWidget(self.clear_site_data, 2, 2, 1, 2)
        capture.addWidget(QLabel("当前目标类别"), 3, 0)
        capture.addWidget(self.online_target, 3, 1)
        capture.addWidget(QLabel("当前网格"), 3, 2)
        capture.addWidget(self.online_grid, 3, 3)
        capture.setColumnStretch(3, 1)
        layout.addWidget(capture_box)
        note = QLabel("在线识别与其他页面共用“识别方案”中的模型和参数。")
        note.setProperty("role", "muted")
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _build_online_data_settings(self) -> QWidget:
        """构建在线图片数据的统计范围设置。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        filter_box = QGroupBox("统计范围")
        filters = QGridLayout(filter_box)
        filters.setHorizontalSpacing(10)
        filters.setVerticalSpacing(10)
        filters.addWidget(QLabel("图片范围"), 0, 0)
        filters.addWidget(self.online_stats_kind, 0, 1)
        filters.addWidget(QLabel("挑战类型"), 0, 2)
        filters.addWidget(self.online_stats_type, 0, 3)
        filters.setColumnStretch(1, 1)
        filters.setColumnStretch(3, 1)
        layout.addWidget(filter_box)

        note_box = QGroupBox("数据说明")
        note_layout = QVBoxLayout(note_box)
        note = QLabel(
            "统计对象为公共在线图片目录中已归档的完整挑战图和替换单格图。"
            "设置筛选范围后，返回“在线图片数据”页点击“刷新统计”。"
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        note_layout.addWidget(note)
        layout.addWidget(note_box)
        layout.addStretch(1)
        return page

    def _build_fusion_settings(self) -> QWidget:
        """构建分类、分割及格子融合设置。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        source_box = QGroupBox("融合验证图片")
        source = QGridLayout(source_box)
        source.setHorizontalSpacing(8)
        source.setVerticalSpacing(6)
        source.addWidget(QLabel("挑战类型"), 0, 0)
        source.addWidget(self.fusion_challenge, 0, 1)
        source.addWidget(QLabel("网格"), 0, 2)
        source.addWidget(self.fusion_grid_label, 0, 3)
        source.addWidget(QLabel("目标类别"), 0, 4)
        source.addWidget(self.fusion_target, 0, 5)
        source.addWidget(QLabel("数据来源"), 1, 0)
        source.addWidget(self.fusion_data_source, 1, 1)
        fusion_source_name = (
            "在线图片"
            if self._selected_source(self.fusion_data_source) == "online"
            else "离线图片"
        )
        self.fusion_data_source_hint = QLabel(
            f"随机图片和下一张当前使用{fusion_source_name}目录"
        )
        self.fusion_data_source_hint.setProperty("role", "muted")
        source.addWidget(self.fusion_data_source_hint, 1, 2, 1, 4)
        source.setColumnStretch(1, 1)
        source.setColumnStretch(5, 1)
        layout.addWidget(source_box)
        note = QLabel("模型、融合策略和阈值统一使用“识别方案”页中的公共配置。")
        note.setProperty("role", "muted")
        layout.addWidget(note)
        layout.addStretch(1)
        return page
