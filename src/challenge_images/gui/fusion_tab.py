"""分割与融合页。

从 ``qt_gui.py`` 抽出：这一页负责载入完整挑战图、运行分类与整图分割的
融合识别、扫描重复图片，并把融合结果应用到在线网页。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import ASSETS_DIR, DEFAULT_DEVICE, RECOMMENDED_SEGMENTATION_MODEL, available_segmentation_model_choices, resolve_default_segmentation_weight
from ..data.sample_manager import SampleManager
from ..grid.grid_engine import draw_grid, grid_for_challenge
from ..recognition import PARAMETER_PRESET_LABELS
from .state import RecognitionOutcome, RecognitionRequest, SOURCE_FUSION
from .theme import set_button_role
from .widgets import _preview_pixmap
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget
from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover
    # mixin 总是被混入 QMainWindow；声明基类让需要 QWidget 的调用
    # 通过类型检查，运行时仍是普通 mixin。
    from PySide6.QtWidgets import QWidget as _MixinBase
else:
    _MixinBase = object


class FusionTabMixin(_MixinBase):
    """分割与融合页：分类 Top-K 与整图 mask 的融合验证。"""

    if TYPE_CHECKING:  # pragma: no cover
        # 以下属性与方法由宿主窗口 QtChallengeGUI 提供。
        # 显式声明让类型检查通过，同时把隐式耦合变成可读的契约。
        fusion_image: Any
        fusion_image_key: Any
        fusion_source: Any
        fusion_sample: Any
        fusion_manager: Any
        fusion_indices: Any
        fusion_preview_image: Any
        challenge: Any
        current: Any
        data_source: Any
        deduplicate: Any
        fusion_canvas: Any
        fusion_challenge: Any
        fusion_data_source: Any
        fusion_data_source_hint: Any
        fusion_detail: Any
        fusion_grid_label: Any
        fusion_seg_weights: Any
        fusion_source_label: Any
        fusion_target: Any
        gui_settings: Any
        image: Any
        offline_data: Any
        online_data: Any
        online_image: Any
        online_image_sha256: Any
        online_sample: Any
        online_target: Any
        online_type: Any
        online_worker: Any
        parameter_preset: Any
        recognition_mode: Any
        segmentation_service: Any
        service: Any
        status: Any
        tabs: Any
        def _choose_model_for_combo(self, *args: Any, **kwargs: Any) -> Any: ...
        def _compose_report(self, *args: Any, **kwargs: Any) -> Any: ...
        def _display_path(self, *args: Any, **kwargs: Any) -> Any: ...
        def _online_click_plan(self, *args: Any, **kwargs: Any) -> Any: ...
        def _parameter_preset_changed(self, *args: Any, **kwargs: Any) -> Any: ...
        def _recognition_parameters(self, *args: Any, **kwargs: Any) -> Any: ...
        def _selected_combo_weight(self, *args: Any, **kwargs: Any) -> Any: ...
        def _selected_source(self, *args: Any, **kwargs: Any) -> Any: ...
        def _selected_weight(self, *args: Any, **kwargs: Any) -> Any: ...
        def _show_actual_recognition_route(self, *args: Any, **kwargs: Any) -> Any: ...
        def _submit_online_click_plan(self, *args: Any, **kwargs: Any) -> Any: ...
        def _submit_recognition(self, *args: Any, **kwargs: Any) -> Any: ...

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

    def _choose_fusion_seg_weights(self) -> None:
        self._choose_model_for_combo(self.fusion_seg_weights, "选择 YOLO 分割模型权重")

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

    def _active_fusion_root(self) -> Path:
        """返回分割与融合当前选择的根目录。"""
        field = (
            self.online_data
            if self._selected_source(self.fusion_data_source) == "online"
            else self.offline_data
        )
        return Path(field.text()).expanduser()

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
