"""在线归档统计页。

从 ``qt_gui.py`` 抽出：这一页只做只读统计，不参与识别流程，
与主窗口其余部分几乎没有耦合。
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QComboBox, QGridLayout, QMessageBox, QPlainTextEdit, QSplitter

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data.online_stats import scan_online_capture
from .theme import set_button_role


class OnlineDataTabMixin:
    """在线图片数据页：按内容哈希统计归档、查看精确重复组。

    该页只读，不删除也不移动任何图片；与识别链路完全解耦，
    因此从主窗口抽出为独立 mixin。
    """

    def _build_online_data_tab(self) -> None:
        """构建在线采集图片的只读统计页面。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(8)

        # 统计范围放在公共设置页，数据页只展示统计结果。
        self.online_stats_kind = QComboBox()
        self.online_stats_kind.addItem("全部在线图片", "all")
        self.online_stats_kind.addItem("完整挑战图", "full_challenge")
        self.online_stats_kind.addItem("替换单格图", "replacement_tile")
        self.online_stats_type = QComboBox()
        self.online_stats_type.addItems(
            ["全部", "dynamic", "imageselect", "tileselect", "multicaptcha"]
        )
        summary_box = QGroupBox("统计摘要")
        summary = QGridLayout(summary_box)
        summary_titles = ("图片总数", "唯一内容", "精确重复组", "多余副本")
        self.online_stats_summary_labels: list[QLabel] = []
        for column, title in enumerate(summary_titles):
            title_label = QLabel(title)
            title_label.setProperty("role", "muted")
            value_label = QLabel("-")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setProperty("role", "statValue")
            summary.addWidget(title_label, 0, column, alignment=Qt.AlignCenter)
            summary.addWidget(value_label, 1, column)
            summary.setColumnStretch(column, 1)
            self.online_stats_summary_labels.append(value_label)
        self.online_stats_message = QLabel(
            "点击“刷新统计”后扫描所选在线图片目录；统计过程只读取图片。"
        )
        self.online_stats_message.setProperty("role", "muted")
        self.online_stats_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary.addWidget(self.online_stats_message, 2, 0, 1, 4)
        layout.addWidget(summary_box)

        category_box = QGroupBox("类型与类别分布")
        category_layout = QVBoxLayout(category_box)
        self.online_category_table = QTableWidget()
        self._setup_online_stats_table(
            self.online_category_table,
            ["归档来源", "挑战类型", "中文类别", "图片数", "唯一内容"],
        )
        self.online_category_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        category_layout.addWidget(self.online_category_table)

        duplicate_box = QGroupBox("精确重复组（SHA-256）")
        duplicate_layout = QVBoxLayout(duplicate_box)
        self.online_duplicate_table = QTableWidget()
        self._setup_online_stats_table(
            self.online_duplicate_table,
            ["重复组", "相同文件数", "多余副本", "归档来源", "类别"],
        )
        self.online_duplicate_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        self.online_duplicate_table.itemSelectionChanged.connect(
            self._show_online_duplicate_detail
        )
        duplicate_layout.addWidget(self.online_duplicate_table, 2)
        self.online_duplicate_detail = QPlainTextEdit()
        self.online_duplicate_detail.setReadOnly(True)
        self.online_duplicate_detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.online_duplicate_detail.setObjectName("reportEditor")
        self.online_duplicate_detail.setPlaceholderText(
            "选择一个重复组后，这里会显示完整 SHA-256 和全部相对路径。"
        )
        self.online_duplicate_detail.setMinimumHeight(150)
        duplicate_layout.addWidget(self.online_duplicate_detail, 1)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(category_box)
        body.addWidget(duplicate_box)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        body.setSizes([620, 700])
        layout.addWidget(body, 1)
        actions = QHBoxLayout()
        actions.addStretch(1)
        refresh_button = QPushButton("刷新统计")
        set_button_role(refresh_button, "primary")
        refresh_button.clicked.connect(self._refresh_online_stats)
        actions.addWidget(refresh_button)
        layout.insertLayout(0, actions)
        self.tabs.addTab(tab, "在线图片数据")

    @staticmethod
    def _setup_online_stats_table(table: QTableWidget, headers: list[str]) -> None:
        """设置统计表格的统一只读行为。"""
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _set_table_rows(table: QTableWidget, rows: list[list[object]]) -> None:
        """一次性更新只读表格，数字列仍按数值顺序显示。"""
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if isinstance(value, int):
                    item.setData(Qt.DisplayRole, value)
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_index, column_index, item)
        table.setSortingEnabled(True)

    def _refresh_online_stats(self) -> None:
        """按当前筛选条件统计在线采集图片和精确重复内容。"""
        capture_root = Path(self.online_data.text()).expanduser()
        archive_kind = str(self.online_stats_kind.currentData() or "all")
        challenge_type = self.online_stats_type.currentText()
        self.online_stats_message.setText("正在读取图片并计算 SHA-256，请稍候…")
        self.status.setText("正在统计在线采集图片…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = scan_online_capture(
                capture_root,
                archive_kind=archive_kind,
                challenge_type=challenge_type,
            )
        except Exception as exc:
            self.online_stats_message.setText(f"统计失败：{exc}")
            self.status.setText("在线图片统计失败")
            QMessageBox.warning(self, "在线图片统计", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        summary_values = (
            result["total"],
            result["unique"],
            result["duplicate_groups"],
            result["extra"],
        )
        for label, value in zip(self.online_stats_summary_labels, summary_values):
            label.setText(f"{value:,}")

        category_rows = [
            [
                row["archive_label"],
                row["challenge_type"],
                row["category"],
                row["count"],
                row["unique"],
            ]
            for row in result["category_rows"]
        ]
        self._set_table_rows(self.online_category_table, category_rows)

        self.online_duplicate_rows = result["duplicate_rows"]
        duplicate_rows = [
            [
                f"{row['sha256'][:12]}…",
                row["count"],
                row["extra"],
                "、".join(row["archive_labels"]),
                "、".join(row["categories"]),
            ]
            for row in self.online_duplicate_rows
        ]
        self._set_table_rows(self.online_duplicate_table, duplicate_rows)
        self.online_duplicate_detail.clear()

        relative_root = self._display_path(capture_root)
        self.online_stats_message.setText(
            f"扫描目录：{relative_root}  |  重复组内文件：{result['duplicate_files']:,} 张"
        )
        self.status.setText(
            f"在线图片统计完成：{result['total']:,} 张，"
            f"精确重复 {result['duplicate_groups']:,} 组，多余副本 {result['extra']:,} 张"
        )

    def _show_online_duplicate_detail(self) -> None:
        """显示当前重复组的完整摘要和所有文件路径。"""
        selected = self.online_duplicate_table.selectionModel().selectedRows()
        if not selected:
            self.online_duplicate_detail.clear()
            return
        row_index = selected[0].row()
        digest_item = self.online_duplicate_table.item(row_index, 0)
        if digest_item is None:
            return
        short_digest = digest_item.text().rstrip("…")
        row = next(
            (
                item
                for item in getattr(self, "online_duplicate_rows", [])
                if item["sha256"].startswith(short_digest)
            ),
            None,
        )
        if row is None:
            return
        paths = [self._display_path(Path(path)) for path in row["files"]]
        detail = [
            f"SHA-256：{row['sha256']}",
            f"文件数：{row['count']}",
            f"多余副本：{row['extra']}",
            f"归档来源：{'、'.join(row['archive_labels'])}",
            f"挑战类型：{'、'.join(row['challenge_types'])}",
            f"类别：{'、'.join(row['categories'])}",
            "",
            "组内文件：",
            *paths,
        ]
        self.online_duplicate_detail.setPlainText("\n".join(detail))
