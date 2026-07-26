"""PySide6 主界面统一主题。"""

APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #f3f6fa;
    color: #182230;
    font-family: "PingFang SC", "Microsoft YaHei";
    font-size: 13px;
}
QLabel, QCheckBox {
    background: transparent;
}
QTabWidget::pane {
    border: 1px solid #d8e0ea;
    border-radius: 14px;
    background: #ffffff;
    top: -2px;
}
QTabBar {
    background: transparent;
}
QTabBar::tab {
    min-width: 134px;
    min-height: 40px;
    margin-right: 5px;
    padding: 2px 16px;
    border: 1px solid #d8e0ea;
    border-bottom: none;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    background: #e9eef5;
    color: #526071;
    font-weight: 600;
}
QTabBar::tab:hover {
    background: #e1eaf6;
    color: #1d5fbf;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #175cd3;
    border-color: #9db7dc;
}
QTabWidget#settingsTabs::pane {
    border-radius: 10px;
    border-color: #dce3ec;
    background: #f8fafc;
}
QTabWidget#settingsTabs QTabBar::tab {
    min-width: 118px;
    min-height: 34px;
    padding: 1px 13px;
    background: #eef2f7;
    font-weight: 500;
}
QTabWidget#settingsTabs QTabBar::tab:selected {
    background: #f8fafc;
    color: #175cd3;
    font-weight: 600;
}
QWidget#settingsHeader {
    border: 1px solid #cddcf2;
    border-radius: 12px;
    background: #eef5ff;
}
QGroupBox {
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    border: 1px solid #dce3ec;
    border-radius: 11px;
    background: #ffffff;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 13px;
    padding: 0 7px;
    color: #29384d;
    background: #ffffff;
}
QPushButton {
    min-height: 34px;
    padding: 2px 13px;
    border: 1px solid #c8d2df;
    border-radius: 8px;
    background: #ffffff;
    color: #344256;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #78a5e3;
    background: #eff6ff;
    color: #175cd3;
}
QPushButton:pressed {
    background: #dbeafe;
}
QPushButton[role="primary"] {
    border-color: #175cd3;
    background: #175cd3;
    color: #ffffff;
}
QPushButton[role="primary"]:hover {
    border-color: #164ca6;
    background: #164ca6;
    color: #ffffff;
}
QPushButton[role="danger"] {
    border-color: #f4b8b3;
    background: #fff4f2;
    color: #c4320a;
}
QPushButton[role="danger"]:hover {
    border-color: #e78980;
    background: #ffe9e5;
    color: #9c2a10;
}
QComboBox, QLineEdit {
    min-height: 32px;
    padding: 0 8px;
    border: 1px solid #c8d2df;
    border-radius: 8px;
    background: #ffffff;
    color: #263548;
    selection-background-color: #cfe2ff;
}
QComboBox:hover, QLineEdit:hover {
    border-color: #9eb4cf;
}
QComboBox:focus, QLineEdit:focus {
    border: 1px solid #4d88d8;
    background: #ffffff;
}
QComboBox QAbstractItemView {
    border: 1px solid #aebdd0;
    border-radius: 8px;
    background: #ffffff;
    color: #263548;
    padding: 6px;
    outline: 0;
    selection-background-color: #deebff;
    selection-color: #174b91;
}
QComboBox QAbstractItemView::item {
    min-height: 32px;
    padding: 3px 8px;
}
QPlainTextEdit {
    border: 1px solid #d7dfe9;
    border-radius: 9px;
    background: #f8fafc;
    color: #263548;
    padding: 8px;
    selection-background-color: #cfe2ff;
}
QPlainTextEdit#reportEditor {
    border: 1px solid #c8d2df;
    border-radius: 10px;
    background: #f8fafc;
    color: #1c2d42;
    padding: 12px;
    font-family: Menlo;
    font-size: 12px;
    selection-background-color: #c4dcff;
    selection-color: #102a43;
}
QPlainTextEdit#reportEditor:focus {
    border: 1px solid #6296dc;
    background: #ffffff;
}
QCheckBox {
    spacing: 8px;
    color: #344256;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QLabel[role="status"] {
    min-height: 22px;
    padding: 7px 11px;
    border: 1px solid #c9dcf5;
    border-radius: 8px;
    background: #edf5ff;
    color: #24579b;
    font-weight: 600;
}
QLabel[role="muted"] {
    color: #66758a;
}
QLabel[role="pageTitle"] {
    color: #153b72;
    font-size: 20px;
    font-weight: 700;
}
QLabel[role="statValue"] {
    padding: 5px 8px;
    color: #175cd3;
    font-size: 24px;
    font-weight: 700;
}
QTableWidget {
    border: 1px solid #d7dfe9;
    border-radius: 9px;
    background: #ffffff;
    alternate-background-color: #f8fafc;
    color: #263548;
    gridline-color: #e7ebf1;
    selection-background-color: #deebff;
    selection-color: #174b91;
}
QHeaderView::section {
    min-height: 34px;
    padding: 4px 8px;
    border: none;
    border-right: 1px solid #dde4ed;
    border-bottom: 1px solid #ccd6e2;
    background: #edf2f7;
    color: #344256;
    font-weight: 600;
}
QSplitter::handle {
    background: #e8edf3;
    width: 6px;
}
QSplitter::handle:hover {
    background: #a8bfde;
}
"""


def set_button_role(button, role: str) -> None:
    """设置按钮语义样式。"""
    button.setProperty("role", role)
