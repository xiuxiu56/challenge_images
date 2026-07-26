"""GUI 通用控件与图像转换辅助。

从 ``qt_gui.py`` 抽出：这些与业务逻辑无关，被三个功能页共用，
留在 2000 多行的主窗口文件里只会增加阅读成本。
"""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import QLineEdit


def _pixmap(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


PREVIEW_MAX_SIZE = (460, 280)


class NumericLineEdit(QLineEdit):
    """不带步进按钮的数值输入框。

    保留 value/setValue 接口，让原有识别流程专注于参数值，
    界面上则只展示普通文本输入框。
    """

    def __init__(
        self,
        value: int | float,
        minimum: int | float,
        maximum: int | float,
        *,
        decimals: int = 3,
        integer: bool = False,
    ) -> None:
        super().__init__()
        self._minimum = minimum
        self._maximum = maximum
        self._decimals = decimals
        self._integer = integer
        if integer:
            self.setValidator(QIntValidator(int(minimum), int(maximum), self))
        else:
            validator = QDoubleValidator(float(minimum), float(maximum), decimals, self)
            validator.setNotation(QDoubleValidator.StandardNotation)
            self.setValidator(validator)
        self.setAlignment(Qt.AlignCenter)
        self.setValue(value)

    def value(self) -> int | float:
        """返回边界内的数值；空输入回退到下限。"""
        try:
            parsed = int(self.text()) if self._integer else float(self.text())
        except ValueError:
            parsed = self._minimum
        return max(self._minimum, min(self._maximum, parsed))

    def setValue(self, value: int | float) -> None:
        """用紧凑形式回填数值。"""
        bounded = max(self._minimum, min(self._maximum, value))
        if self._integer:
            self.setText(str(int(bounded)))
            return
        text = f"{float(bounded):.{self._decimals}f}".rstrip("0").rstrip(".")
        self.setText(text or "0")


def _preview_pixmap(image: Image.Image) -> QPixmap:
    """按在线采集的统一规格生成等比例预览图。

    在线和离线都使用 460×280 的最大显示边界。对于正方形
    3×3 / 4×4 挑战图，最终统一显示为 280×280。
    """
    preview = image.copy()
    preview.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    return _pixmap(preview)
