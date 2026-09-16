"""Local PySide6/PyVista QA viewer for MEVIEW dense-mesh sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QWidget,
)


class ActiveFrameSlider(QSlider):
    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.active_start: int | None = None
        self.active_end: int | None = None
        self.reference: int | None = None

    def set_markers(self, start: int | None, end: int | None, reference: int | None) -> None:
        self.active_start, self.active_end, self.reference = start, end, reference
        self.update()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        scale = groove.width() / (self.maximum() - self.minimum())

        def position(frame: int) -> int:
            return groove.x() + round((frame - self.minimum()) * scale)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        marker_top, marker_bottom = groove.y() - 5, groove.bottom() + 5
        if self.active_start is not None and self.active_end is not None:
            start, end = position(self.active_start), position(self.active_end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e6a23c80"))
            painter.drawRoundedRect(start, groove.y() - 2, max(3, end - start + 1), 4, 2, 2)
            painter.setPen(QColor("#e6a23c"))
            painter.drawLine(start, marker_top, start, marker_bottom)
            painter.drawLine(end, marker_top, end, marker_bottom)
        if self.reference is not None:
            reference = position(self.reference)
            painter.setPen(QColor("#5dade2"))
            painter.drawLine(reference, marker_top - 2, reference, marker_bottom + 2)
        painter.end()


class LandmarkImageWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.image = QPixmap()
        self.frame: Any = None
        self.status = "No MediaPipe record"
        self.show_478 = True
        self.show_20 = True
        self.show_rois = True
        self.point_size = 10

    def set_frame(self, image_path: Path, frame: Any) -> None:
        self.image = QPixmap(str(image_path)) if image_path.is_file() else QPixmap()
        self.frame = frame
        self.status = (
            "No illustration image"
            if self.image.isNull()
            else (
                "No face"
                if frame is not None and frame.status == "no_face"
                else "Ready"
                if frame is not None
                else "No MediaPipe record"
            )
        )
        self.update()

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202124"))
        if self.image.isNull():
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.status)
            painter.end()
            return
        target = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target_rect = target.toRect()
        target_rect.moveTo(
            (self.width() - target_rect.width()) // 2,
            (self.height() - target_rect.height()) // 2,
        )
        painter.drawPixmap(target_rect, self.image)
        if self.frame is not None and self.frame.status == "ok":

            def draw(points: np.ndarray, color: str) -> None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(color))
                radius = max(2, self.point_size / 2)
                for x, y, _ in points:
                    if not np.isfinite((x, y)).all():
                        continue
                    px = target_rect.left() + float(np.clip(x, 0, 1)) * target_rect.width()
                    py = target_rect.top() + float(np.clip(y, 0, 1)) * target_rect.height()
                    painter.drawEllipse(
                        round(px - radius),
                        round(py - radius),
                        round(2 * radius),
                        round(2 * radius),
                    )

            if self.show_478:
                draw(self.frame.landmarks478, "#44aaff")
            if self.show_20 and len(self.frame.landmarks20):
                draw(self.frame.landmarks20, "#ffcc00")
            if self.show_rois and len(self.frame.roi_centers):
                draw(self.frame.roi_centers, "#ff5555")
        painter.setPen(QColor("white"))
        painter.drawText(8, 20, self.status)
        painter.end()
