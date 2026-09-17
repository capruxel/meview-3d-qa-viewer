"""Local PySide6/PyVista QA viewer for MEVIEW dense-mesh sequences."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider, QWidget

from meviewer.viewer.session import RegionalMotionSeries


class ActiveFrameSlider(QSlider):
    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.active_start: int | None = None
        self.active_end: int | None = None
        self.reference: int | None = None
        self.setAccessibleName("Frame selector with active-window markers")
        self.setToolTip("Select a frame.")

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
        if self.reference is not None:
            reference = position(self.reference)
            painter.setPen(QColor("#5dade2"))
            painter.drawLine(reference, marker_top - 2, reference, marker_bottom + 2)
        painter.end()


class RegionalMotionChart(QWidget):
    frame_selected = Signal(int)

    _ROIS = ("left_brow", "right_brow", "left_mouth_corner", "right_mouth_corner")
    _ROI_STYLES = (
        ("Left brow", "#38bdf8", Qt.PenStyle.SolidLine),
        ("Right brow", "#4ade80", Qt.PenStyle.DashLine),
        ("Left mouth corner", "#fbbf24", Qt.PenStyle.DotLine),
        ("Right mouth corner", "#f472b6", Qt.PenStyle.DashDotLine),
    )
    _METRICS: dict[Literal["displacement", "velocity", "acceleration"], tuple[str, str]] = {
        "displacement": ("Δy (up +)", "displacement"),
        "velocity": ("v_y (up +)", "velocity"),
        "acceleration": ("a_y (up +)", "acceleration"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.frames: tuple[int, ...] = ()
        self.data: Mapping[str, RegionalMotionSeries] = {}
        self.active: tuple[int, int] | None = None
        self.selected = 0
        self.motion_metric: Literal["displacement", "velocity", "acceleration"] = "displacement"
        self.status: str | None = None
        self.setMinimumHeight(320)
        self._set_accessibility()

    def _set_accessibility(self) -> None:
        label, name = self._METRICS[self.motion_metric]
        self.setAccessibleName(f"Regional motion chart: {name}")
        self.setToolTip(f"{label} regional-motion evidence. Click to select a frame.")

    def set_data(
        self,
        frames: tuple[int, ...],
        data: Mapping[str, RegionalMotionSeries],
        active: tuple[int, int] | None,
        selected: int,
        *,
        metric: Literal["displacement", "velocity", "acceleration"] = "displacement",
        status: str | None = None,
    ) -> None:
        self.frames, self.data, self.active, self.selected = frames, data, active, selected
        self.motion_metric, self.status = metric, status
        self._set_accessibility()
        self.update()

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202124"))
        if len(self.frames) < 2:
            painter.end()
            return
        left, right, top, bottom = 58, 10, 68, 32
        width, height = self.width() - left - right, self.height() - top - bottom
        xscale = width / (len(self.frames) - 1)
        indices = {frame: index for index, frame in enumerate(self.frames)}
        label, _ = self._METRICS[self.motion_metric]
        series = {roi: getattr(values, self.motion_metric) for roi, values in self.data.items()}
        finite = [
            float(value)
            for points in series.values()
            for value in points
            if value is not None and np.isfinite(value)
        ]
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(left, 16, label)
        for row, offset in enumerate((0, 2)):
            for col in range(2):
                display, color, style = self._ROI_STYLES[offset + col]
                legend_x = left + col * (width // 2)
                y = 34 + row * 16
                painter.setPen(QPen(QColor(color), 2, style))
                painter.drawLine(legend_x, y - 4, legend_x + 16, y - 4)
                painter.setPen(QColor("#f8fafc"))
                painter.drawText(legend_x + 20, y, display)
        if not finite:
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.status or "Regional motion unavailable",
            )
            painter.end()
            return
        if self.active and self.active[0] in indices and self.active[1] in indices:
            start, end = indices[self.active[0]], indices[self.active[1]]
            painter.fillRect(
                round(left + start * xscale),
                top,
                max(1, round((end - start) * xscale)),
                height,
                QColor("#f5b04130"),
            )
        zero = round(top + height / 2)
        scale = max(abs(value) for value in finite)
        painter.setPen(QColor("#475569"))
        painter.drawLine(left, zero, self.width() - right, zero)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(4, top + 10, f"{scale:.3g}")
        painter.drawText(4, top + height, f"{-scale:.3g}")
        painter.drawText(left, self.height() - 8, str(self.frames[0]))
        end_label = str(self.frames[-1])
        painter.drawText(
            self.width() - right - painter.fontMetrics().horizontalAdvance(end_label),
            self.height() - 8,
            end_label,
        )
        for roi, (_, color, style) in zip(self._ROIS, self._ROI_STYLES, strict=True):
            points = series.get(roi, ())
            painter.setPen(QPen(QColor(color), 2, style))
            for index in range(1, min(len(points), len(self.frames))):
                previous, current = points[index - 1], points[index]
                if (
                    previous is not None
                    and current is not None
                    and np.isfinite(previous)
                    and np.isfinite(current)
                ):
                    painter.drawLine(
                        round(left + (index - 1) * xscale),
                        round(zero - float(previous) / scale * height / 2),
                        round(left + index * xscale),
                        round(zero - float(current) / scale * height / 2),
                    )
        painter.setPen(QColor("#f8fafc"))
        x = left + self.selected * xscale
        painter.drawLine(round(x), top, round(x), top + height)
        painter.end()

    def _index_from_x(self, x: float) -> int:
        return max(
            0,
            min(
                len(self.frames) - 1,
                round((x - 58) / max(1, self.width() - 68) * (len(self.frames) - 1)),
            ),
        )

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.frames:
            self.frame_selected.emit(self._index_from_x(event.position().x()))
        super().mouseReleaseEvent(event)


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
