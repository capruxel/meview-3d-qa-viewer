"""Offscreen public interaction checks for the review widgets."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from meviewer.annotations import RegionalMotionAnnotation
from meviewer.viewer.widgets import ActiveFrameSlider, RegionalMotionChart


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_review_widgets_render_and_emit_inclusive_gestures() -> None:
    app()
    annotation = RegionalMotionAnnotation(
        sequence="lfann-v3/sub01/01",
        roi_name="left_brow",
        start_frame=10,
        end_frame=11,
        direction="raise",
        confidence=3,
    )
    chart = RegionalMotionChart()
    chart.resize(400, 280)
    chart.set_data(
        (10, 11),
        {"left_brow": ((0.0, 1.0), (None, 0.5), (None, None))},
        (10, 11),
        0,
        (annotation,),
    )
    selected: list[int] = []
    dragged: list[tuple[int, int]] = []
    chart.frame_selected.connect(selected.append)
    chart.drag_selected.connect(lambda start, end: dragged.append((start, end)))
    chart.show()
    assert not chart.grab().isNull()
    QTest.mouseClick(
        chart, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 80)
    )
    QTest.mousePress(
        chart, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 80)
    )
    QTest.mouseMove(chart, QPoint(390, 80))
    QTest.mouseRelease(
        chart, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(390, 80)
    )
    assert selected == [0]
    assert dragged == [(0, 1)]
    assert chart.accessibleName()
    assert chart.toolTip()

    slider = ActiveFrameSlider()
    slider.setRange(0, 1)
    slider.resize(400, 32)
    slider.set_markers(0, 1, 0)
    slider.set_annotation_markers([(0, 1)])
    slider_dragged: list[tuple[int, int]] = []
    slider.drag_selected.connect(lambda start, end: slider_dragged.append((start, end)))
    slider.show()
    assert not slider.grab().isNull()
    QTest.mousePress(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 16)
    )
    QTest.mouseMove(slider, QPoint(390, 16))
    QTest.mouseRelease(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(390, 16)
    )
    assert slider_dragged == [(0, 1)]
    assert slider.accessibleName()
    assert slider.toolTip()
