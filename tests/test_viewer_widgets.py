"""Offscreen public interaction checks for evidence-review widgets."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from meviewer.viewer.session import RegionalMotionSeries
from meviewer.viewer.widgets import ActiveFrameSlider, RegionalMotionChart


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def chart_data() -> dict[str, RegionalMotionSeries]:
    return {
        "left_brow": RegionalMotionSeries((0.0, 1.0, 0.5), (None, 1.0, -0.5), (None, None, -1.5)),
        "right_brow": RegionalMotionSeries(
            (0.0, 0.5, 0.25), (None, 0.5, -0.25), (None, None, -0.75)
        ),
        "left_mouth_corner": RegionalMotionSeries(
            (0.0, -0.5, -0.25), (None, -0.5, 0.25), (None, None, 0.75)
        ),
        "right_mouth_corner": RegionalMotionSeries(
            (0.0, -1.0, -0.5), (None, -1.0, 0.5), (None, None, 1.5)
        ),
    }


def test_chart_left_edge_selects_first_frame() -> None:
    app()
    chart = RegionalMotionChart()
    chart.resize(400, 280)
    chart.set_data((10, 11, 12), chart_data(), (10, 11), 0, status=None)
    selected: list[int] = []
    chart.frame_selected.connect(selected.append)

    QTest.mouseClick(
        chart, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(0, 80)
    )

    assert selected == [0]


def test_chart_velocity_accessibility_describes_evidence_only() -> None:
    app()
    chart = RegionalMotionChart()
    chart.set_data((10, 11, 12), chart_data(), (10, 11), 0, metric="velocity", status=None)

    assert chart.motion_metric == "velocity"
    assert chart.accessibleName() == "Regional motion chart: velocity"
    assert "v_y (up +)" in chart.toolTip()
    assert "annotation" not in chart.toolTip().lower()
    assert "drag" not in chart.toolTip().lower()


def test_chart_renders_finite_active_evidence() -> None:
    app()
    chart = RegionalMotionChart()
    chart.resize(400, 280)
    chart.set_data((10, 11, 12), chart_data(), (10, 11), 0, status=None)
    chart.show()

    assert not chart.grab().isNull()
    assert chart.active == (10, 11)


def test_slider_exposes_only_frame_selection() -> None:
    app()
    slider = ActiveFrameSlider()
    slider.setRange(0, 2)
    slider.resize(400, 32)
    slider.set_markers(0, 1, 0)
    assert slider.accessibleName() == "Frame selector with active-window markers"
    slider.show()

    assert not hasattr(slider, "drag_selected")
    assert not hasattr(slider, "annotation_markers")
    assert not hasattr(slider, "set_annotation_markers")
    QTest.mousePress(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(10, 16)
    )
    QTest.mouseMove(slider, QPoint(390, 16))
    QTest.mouseRelease(
        slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(390, 16)
    )
    assert slider.value() == 2
