"""Public policy checks for the desktop viewer."""

from types import SimpleNamespace

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMainWindow

from meviewer.viewer.window import FIXED_MOTION_CLIM, MeshViewer, diagnostic_color_limits


def test_diagnostic_color_limits_support_fixed_and_stable_auto_modes() -> None:
    assert diagnostic_color_limits(0.02, auto=False) == FIXED_MOTION_CLIM
    assert diagnostic_color_limits(0.02, auto=True) == (0.0, 0.02)
    assert diagnostic_color_limits(None, auto=True) == FIXED_MOTION_CLIM


class _ReviewChart(QObject):
    frame_selected = Signal(int)


class _Player:
    def __init__(self) -> None:
        self.rate: float | None = None

    def setPlaybackRate(self, rate: float) -> None:
        self.rate = rate

    def stop(self) -> None:
        pass


def test_playback_controls_update_timer_interval_and_raw_video_rate(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MeshViewer.__new__(MeshViewer)
    QMainWindow.__init__(window)
    window.timer = QTimer(window)
    window.snapshot = SimpleNamespace(raw_video=SimpleNamespace(fps=40.0))
    window.raw_player = _Player()
    window.review_chart = _ReviewChart()
    window.set_frame_by_index = lambda _: None
    window.first_frame = lambda: None
    window.previous_frame = lambda: None
    window.toggle_play = lambda: None
    window.next_frame = lambda: None
    window.set_reference_to_current = lambda: None
    window.player_panel = window._player()

    window.fps_spin.setValue(10)
    window.speed_spin.setValue(2.0)
    app.processEvents()

    assert window.timer.interval() == 50
    window.timer.start()
    window.speed_spin.setValue(1.0)
    app.processEvents()
    assert window.raw_player.rate == 0.25
    window.timer.stop()
