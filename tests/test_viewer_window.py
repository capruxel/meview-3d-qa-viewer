"""Public policy checks for the desktop viewer."""

from types import SimpleNamespace

from meviewer.viewer.window import FIXED_MOTION_CLIM, MeshViewer, diagnostic_color_limits


def test_diagnostic_color_limits_support_fixed_and_stable_auto_modes() -> None:
    assert diagnostic_color_limits(0.02, auto=False) == FIXED_MOTION_CLIM
    assert diagnostic_color_limits(0.02, auto=True) == (0.0, 0.02)
    assert diagnostic_color_limits(None, auto=True) == FIXED_MOTION_CLIM


class _Timer:
    def __init__(self) -> None:
        self.interval: int | None = None

    def setInterval(self, interval: int) -> None:
        self.interval = interval

    def isActive(self) -> bool:
        return True


class _Player:
    def __init__(self) -> None:
        self.rate: float | None = None

    def setPlaybackRate(self, rate: float) -> None:
        self.rate = rate


def test_playback_controls_update_timer_interval_and_raw_video_rate() -> None:
    window = SimpleNamespace(
        timer=_Timer(),
        fps_spin=SimpleNamespace(value=lambda: 10),
        speed_spin=SimpleNamespace(value=lambda: 2.0),
        raw_video=SimpleNamespace(fps=40.0),
        raw_player=_Player(),
    )
    window._raw_playback_rate = lambda: MeshViewer._raw_playback_rate(window)

    MeshViewer.update_timer_interval(window)

    assert window.timer.interval == 50
    assert window.raw_player.rate == 0.5
