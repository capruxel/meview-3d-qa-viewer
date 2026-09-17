"""Public policy checks for the desktop viewer."""

from meviewer.viewer.window import FIXED_MOTION_CLIM, diagnostic_color_limits


def test_diagnostic_color_limits_support_fixed_and_stable_auto_modes() -> None:
    assert diagnostic_color_limits(0.02, auto=False) == FIXED_MOTION_CLIM
    assert diagnostic_color_limits(0.02, auto=True) == (0.0, 0.02)
    assert diagnostic_color_limits(None, auto=True) == FIXED_MOTION_CLIM
