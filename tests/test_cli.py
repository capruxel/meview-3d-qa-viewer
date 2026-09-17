import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "meviewer.cli.main", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_lists_commands() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert all(command in result.stdout for command in ("audit", "extract", "viewer"))


def test_dispatches_help_without_optional_landmarks() -> None:
    result = run_cli("extract", "--help")
    assert result.returncode == 0
    assert "--output-root" in result.stdout
    assert "Install the viewer landmarks extra" not in result.stderr


def test_unknown_command_exits_with_argparse_code() -> None:
    result = run_cli("unknown")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_viewer_argument_failures_use_argparse_exit_code(tmp_path: Path) -> None:
    no_root = run_cli("viewer", "--active-frames", str(tmp_path / "active.json"))
    no_active_frames = run_cli("viewer", "--data-root", str(tmp_path))
    manifest = tmp_path / "viewer.toml"
    manifest.write_text("not = [valid", encoding="utf-8")
    invalid_manifest = run_cli("viewer", "--data-root", str(tmp_path), "--manifest", str(manifest))
    for result in (no_root, no_active_frames, invalid_manifest):
        assert result.returncode == 2
        assert result.stderr.startswith("usage:")
    assert "--data-root is required" in no_root.stderr
    assert "--active-frames or --manifest is required" in no_active_frames.stderr
    assert "Invalid viewer manifest" in invalid_manifest.stderr
