import subprocess
import sys


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
