import subprocess
import sys
import unittest


class UnifiedCliTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "meviewer.cli.main", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_lists_commands(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("audit", result.stdout)
        self.assertIn("extract", result.stdout)
        self.assertIn("viewer", result.stdout)

    def test_dispatches_help_without_optional_landmarks(self) -> None:
        result = self.run_cli("extract", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--output-root", result.stdout)
        self.assertNotIn("Install the viewer landmarks extra", result.stderr)


if __name__ == "__main__":
    unittest.main()
