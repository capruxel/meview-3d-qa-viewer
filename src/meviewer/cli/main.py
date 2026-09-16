"""Unified command-line entry point for MEVIEW tools."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

_COMMANDS = {
    "audit": ("Dataset audit", "meviewer.cli.audit"),
    "extract": ("MediaPipe landmark extraction", "meviewer.cli.extract"),
    "viewer": ("Interactive mesh viewer", "meviewer.cli.viewer"),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=sorted(_COMMANDS), nargs="?")
        parser.epilog = "\n".join(
            f"  {name}: {description}" for name, (description, _) in _COMMANDS.items()
        )
        parser.print_help()
        return 0

    command = args.pop(0)
    try:
        module_name = _COMMANDS[command][1]
    except KeyError:
        choices = ", ".join(sorted(_COMMANDS))
        print(f"meviewer: invalid choice: {command!r} (choose from {choices})", file=sys.stderr)
        return 2
    module = __import__(module_name, fromlist=["main"])
    original_argv = sys.argv
    try:
        sys.argv = [f"meviewer {command}", *args]
        return module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
