"""TOML defaults for the standalone viewer tools."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-C",
        "--config-file",
        dest="config_file",
        type=Path,
        help="Load command defaults from this TOML file.",
    )


def apply_config_defaults(parser: argparse.ArgumentParser, section: str) -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("-C", "--config-file", dest="config_file", type=Path)
    config_file = bootstrap.parse_known_args()[0].config_file
    if config_file is None:
        return
    try:
        with config_file.open("rb") as handle:
            document: dict[str, Any] = tomllib.load(handle)
    except OSError as exc:
        raise SystemExit(f"Cannot read config file {config_file}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"Invalid TOML config file {config_file}: {exc}") from exc

    values = document.get(section, {})
    if not isinstance(values, dict):
        raise SystemExit(f"Config section [{section}] must be a TOML table: {config_file}")
    actions = {action.dest: action for action in parser._actions}
    defaults: dict[str, Any] = {}
    for key, value in values.items():
        action = actions.get(key)
        if action is None or key == "config_file":
            continue
        converter = action.type
        if callable(converter) and isinstance(value, str):
            value = cast(Callable[[str], Any], converter)(value)
        defaults[key] = value
    parser.set_defaults(**defaults)
