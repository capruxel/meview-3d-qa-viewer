"""Resolve a viewer input manifest against one host data root."""

from __future__ import annotations

import tomllib
from pathlib import Path


def load(path: Path, data_root: Path) -> dict[str, Path]:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as error:
        raise ValueError(f"Cannot read viewer manifest {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid viewer manifest {path}: {error}") from error
    inputs = document.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"mesh_root", "active_frames"}:
        raise ValueError("Viewer manifest [inputs] must contain only mesh_root and active_frames")
    resolved = {}
    for name, relative in inputs.items():
        if not isinstance(relative, str):
            raise ValueError(f"Viewer manifest {name} must be a string")
        target = (data_root / relative).resolve()
        try:
            target.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"Viewer manifest {name} escapes data root: {relative}") from error
        resolved[name] = target
    return resolved
