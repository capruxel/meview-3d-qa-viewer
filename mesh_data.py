"""Shared indexing and validation for MEVIEW dense-mesh sequences."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = {
    "v2": "frontal_meshes_MEVIEW_v2",
    "v3": "frontal_meshes_MEVIEW_v3",
    "lfann-v3": "frontal_meshes_LFANN_v3",
}
ASSET_RE = re.compile(r"^(?P<frame>\d+)_frontal_(?P<kind>vertices|mesh|illustration)\.(?P<suffix>npy|obj|jpg)$")


@dataclass(frozen=True)
class SequenceIndex:
    variant: str
    subject: str
    video: str
    directory: Path
    frames: dict[int, dict[str, Path]]

    @property
    def key(self) -> str:
        return f"{self.variant}/{self.subject}/{self.video}"

    @property
    def frame_numbers(self) -> list[int]:
        return sorted(self.frames)


def load_active_frames(path: Path) -> dict[str, tuple[int, int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Active-frame configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid active-frame JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Active-frame configuration must be an object: {path}")

    active_frames: dict[str, tuple[int, int]] = {}
    for key, window in raw.items():
        if not isinstance(key, str) or not isinstance(window, dict) or set(window) != {"onset", "offset"}:
            raise ValueError(f"Invalid active-frame entry for {key!r}")
        onset, offset = window["onset"], window["offset"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (onset, offset)) or onset > offset:
            raise ValueError(f"Invalid onset/offset for {key!r}: {window!r}")
        active_frames[key] = (onset, offset)
    return active_frames


def index_sequence(variant: str, subject_dir: Path, video_dir: Path) -> SequenceIndex:
    frames: dict[int, dict[str, Path]] = {}
    for path in video_dir.iterdir():
        match = ASSET_RE.match(path.name)
        if not match:
            continue
        frame = int(match["frame"])
        kind = {"vertices": "npy", "mesh": "obj", "illustration": "jpg"}[match["kind"]]
        assets = frames.setdefault(frame, {})
        if kind in assets:
            raise ValueError(f"Duplicate {kind} asset for frame {frame}: {assets[kind]}, {path}")
        assets[kind] = path
    return SequenceIndex(variant, subject_dir.name, video_dir.name, video_dir, frames)


def scan_sequences(data_root: Path, roots: Mapping[str, Path] | None = None) -> list[SequenceIndex]:
    sequences: list[SequenceIndex] = []
    roots = roots or {variant: data_root / directory_name for variant, directory_name in VARIANTS.items()}
    for variant, variant_root in roots.items():
        if not variant_root.is_dir():
            continue
        for subject_dir in sorted((path for path in variant_root.iterdir() if path.is_dir()), key=lambda path: path.name):
            for video_dir in sorted((path for path in subject_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
                sequences.append(index_sequence(variant, subject_dir, video_dir))
    return sequences


def load_vertices(path: Path) -> np.ndarray:
    vertices = np.load(path, allow_pickle=False)
    if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[0] != 3:
        raise ValueError(f"Expected numeric (3, N) array: {path}; got {getattr(vertices, 'shape', None)}")
    if not np.issubdtype(vertices.dtype, np.number):
        raise ValueError(f"Expected numeric (3, N) array: {path}; got {vertices.dtype}")
    if vertices.shape[1] == 0:
        raise ValueError(f"Expected at least one vertex: {path}")
    return np.ascontiguousarray(vertices.T)


def parse_obj(path: Path) -> tuple[int, int, list[str]]:
    vertex_count = 0
    face_count = 0
    errors: list[str] = []
    face_indices: list[tuple[int, int]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as obj_file:
            for line_number, line in enumerate(obj_file, 1):
                fields = line.split()
                if not fields:
                    continue
                if fields[0] == "v":
                    vertex_count += 1
                elif fields[0] == "f":
                    face_count += 1
                    if len(fields) < 4:
                        errors.append(f"{path}:{line_number}: face has fewer than 3 vertices")
                        continue
                    for token in fields[1:]:
                        try:
                            face_indices.append((line_number, int(token.split("/", 1)[0])))
                        except ValueError:
                            errors.append(f"{path}:{line_number}: invalid face index {token!r}")
    except OSError as exc:
        return 0, 0, [f"Cannot read OBJ {path}: {exc}"]
    for line_number, index in face_indices:
        if index == 0 or index > vertex_count or index < -vertex_count:
            errors.append(f"{path}:{line_number}: face index {index} outside 1..{vertex_count}")
    return vertex_count, face_count, errors


class FrameCache:
    """Three-frame LRU cache; only playback neighbors remain resident."""

    def __init__(self, sequence: SequenceIndex, vertex_count: int):
        self.sequence = sequence
        self.vertex_count = vertex_count
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame: int) -> np.ndarray:
        if frame in self._cache:
            self._cache.move_to_end(frame)
            return self._cache[frame]
        path = self.sequence.frames[frame].get("npy")
        if path is None:
            raise ValueError(f"Missing NPY asset for {self.sequence.key} frame {frame}")
        vertices = load_vertices(path)
        if len(vertices) != self.vertex_count:
            raise ValueError(
                f"Vertex count mismatch for {path}: expected {self.vertex_count}, got {len(vertices)}"
            )
        self._cache[frame] = vertices
        while len(self._cache) > 3:
            self._cache.popitem(last=False)
        return vertices

    def preload_neighbors(self, frame: int) -> None:
        frames = self.sequence.frame_numbers
        index = frames.index(frame)
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(frames):
                self.get(frames[neighbor_index])
