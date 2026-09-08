"""Validated loading of offline MediaPipe frame records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mesh_data import SequenceIndex


@dataclass(frozen=True)
class MediaPipeFrame:
    landmarks478: np.ndarray
    landmarks20: np.ndarray
    roi_centers: np.ndarray
    roi_names: tuple[str, ...]
    status: str


def mediapipe_frame_path(root: Path, sequence: SequenceIndex, frame: int) -> Path:
    return root / sequence.variant / sequence.subject / sequence.video / f"{frame:03d}.npz"


def _array(values: Any, name: str, path: Path) -> np.ndarray:
    try:
        return np.asarray(values[name])
    except KeyError as exc:
        raise ValueError(f"Missing MediaPipe array {name!r}: {path}") from exc


def _points(values: np.ndarray, shape: tuple[int, int], name: str, path: Path) -> np.ndarray:
    if values.shape != shape or values.dtype.kind not in "fiu":
        raise ValueError(f"Invalid {name} shape/dtype in {path}: {values.shape}/{values.dtype}")
    return np.asarray(values, dtype=np.float32)


def load_mediapipe_frame(path: Path) -> MediaPipeFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as values:
            raw_landmarks478 = _array(values, "landmarks478", path)
            raw_landmarks20 = _array(values, "landmarks20", path)
            raw_roi_centers = _array(values, "roi_centers", path)
            raw_names = _array(values, "roi_names", path)
            raw_status = _array(values, "status", path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid MediaPipe record {path}: {exc}") from exc

    if raw_names.ndim != 1 or raw_names.dtype.kind not in "OUSU" or len(raw_names) not in (0, 4):
        raise ValueError(f"Invalid roi_names shape/dtype in {path}: {raw_names.shape}/{raw_names.dtype}")
    if raw_status.shape != () or raw_status.dtype.kind not in "OUSU":
        raise ValueError(f"Invalid status shape/dtype in {path}: {raw_status.shape}/{raw_status.dtype}")
    status = str(raw_status.item())
    if status not in {"ok", "no_face"}:
        raise ValueError(f"Invalid MediaPipe status in {path}: {status!r}")
    landmarks478 = _points(raw_landmarks478, (478, 3), "landmarks478", path)
    selected_shape = (20, 3) if len(raw_names) else (0, 3)
    centers_shape = (4, 3) if len(raw_names) else (0, 3)
    landmarks20 = _points(raw_landmarks20, selected_shape, "landmarks20", path)
    roi_centers = _points(raw_roi_centers, centers_shape, "roi_centers", path)
    roi_names = tuple(str(name) for name in raw_names.tolist())
    if len(set(roi_names)) != len(roi_names) or any(not name for name in roi_names):
        raise ValueError(f"Invalid roi_names in {path}: {roi_names!r}")
    if status == "ok" and not all(np.isfinite(points).all() for points in (landmarks478, landmarks20, roi_centers)):
        raise ValueError(f"Non-finite detected landmarks in {path}")
    return MediaPipeFrame(landmarks478, landmarks20, roi_centers, roi_names, status)
