"""PyVista mesh runtime for MEVIEW sequences."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from meviewer.assets import SequenceIndex, VertexFrames


@lru_cache(maxsize=3)
def load_vertex_colors(path: Path, vertex_count: int) -> np.ndarray:
    with path.open(encoding="utf-8", errors="replace") as obj_file:
        values = np.fromstring(
            " ".join(line[2:] for line in obj_file if line.startswith("v ")),
            sep=" ",
            dtype=float,
        )
    if values.size != vertex_count * 6:
        raise ValueError(f"Expected {vertex_count} colored vertices in {path}")
    return np.clip(np.rint(values.reshape(vertex_count, 6)[:, 3:] * 255), 0, 255).astype(np.uint8)


def face_indices_are_valid(mesh: pv.PolyData) -> bool:
    faces = mesh.faces
    position = 0
    while position < len(faces):
        size = int(faces[position])
        if size < 3 or position + size >= len(faces):
            return False
        indices = faces[position + 1 : position + size + 1]
        if (
            len(indices) != size
            or indices.min(initial=0) < 0
            or indices.max(initial=-1) >= mesh.n_points
        ):
            return False
        position += size + 1
    return position == len(faces)


class MeshSequence:
    """One fixed topology plus a bounded cache of NPY vertex coordinates."""

    def __init__(self, index: SequenceIndex, mesh: pv.PolyData, vertex_frames: VertexFrames):
        self.index = index
        self.mesh = mesh
        self._vertex_frames = vertex_frames
        self.frames = index.frame_numbers
        self.current_frame = self.frames[0]
        self.reference_frame = self.frames[0]
        self._diagnostic_percentiles: dict[tuple[str, int | None], float | None] = {}

    @classmethod
    def load(cls, index: SequenceIndex) -> MeshSequence:
        if not index.frame_numbers:
            raise ValueError(f"No numbered mesh frames: {index.directory}")
        missing = [
            f"{frame:03d} ({', '.join(index.missing_assets(frame))})"
            for frame in index.frame_numbers
            if index.missing_assets(frame)
        ]
        if missing:
            raise ValueError(f"Missing paired assets in {index.key}: {'; '.join(missing)}")
        first_frame = index.frame_numbers[0]
        first_obj = index.asset_path(first_frame, "mesh")
        assert first_obj is not None
        try:
            mesh = pv.read(first_obj)
        except Exception as exc:
            raise ValueError(f"Cannot read OBJ topology {first_obj}: {exc}") from exc
        if (
            not isinstance(mesh, pv.PolyData)
            or mesh.n_points == 0
            or not face_indices_are_valid(mesh)
        ):
            raise ValueError(f"Invalid OBJ topology: {first_obj}")
        vertex_frames = index.vertex_frames(mesh.n_points)
        vertex_frames.get(first_frame)
        return cls(index, mesh, vertex_frames)

    def set_frame(self, frame: int) -> np.ndarray:
        if not self.index.has_frame(frame):
            raise ValueError(f"Frame {frame} not present in {self.index.key}")
        points = self._vertex_frames.get(frame)
        self.mesh.points[:] = points
        mesh_path = self.index.asset_path(frame, "mesh")
        assert mesh_path is not None
        colors = load_vertex_colors(mesh_path, self.mesh.n_points)
        if "vertex_colors" in self.mesh.point_data:
            self.mesh.point_data["vertex_colors"][:] = colors
        else:
            self.mesh.point_data["vertex_colors"] = colors
        self.mesh.GetPoints().Modified()
        self.mesh.GetPointData().GetArray("vertex_colors").Modified()
        self.mesh.Modified()
        self.current_frame = frame
        self._vertex_frames.preload_neighbors(frame)
        return points

    def set_reference(self, frame: int) -> None:
        if not self.index.has_frame(frame):
            raise ValueError(f"Reference frame {frame} not present in {self.index.key}")
        self._vertex_frames.get(frame)
        self.reference_frame = frame

    def reference_points(self, indices: Sequence[int]) -> np.ndarray:
        return self._vertex_frames.get(self.reference_frame)[list(indices)]

    def previous_frames(self) -> tuple[int | None, int | None]:
        current_index = self.frames.index(self.current_frame)
        previous = self.frames[current_index - 1] if current_index >= 1 else None
        previous_previous = self.frames[current_index - 2] if current_index >= 2 else None
        return previous, previous_previous

    def diagnostics(self) -> dict[str, np.ndarray | None]:
        current = self._vertex_frames.get(self.current_frame)
        reference = self._vertex_frames.get(self.reference_frame)
        previous, previous_previous = self.previous_frames()
        velocity = (
            None
            if previous is None
            else np.linalg.norm(current - self._vertex_frames.get(previous), axis=1)
        )
        acceleration = (
            None
            if previous is None or previous_previous is None
            else np.linalg.norm(
                current
                - 2 * self._vertex_frames.get(previous)
                + self._vertex_frames.get(previous_previous),
                axis=1,
            )
        )
        return {
            "displacement": np.linalg.norm(current - reference, axis=1),
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def diagnostic_percentile(self, layer: str, percentile: float = 0.99) -> float | None:
        """Return one stable whole-sequence percentile for a diagnostic layer."""
        if layer not in {"displacement", "velocity", "acceleration"}:
            raise ValueError(f"Unknown diagnostic layer: {layer}")
        reference = self.reference_frame if layer == "displacement" else None
        key = (layer, reference)
        if key in self._diagnostic_percentiles:
            return self._diagnostic_percentiles[key]
        reference_points = (
            self._vertex_frames.get(self.reference_frame) if layer == "displacement" else None
        )
        values: list[np.ndarray] = []
        for number, frame in enumerate(self.frames):
            current = self._vertex_frames.get(frame)
            if layer == "displacement":
                assert reference_points is not None
                values.append(np.linalg.norm(current - reference_points, axis=1))
            elif layer == "velocity" and number:
                values.append(
                    np.linalg.norm(
                        current - self._vertex_frames.get(self.frames[number - 1]), axis=1
                    )
                )
            elif layer == "acceleration" and number > 1:
                values.append(
                    np.linalg.norm(
                        current
                        - 2 * self._vertex_frames.get(self.frames[number - 1])
                        + self._vertex_frames.get(self.frames[number - 2]),
                        axis=1,
                    )
                )
        result = (
            float(np.quantile(np.concatenate(values), percentile, method="lower"))
            if values
            else None
        )
        self._diagnostic_percentiles[key] = result
        return result

    def pooling_bin(self, bin_number: int) -> tuple[np.ndarray, float]:
        current = self._vertex_frames.get(self.current_frame)
        reference = self._vertex_frames.get(self.reference_frame)
        motion = np.ascontiguousarray((current - reference).T * 100.0, dtype=np.float32)
        pooled = F.adaptive_max_pool1d(torch.from_numpy(motion).unsqueeze(0), 64)[
            0, :, bin_number
        ].numpy()
        start = (bin_number * len(current)) // 64
        end = ((bin_number + 1) * len(current) + 63) // 64
        return np.arange(start, min(end, len(current))), float(np.linalg.norm(pooled))
