"""Interface checks for indexed MEVIEW mesh assets."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from meviewer.assets import scan_sequences


def write_frame(
    sequence_directory: Path, frame: int, *, mesh: bool = True, illustration: bool = True
) -> None:
    stem = sequence_directory / f"{frame:03d}_frontal_"
    np.save(stem.with_name(f"{stem.name}vertices.npy"), np.arange(9).reshape(3, 3) + frame)
    if mesh:
        stem.with_name(f"{stem.name}mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
    if illustration:
        stem.with_name(f"{stem.name}illustration.jpg").write_bytes(b"jpg")


def test_indexed_assets_are_typed_and_load_vertices(tmp_path: Path) -> None:
    sequence_directory = tmp_path / "sub01" / "01"
    sequence_directory.mkdir(parents=True)
    for frame in range(1, 5):
        write_frame(sequence_directory, frame)
    write_frame(sequence_directory, 5, mesh=False, illustration=False)

    index = scan_sequences(tmp_path, {"v3": tmp_path})[0]

    assert index.frame_numbers == (1, 2, 3, 4, 5)
    assert index.missing_assets(5) == ("obj", "jpg")
    assets = index.assets_for(1)
    assert assets.vertices == sequence_directory / "001_frontal_vertices.npy"
    with pytest.raises(FrozenInstanceError):
        assets.mesh = None  # type: ignore[misc]

    frames = index.vertex_frames(3)
    np.testing.assert_array_equal(frames.get(1), np.arange(9).reshape(3, 3).T + 1)
    for frame in range(1, 5):
        np.testing.assert_array_equal(frames.get(frame), np.arange(9).reshape(3, 3).T + frame)
