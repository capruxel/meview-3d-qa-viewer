"""Interface checks for indexed MEVIEW mesh assets."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from mesh_data import scan_sequences


class MeshDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.sequence_directory = self.root / "sub01" / "01"
        self.sequence_directory.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_frame(self, frame: int, *, mesh: bool = True, illustration: bool = True) -> None:
        stem = self.sequence_directory / f"{frame:03d}_frontal_"
        np.save(stem.with_name(f"{stem.name}vertices.npy"), np.arange(9).reshape(3, 3) + frame)
        if mesh:
            stem.with_name(f"{stem.name}mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
        if illustration:
            stem.with_name(f"{stem.name}illustration.jpg").write_bytes(b"jpg")

    def test_indexed_assets_are_typed_and_load_vertices(self) -> None:
        for frame in range(1, 5):
            self.write_frame(frame)
        self.write_frame(5, mesh=False, illustration=False)

        index = scan_sequences(self.root, {"v3": self.root})[0]

        self.assertEqual(index.frame_numbers, (1, 2, 3, 4, 5))
        self.assertEqual(index.missing_assets(5), ("obj", "jpg"))
        assets = index.assets_for(1)
        self.assertEqual(assets.vertices, self.sequence_directory / "001_frontal_vertices.npy")
        with self.assertRaises(FrozenInstanceError):
            assets.mesh = None  # type: ignore[misc]

        frames = index.vertex_frames(3)
        np.testing.assert_array_equal(frames.get(1), np.arange(9).reshape(3, 3).T + 1)
        for frame in range(1, 5):
            np.testing.assert_array_equal(frames.get(frame), np.arange(9).reshape(3, 3).T + frame)


if __name__ == "__main__":
    unittest.main()
