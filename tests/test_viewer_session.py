"""Interface checks for non-Qt viewer session transitions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from meviewer.viewer.session import ViewerSession


class ViewerSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.mesh_root = self.root / "mesh"
        self.sequence_directory = self.mesh_root / "sub01" / "01"
        self.sequence_directory.mkdir(parents=True)
        for frame in (1, 2):
            stem = self.sequence_directory / f"{frame:03d}_frontal_"
            np.save(stem.with_name(f"{stem.name}vertices.npy"), np.eye(3) * frame)
            stem.with_name(f"{stem.name}mesh.obj").write_text(
                "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n",
                encoding="utf-8",
            )
            stem.with_name(f"{stem.name}illustration.jpg").write_bytes(b"jpg")
        self.active_frames = self.root / "active.json"
        self.active_frames.write_text(json.dumps({"sub01_01": {"onset": 2, "offset": 2}}))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def session(self, mediapipe_root: Path | None = None) -> ViewerSession:
        return ViewerSession(
            self.root,
            self.mesh_root,
            None,
            self.active_frames,
            None,
            None,
            mediapipe_root,
        )

    def test_selection_and_transitions_are_snapshots(self) -> None:
        session = self.session()
        index = next(iter(session.indices.values()))

        snapshot = session.select(index.key)

        self.assertIsNotNone(snapshot.sequence)
        self.assertEqual(snapshot.sequence.reference_frame, 2)
        self.assertEqual(snapshot.sequence.current_frame, 1)
        self.assertEqual(
            snapshot.mediapipe_image, self.sequence_directory / "001_frontal_illustration.jpg"
        )
        self.assertEqual(snapshot.raw_video_error, "Raw video root not configured")
        self.assertIsNone(snapshot.mediapipe_error)
        next_snapshot = session.set_frame(2)
        self.assertEqual(next_snapshot.sequence.current_frame, 2)
        reference_snapshot = session.set_reference(1)
        self.assertEqual(reference_snapshot.sequence.reference_frame, 1)

    def test_invalid_mediapipe_record_is_nonfatal(self) -> None:
        mediapipe_root = self.root / "mediapipe"
        record = mediapipe_root / "lfann-v3" / "sub01" / "01" / "001.npz"
        record.parent.mkdir(parents=True)
        np.savez(record, invalid=np.asarray(1))

        snapshot = self.session(mediapipe_root).select("lfann-v3/sub01/01")

        self.assertIsNotNone(snapshot.sequence)
        self.assertIsNotNone(snapshot.mediapipe_error)


if __name__ == "__main__":
    unittest.main()
