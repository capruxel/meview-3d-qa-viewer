"""Interface checks for non-Qt viewer session transitions."""

import json
from pathlib import Path

import numpy as np

from meviewer.viewer.session import ViewerSession


def make_session(tmp_path: Path, mediapipe_root: Path | None = None) -> ViewerSession:
    mesh_root = tmp_path / "mesh"
    sequence_directory = mesh_root / "sub01" / "01"
    sequence_directory.mkdir(parents=True)
    for frame in (1, 2):
        stem = sequence_directory / f"{frame:03d}_frontal_"
        np.save(stem.with_name(f"{stem.name}vertices.npy"), np.eye(3) * frame)
        stem.with_name(f"{stem.name}mesh.obj").write_text(
            "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n",
            encoding="utf-8",
        )
        stem.with_name(f"{stem.name}illustration.jpg").write_bytes(b"jpg")
    active_frames = tmp_path / "active.json"
    active_frames.write_text(json.dumps({"sub01_01": {"onset": 2, "offset": 2}}))
    return ViewerSession(tmp_path, mesh_root, None, active_frames, None, None, mediapipe_root)


def test_selection_and_transitions_are_snapshots(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    index = next(iter(session.indices.values()))
    snapshot = session.select(index.key)

    assert snapshot.sequence is not None
    assert snapshot.sequence.reference_frame == 2
    assert snapshot.sequence.current_frame == 1
    assert snapshot.mediapipe_image == tmp_path / "mesh/sub01/01/001_frontal_illustration.jpg"
    assert snapshot.raw_video_error == "Raw video root not configured"
    assert snapshot.mediapipe_error is None
    assert session.set_frame(2).sequence.current_frame == 2
    assert session.set_reference(1).sequence.reference_frame == 1


def test_invalid_mediapipe_record_is_nonfatal(tmp_path: Path) -> None:
    mediapipe_root = tmp_path / "mediapipe"
    record = mediapipe_root / "lfann-v3" / "sub01" / "01" / "001.npz"
    record.parent.mkdir(parents=True)
    np.savez(record, invalid=np.asarray(1))

    snapshot = make_session(tmp_path, mediapipe_root).select("lfann-v3/sub01/01")

    assert snapshot.sequence is not None
    assert snapshot.mediapipe_error is not None
