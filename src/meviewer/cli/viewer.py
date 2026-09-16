"""Local PySide6/PyVista QA viewer for MEVIEW dense-mesh sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from meviewer.assets import SequenceIndex, load_active_frames, scan_sequences
from meviewer.config import add_config_argument, apply_config_defaults
from meviewer.manifest import load as load_viewer_manifest
from meviewer.viewer.mesh import MeshSequence
from meviewer.viewer.session import probe_raw_video
from meviewer.viewer.window import MeshViewer

MOTION_CLIM = (0.0, 1.0)  # Fixed QA legend; this is not the 3456-D feature scale.


def find_sequence(data_root: Path, variant: str, subject: str, video: str) -> SequenceIndex:
    key = f"{variant}/{subject}/{video}"
    for sequence in scan_sequences(data_root):
        if sequence.key == key:
            return sequence
    raise ValueError(f"Sequence not found: {key}")


def self_test(data_root: Path, active_frames_path: Path) -> int:
    active_frames = load_active_frames(active_frames_path)
    v3 = MeshSequence.load(find_sequence(data_root, "v3", "sub01", "01"))
    assert v3.mesh.n_points == 35709
    assert active_frames["sub01_01"] == (48, 63)
    first, second, last = v3.frames[0], v3.frames[1], v3.frames[-1]
    v3.set_frame(first)
    assert v3.mesh.point_data["vertex_colors"].shape == (35709, 3)
    assert v3.mesh.point_data["vertex_colors"].dtype == np.uint8
    assert v3.diagnostics()["velocity"] is None and v3.diagnostics()["acceleration"] is None
    first_points = v3.mesh.points.copy()
    v3.set_frame(second)
    assert not np.array_equal(first_points, v3.mesh.points)
    v3.set_frame(last)
    assert v3.mesh.n_points == 35709
    raw_v3 = probe_raw_video(data_root / "raw" / "me-cuts", v3.index)
    assert raw_v3.frame_count == len(v3.frames) == 89 and raw_v3.position_ms(1) == 0
    v2 = MeshSequence.load(find_sequence(data_root, "v2", "sub11", "03"))
    v2.set_frame(v2.frames[-1])
    raw_v2 = probe_raw_video(data_root / "raw" / "me-cuts", v2.index)
    assert len(v2.frames) == 69 < raw_v2.frame_count == 72
    print("viewer self-test passed: colored meshes and raw-video frame locks validated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mesh-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--active-frames", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--landmarks-root", type=Path)
    parser.add_argument("--mediapipe-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    apply_config_defaults(parser, "viewer")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root is required")
    manifest = load_viewer_manifest(args.manifest, args.data_root) if args.manifest else {}
    mesh_root = args.mesh_root or manifest.get("mesh_root") or args.data_root
    active_frames = args.active_frames or manifest.get("active_frames")
    if active_frames is None:
        parser.error("--active-frames or --manifest is required")
    if args.self_test:
        return self_test(args.data_root, active_frames)
    app = QApplication(sys.argv)
    window = MeshViewer(
        args.data_root,
        mesh_root,
        args.raw_root,
        active_frames,
        args.results_dir,
        args.landmarks_root,
        args.mediapipe_root,
    )
    window.resize(1600, 950)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
