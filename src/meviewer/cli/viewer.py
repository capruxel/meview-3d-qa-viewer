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


def find_sequence(mesh_root: Path, variant: str, subject: str, video: str) -> SequenceIndex:
    key = f"{variant}/{subject}/{video}"
    for sequence in scan_sequences(mesh_root):
        if sequence.key == key:
            return sequence
    raise ValueError(f"Sequence not found: {key}")


def self_test(
    data_root: Path, mesh_root: Path, raw_root: Path | None, active_frames_path: Path
) -> int:
    active_frames = load_active_frames(active_frames_path)
    v3 = MeshSequence.load(find_sequence(mesh_root, "v3", "sub01", "01"))
    if v3.mesh.n_points != 35709:
        raise ValueError("v3 mesh must contain 35709 points")
    if active_frames.get("sub01_01") != (48, 63):
        raise ValueError("sub01_01 active window must be (48, 63)")
    first, second, last = v3.frames[0], v3.frames[1], v3.frames[-1]
    v3.set_frame(first)
    if v3.mesh.point_data["vertex_colors"].shape != (35709, 3):
        raise ValueError("v3 vertex colors must have shape (35709, 3)")
    if v3.mesh.point_data["vertex_colors"].dtype != np.uint8:
        raise ValueError("v3 vertex colors must be uint8")
    if v3.diagnostics()["velocity"] is not None or v3.diagnostics()["acceleration"] is not None:
        raise ValueError("first v3 frame must have no velocity or acceleration")
    first_points = v3.mesh.points.copy()
    v3.set_frame(second)
    if np.array_equal(first_points, v3.mesh.points):
        raise ValueError("v3 mesh points must change between first two frames")
    v3.set_frame(last)
    if v3.mesh.n_points != 35709:
        raise ValueError("last v3 frame must contain 35709 points")
    resolved_raw_root = raw_root or data_root / "raw" / "me-cuts"
    raw_v3 = probe_raw_video(resolved_raw_root, v3.index)
    if raw_v3.frame_count != len(v3.frames) or len(v3.frames) != 89 or raw_v3.position_ms(1) != 0:
        raise ValueError("v3 raw video must lock to its 89 mesh frames")
    v2 = MeshSequence.load(find_sequence(mesh_root, "v2", "sub11", "03"))
    v2.set_frame(v2.frames[-1])
    raw_v2 = probe_raw_video(resolved_raw_root, v2.index)
    if not (len(v2.frames) == 69 < raw_v2.frame_count == 72):
        raise ValueError("v2 raw video must retain its three-frame tail")
    print("viewer self-test passed: colored meshes and raw-video frame locks validated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect mesh sequences and regional-motion evidence.",
        epilog="Direct arguments override [viewer] TOML defaults. --self-test does not open Qt.",
    )
    add_config_argument(parser)
    parser.add_argument("--data-root", type=Path, help="Required dataset root.")
    parser.add_argument("--manifest", type=Path, help="Optional TOML mesh/active-frame manifest.")
    parser.add_argument("--mesh-root", type=Path, help="Override manifest mesh_root.")
    parser.add_argument("--raw-root", type=Path, help="Optional raw-video root.")
    parser.add_argument("--active-frames", type=Path, help="Override manifest active_frames.")
    parser.add_argument("--results-dir", type=Path, help="Optional prediction results directory.")
    parser.add_argument("--landmarks-root", type=Path, help="Optional landmark mapping directory.")
    parser.add_argument("--mediapipe-root", type=Path, help="Optional MediaPipe frame-record root.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the reference dataset without opening Qt.",
    )
    apply_config_defaults(parser, "viewer")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root is required")
    try:
        manifest = load_viewer_manifest(args.manifest, args.data_root) if args.manifest else {}
    except ValueError as exc:
        parser.error(str(exc))
    mesh_root = args.mesh_root or manifest.get("mesh_root") or args.data_root
    active_frames = args.active_frames or manifest.get("active_frames")
    if active_frames is None:
        parser.error("--active-frames or --manifest is required")
    if args.self_test:
        try:
            return self_test(args.data_root, mesh_root, args.raw_root, active_frames)
        except ValueError as exc:
            print(f"viewer self-test failed: {exc}", file=sys.stderr)
            return 1
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
