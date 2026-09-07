#!/usr/bin/env python3
"""Audit MEVIEW V2/V3 mesh assets without repairing missing frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mesh_data import VARIANTS, SequenceIndex, load_active_frames, load_vertices, parse_obj, scan_sequences

EXPECTED = {
    "v2": {"frame_count": 2009, "vertex_count": 38365},
    "v3": {"frame_count": 2012, "vertex_count": 35709},
}
REQUIRED_ASSETS = ("npy", "obj", "jpg")


def bounds_payload(minimum: np.ndarray | None, maximum: np.ndarray | None) -> dict[str, list[float] | None]:
    return {
        "min": minimum.tolist() if minimum is not None else None,
        "max": maximum.tolist() if maximum is not None else None,
    }


def audit_sequence(sequence: SequenceIndex, active_frames: dict[str, tuple[int, int]], landmarks_root: Path | None) -> dict[str, Any]:
    missing_assets: list[str] = []
    errors: list[str] = []
    vertex_count: int | None = None
    coordinate_min: np.ndarray | None = None
    coordinate_max: np.ndarray | None = None
    frame_numbers = sequence.frame_numbers
    active_frame_missing: list[int] = []

    asset_names = {"npy": "vertices.npy", "obj": "mesh.obj", "jpg": "illustration.jpg"}
    for frame in frame_numbers:
        assets = sequence.frames[frame]
        for asset in REQUIRED_ASSETS:
            if asset not in assets:
                missing_assets.append(
                    f"frame {frame:03d}: missing {asset}: "
                    f"{sequence.directory / f'{frame:03d}_frontal_{asset_names[asset]}'}"
                )
    sequence_key = f"{sequence.subject}_{sequence.video}"

    if sequence_key in active_frames:
        onset, offset = active_frames[sequence_key]
        active_frame_missing.extend(frame for frame in range(onset, offset + 1) if "npy" not in sequence.frames.get(frame, {}))

    for frame in frame_numbers:
        npy_path = sequence.frames[frame].get("npy")
        if npy_path is None:
            continue
        try:
            vertices = load_vertices(npy_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if not np.isfinite(vertices).all():
            errors.append(f"Non-finite coordinate values: {npy_path}")
            continue
        if vertex_count is None:
            vertex_count = len(vertices)
        elif len(vertices) != vertex_count:
            errors.append(f"Vertex count mismatch: {npy_path}: expected {vertex_count}, got {len(vertices)}")
        frame_min = vertices.min(axis=0)
        frame_max = vertices.max(axis=0)
        coordinate_min = frame_min if coordinate_min is None else np.minimum(coordinate_min, frame_min)
        coordinate_max = frame_max if coordinate_max is None else np.maximum(coordinate_max, frame_max)

    face_count: int | None = None
    obj_paths = [sequence.frames[frame]["obj"] for frame in frame_numbers if "obj" in sequence.frames[frame]]
    if obj_paths:
        obj_vertex_count, face_count, topology_errors = parse_obj(obj_paths[0])
        errors.extend(topology_errors)
        if vertex_count is not None and obj_vertex_count != vertex_count:
            errors.append(f"OBJ/NPY vertex mismatch: {obj_paths[0]}: OBJ={obj_vertex_count}, NPY={vertex_count}")
    else:
        errors.append(f"No OBJ mesh available: {sequence.directory}")

    landmark_candidates = []
    if landmarks_root is not None:
        landmark_candidates.append(landmarks_root / f"{sequence.variant}.json")
    landmark_candidates.extend(
        path for path in sequence.directory.iterdir() if any(token in path.name.lower() for token in ("landmark", "keypoint"))
    )
    landmark_sidecar = next((str(path) for path in landmark_candidates if path.is_file()), None)
    valid = not missing_assets and not active_frame_missing and not errors
    return {
        "variant": sequence.variant,
        "subject": sequence.subject,
        "video": sequence.video,
        "key": sequence.key,
        "frame_count": len(frame_numbers),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "missing_assets": missing_assets,
        "active_frame_missing": active_frame_missing,
        "coordinate_bounds": bounds_payload(coordinate_min, coordinate_max),
        "landmark_sidecar": landmark_sidecar,
        "errors": errors,
        "valid": valid,
    }


def summarize(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    variants: dict[str, dict[str, Any]] = {}
    baseline_errors: list[str] = []
    for variant in VARIANTS:
        items = [record for record in records if record["variant"] == variant]
        frame_count = sum(item["frame_count"] for item in items)
        vertex_counts = sorted({item["vertex_count"] for item in items if item["vertex_count"] is not None})
        variants[variant] = {
            "sequence_count": len(items),
            "frame_count": frame_count,
            "vertex_counts": vertex_counts,
            "valid_sequence_count": sum(item["valid"] for item in items),
            "invalid_sequence_count": sum(not item["valid"] for item in items),
            "active_frame_missing_count": sum(len(item["active_frame_missing"]) for item in items),
        }
        expected = EXPECTED[variant]
        if frame_count != expected["frame_count"]:
            baseline_errors.append(f"{variant} frame count: expected {expected['frame_count']}, got {frame_count}")
        if vertex_counts != [expected["vertex_count"]]:
            baseline_errors.append(f"{variant} vertex counts: expected [{expected['vertex_count']}], got {vertex_counts}")
    return {"variants": variants, "sequence_count": len(records), "valid": not baseline_errors and all(item["valid"] for item in records)}, baseline_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--active-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks-root", type=Path)
    args = parser.parse_args()

    try:
        active_frames = load_active_frames(args.active_frames)
        records = [
            audit_sequence(sequence, active_frames, args.landmarks_root)
            for sequence in scan_sequences(args.data_root)
        ]
    except (OSError, ValueError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2

    summary, baseline_errors = summarize(records)
    payload = {"summary": summary, "baseline_errors": baseline_errors, "sequences": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {summary['sequence_count']} sequences, valid={summary['valid']}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
