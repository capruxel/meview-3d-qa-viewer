#!/usr/bin/env python3
"""Extract normalized MediaPipe landmarks from viewer illustration frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_data import scan_sequences


def load_config(path: Path | None) -> tuple[list[int], list[str], dict[str, list[int]]]:
    if path is None:
        return [], [], {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read MediaPipe config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MediaPipe config {path}: {exc}") from exc
    selected = config.get("landmarks20")
    rois = config.get("rois")
    if not isinstance(selected, list) or len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError("config.landmarks20 must contain exactly 20 unique indices")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 478 for index in selected):
        raise ValueError("config.landmarks20 indices must be integers in [0, 478)")
    if not isinstance(rois, dict) or len(rois) != 4 or any(not isinstance(name, str) or not name for name in rois):
        raise ValueError("config.rois must contain exactly four named regions")
    for name, indices in rois.items():
        if not isinstance(indices, list) or not indices or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 478 for index in indices
        ):
            raise ValueError(f"config.rois[{name!r}] must contain valid MediaPipe indices")
    return selected, list(rois), rois


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=("v2", "v3"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {args.model}")
    selected, roi_names, rois = load_config(args.config)
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("Install the viewer landmarks extra before running this command") from exc

    indices = scan_sequences(args.data_root)
    if args.variant:
        indices = [index for index in indices if index.variant == args.variant]
    if not indices:
        raise FileNotFoundError(f"No viewer sequences found under {args.data_root}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    count = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for sequence in indices:
            for frame in sequence.frame_numbers:
                image_path = sequence.frames[frame].get("jpg")
                if image_path is None or not image_path.is_file():
                    raise FileNotFoundError(f"Missing illustration for {sequence.key} frame {frame}: {image_path}")
                output_path = args.output_root / sequence.variant / sequence.subject / sequence.video / f"{frame:03d}.npz"
                if output_path.exists() and not args.overwrite:
                    continue
                image = mp.Image.create_from_file(str(image_path))
                result = landmarker.detect(image)
                if result.face_landmarks:
                    points = np.asarray([[point.x, point.y, point.z] for point in result.face_landmarks[0]], dtype=np.float32)
                    if points.shape != (478, 3):
                        raise ValueError(f"MediaPipe returned {points.shape} for {image_path}; expected (478, 3)")
                    selected_points = points[selected] if selected else np.empty((0, 3), dtype=np.float32)
                    centers = np.asarray([points[indices].mean(axis=0) for indices in rois.values()], dtype=np.float32) if rois else np.empty((0, 3), dtype=np.float32)
                    status = "ok"
                else:
                    points = np.zeros((478, 3), dtype=np.float32)
                    selected_points = np.zeros((20, 3), dtype=np.float32) if selected else np.empty((0, 3), dtype=np.float32)
                    centers = np.zeros((4, 3), dtype=np.float32) if rois else np.empty((0, 3), dtype=np.float32)
                    status = "no_face"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_path,
                    landmarks478=points,
                    landmarks20=selected_points,
                    roi_centers=centers,
                    roi_names=np.asarray(roi_names),
                    status=np.asarray(status),
                )
                count += 1
    print(f"Extracted MediaPipe records: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
