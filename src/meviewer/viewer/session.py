"""Session transitions for the MEVIEW mesh viewer."""

import csv
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from meviewer.annotations import (
    AnnotationDocument,
    RegionalMotionAnnotation,
    load_annotation_document,
    save_annotation_document,
)
from meviewer.assets import SequenceIndex, load_active_frames, scan_sequences
from meviewer.mediapipe import MediaPipeFrame, load_mediapipe_frame, mediapipe_frame_path
from meviewer.viewer.mesh import MeshSequence

LABELS = {0: "Positive", 1: "Negative", 2: "Surprise"}
_EMPTY_MAPPING: Mapping[str, str] = MappingProxyType({})


def _mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class RawVideo:
    path: Path
    frame_count: int
    fps: float

    def position_ms(self, mesh_frame: int) -> int:
        return round((mesh_frame - 1) * 1000 / self.fps)


def probe_raw_video(raw_root: Path | None, index: SequenceIndex) -> RawVideo:
    if raw_root is None:
        raise ValueError("Raw video root not configured")
    path = raw_root / "cuts" / f"{index.subject}-{int(index.video)}.mp4"
    if not path.is_file():
        raise ValueError(f"Raw video missing: {path}")
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect raw video {path}: {result.stderr.strip()}")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/", 1))
        frame_count, fps = int(stream["nb_read_frames"]), numerator / denominator
    except (
        IndexError,
        KeyError,
        ValueError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"Invalid raw-video metadata: {path}") from exc
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Invalid raw-video timing: {path}")
    return RawVideo(path, frame_count, fps)


@dataclass(frozen=True)
class LandmarkMapping:
    variant: str
    vertex_count: int
    landmarks: dict[str, int]


def load_landmarks(root: Path | None, variant: str, vertex_count: int) -> LandmarkMapping | None:
    if root is None:
        return None
    path = root / f"{variant}.json"
    if not path.is_file():
        return None

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate landmark mapping key {key!r}: {path}")
            result[key] = value
        return result

    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid landmark mapping {path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"variant", "vertex_count", "landmarks"}:
        raise ValueError(f"Invalid landmark mapping schema: {path}")
    if raw["variant"] != variant or raw["vertex_count"] != vertex_count:
        raise ValueError(
            f"Landmark mapping mismatch: {path}: expected {variant}/{vertex_count}, "
            f"got {raw['variant']!r}/{raw['vertex_count']!r}"
        )
    landmarks = raw["landmarks"]
    if not isinstance(landmarks, dict) or not landmarks:
        raise ValueError(f"Landmark mapping must contain landmarks: {path}")
    for name, index in landmarks.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(index, int)
            or isinstance(index, bool)
        ):
            raise ValueError(f"Invalid landmark entry {name!r}: {index!r}: {path}")
        if not 0 <= index < vertex_count:
            raise ValueError(f"Landmark index out of bounds for {name!r}: {index}: {path}")
    return LandmarkMapping(variant, vertex_count, landmarks)


def sequence_metadata(data_root: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for variant in ("v2", "v3"):
        try:
            groups = np.load(data_root / f"groups_{variant}.npy", allow_pickle=False)
            videos = np.load(data_root / f"video_ids_{variant}.npy", allow_pickle=False)
            labels = np.load(data_root / f"labels_{variant}.npy", allow_pickle=False)
        except OSError:
            continue
        if not (len(groups) == len(videos) == len(labels)):
            continue
        for subject, video, label in zip(groups, videos, labels, strict=True):
            key = f"{variant}/{subject}/{str(video).zfill(2)}"
            metadata[key] = {
                "label": LABELS.get(int(label), str(label)),
                "subject": str(subject),
                "video": str(video).zfill(2),
            }
    return metadata


def prediction_metadata(
    data_root: Path, results_dir: Path | None
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
    predictions: dict[str, dict[str, str]] = {}
    metrics: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    if results_dir is None:
        return predictions, metrics, warnings
    metrics_path = results_dir / "metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as file:
            metrics = {row["method"].lower(): row for row in csv.DictReader(file)}
    predictions_path = results_dir / "predictions.csv"
    if not predictions_path.is_file():
        return predictions, metrics, warnings
    with predictions_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for variant in ("v2", "v3"):
        try:
            groups = np.load(data_root / f"groups_{variant}.npy", allow_pickle=False)
            videos = np.load(data_root / f"video_ids_{variant}.npy", allow_pickle=False)
            labels = np.load(data_root / f"labels_{variant}.npy", allow_pickle=False)
        except OSError as exc:
            warnings.append(f"Prediction mapping unavailable for {variant}: {exc}")
            continue
        order = [index for subject in np.unique(groups) for index in np.where(groups == subject)[0]]
        variant_rows = sorted(
            (row for row in rows if row.get("method", "").lower() == variant),
            key=lambda row: int(row["sample_order"]),
        )
        if len(variant_rows) != len(order):
            warnings.append(
                f"Prediction mapping unavailable for {variant}: expected {len(order)} rows, got {len(variant_rows)}"
            )
            continue
        for row, sample_index in zip(variant_rows, order, strict=True):
            subject, video, label = (
                str(groups[sample_index]),
                str(videos[sample_index]).zfill(2),
                LABELS.get(int(labels[sample_index]), str(labels[sample_index])),
            )
            if row.get("video_id", "").zfill(2) != video or row.get("true_label") != label:
                warnings.append(
                    f"Prediction mismatch for {variant}/{subject}/{video}; row {row.get('sample_order')} ignored"
                )
                continue
            predictions[f"{variant}/{subject}/{video}"] = row
    return predictions, metrics, warnings


@dataclass(frozen=True)
class ViewerSessionSnapshot:
    index: SequenceIndex | None = None
    sequence: MeshSequence | None = None
    sequence_error: str | None = None
    raw_video: RawVideo | None = None
    raw_video_error: str | None = None
    landmark_mapping: LandmarkMapping | None = None
    landmark_error: str | None = None
    active_frames: tuple[int, int] | None = None
    metadata: Mapping[str, str] = _EMPTY_MAPPING
    prediction: Mapping[str, str] | None = None
    metrics: Mapping[str, str] = _EMPTY_MAPPING
    prediction_warnings: tuple[str, ...] = ()
    mediapipe_image: Path | None = None
    mediapipe_record: MediaPipeFrame | None = None
    mediapipe_error: str | None = None
    annotations: tuple[RegionalMotionAnnotation, ...] = ()
    chart_data: Mapping[str, tuple[tuple[float | None, ...], ...]] = _EMPTY_MAPPING
    chart_status: str | None = None


class ViewerSession:
    def __init__(
        self,
        data_root: Path,
        mesh_root: Path,
        raw_root: Path | None,
        active_frames_path: Path,
        results_dir: Path | None,
        landmarks_root: Path | None,
        mediapipe_root: Path | None,
        annotation_output: Path | None = None,
    ) -> None:
        sequences = scan_sequences(mesh_root)
        if not sequences:
            sequences = scan_sequences(mesh_root, {"lfann-v3": mesh_root})
        self.indices = MappingProxyType({sequence.key: sequence for sequence in sequences})
        self._annotation_output = annotation_output
        self._annotations = (
            load_annotation_document(annotation_output, self.indices)
            if annotation_output is not None
            else AnnotationDocument()
        )
        self._raw_root = raw_root
        self._active_frames = load_active_frames(active_frames_path)
        self._metadata = sequence_metadata(data_root)
        self._predictions, self._metrics, self._prediction_warnings = prediction_metadata(
            data_root, results_dir
        )
        self._landmarks_root = landmarks_root
        self._mediapipe_root = mediapipe_root
        self._snapshot = ViewerSessionSnapshot()

    @property
    def snapshot(self) -> ViewerSessionSnapshot:
        return self._snapshot

    def active_window(self, index: SequenceIndex) -> tuple[int, int] | None:
        return self._active_frames.get(f"{index.subject}_{index.video}")

    def select(self, key: str) -> ViewerSessionSnapshot:
        try:
            index = self.indices[key]
        except KeyError as exc:
            raise ValueError(f"Sequence not found: {key}") from exc
        active = self.active_window(index)
        metadata = _mapping(self._metadata.get(key, {}))
        prediction = self._predictions.get(key)
        snapshot = ViewerSessionSnapshot(
            index=index,
            active_frames=active,
            metadata=metadata,
            prediction=None if prediction is None else _mapping(prediction),
            metrics=_mapping(self._metrics.get(index.variant, {})),
            prediction_warnings=tuple(self._prediction_warnings),
            annotations=tuple(
                item for item in self._annotations.annotations if item.sequence == key
            ),
        )
        try:
            sequence = MeshSequence.load(index)
        except ValueError as exc:
            self._snapshot = replace(snapshot, sequence_error=str(exc))
            return self._snapshot
        if active and index.has_frame(active[0]):
            sequence.set_reference(active[0])
        sequence.set_frame(sequence.frames[0])
        raw_video: RawVideo | None = None
        raw_video_error: str | None = None
        try:
            raw_video = probe_raw_video(self._raw_root, index)
        except ValueError as exc:
            raw_video_error = str(exc)
        landmark_mapping: LandmarkMapping | None = None
        landmark_error: str | None = None
        try:
            landmark_mapping = load_landmarks(
                self._landmarks_root, index.variant, sequence.mesh.n_points
            )
        except ValueError as exc:
            landmark_error = str(exc)
        self._snapshot = replace(
            snapshot,
            sequence=sequence,
            raw_video=raw_video,
            raw_video_error=raw_video_error,
            landmark_mapping=landmark_mapping,
            landmark_error=landmark_error,
            mediapipe_image=index.asset_path(sequence.current_frame, "illustration"),
        )
        chart_data, chart_status = self._build_chart_data(index, sequence, active)
        self._snapshot = replace(self._snapshot, chart_data=chart_data, chart_status=chart_status)
        return self._update_mediapipe()

    def _build_chart_data(
        self, index: SequenceIndex, sequence: MeshSequence, active: tuple[int, int] | None
    ) -> tuple[Mapping[str, tuple[tuple[float | None, ...], ...]], str | None]:
        names = ("left_brow", "right_brow", "left_mouth_corner", "right_mouth_corner")
        centers: dict[str, list[np.ndarray | None]] = {name: [] for name in names}
        candidate_records = 0
        validation_errors: list[str] = []
        usable_frames = 0
        for frame in sequence.frames:
            try:
                record = (
                    load_mediapipe_frame(mediapipe_frame_path(self._mediapipe_root, index, frame))
                    if self._mediapipe_root is not None
                    else None
                )
            except FileNotFoundError:
                record = None
            except ValueError as exc:
                record = None
                candidate_records += 1
                validation_errors.append(str(exc))
            else:
                candidate_records += record is not None
            if record is None or record.status == "no_face" or set(record.roi_names) != set(names):
                for values in centers.values():
                    values.append(None)
                continue
            positions = dict(zip(record.roi_names, record.roi_centers, strict=True))
            frame_usable = False
            for name in names:
                point = positions[name]
                usable = point if np.isfinite(point).all() else None
                centers[name].append(usable)
                frame_usable |= usable is not None
            usable_frames += frame_usable
        result: dict[str, tuple[tuple[float | None, ...], ...]] = {}
        for name, values in centers.items():
            baseline = (
                values[sequence.frames.index(active[0])]
                if active and active[0] in sequence.frames
                else None
            )
            displacement = [
                None if point is None or baseline is None else float(baseline[1] - point[1])
                for point in values
            ]
            velocity = [
                None
                if i < 1 or values[i] is None or values[i - 1] is None
                else float(values[i - 1][1] - values[i][1])
                for i in range(len(values))
            ]
            acceleration = [
                None
                if (i < 2 or values[i] is None or values[i - 1] is None or values[i - 2] is None)
                else float(2 * values[i - 1][1] - values[i][1] - values[i - 2][1])
                for i in range(len(values))
            ]
            result[name] = (tuple(displacement), tuple(velocity), tuple(acceleration))
        if usable_frames:
            status = (
                None
                if usable_frames == len(sequence.frames)
                else (
                    f"Regional motion incomplete: {usable_frames}/{len(sequence.frames)} "
                    "frames usable"
                )
            )
        elif not candidate_records:
            status = "Regional motion unavailable: no viewer-format MediaPipe records"
        elif len(validation_errors) == candidate_records:
            status = f"Regional motion unavailable: {validation_errors[0]}"
        else:
            status = f"Regional motion incomplete: 0/{len(sequence.frames)} frames usable"
        return MappingProxyType(result), status

    def set_frame(self, frame: int) -> ViewerSessionSnapshot:
        sequence = self._require_sequence()
        sequence.set_frame(frame)
        self._snapshot = replace(
            self._snapshot,
            mediapipe_image=sequence.index.asset_path(frame, "illustration"),
            mediapipe_record=None,
            mediapipe_error=None,
        )
        return self._update_mediapipe()

    def set_reference(self, frame: int) -> ViewerSessionSnapshot:
        sequence = self._require_sequence()
        sequence.set_reference(frame)
        self._snapshot = replace(self._snapshot, sequence=sequence)
        return self._snapshot

    def _require_sequence(self) -> MeshSequence:
        sequence = self._snapshot.sequence
        if sequence is None:
            raise ValueError("No sequence selected")
        return sequence

    def _update_mediapipe(self) -> ViewerSessionSnapshot:
        sequence = self._require_sequence()
        if self._mediapipe_root is None:
            return self._snapshot
        record_path = mediapipe_frame_path(
            self._mediapipe_root, sequence.index, sequence.current_frame
        )
        try:
            record = load_mediapipe_frame(record_path)
        except FileNotFoundError:
            return self._snapshot
        except ValueError as exc:
            self._snapshot = replace(self._snapshot, mediapipe_error=str(exc))
            return self._snapshot
        self._snapshot = replace(self._snapshot, mediapipe_record=record)
        return self._snapshot

    def reload_annotations(self) -> ViewerSessionSnapshot:
        if self._annotation_output is None:
            self._annotations = AnnotationDocument()
        else:
            self._annotations = load_annotation_document(self._annotation_output, self.indices)
        selected = self._snapshot.index
        self._snapshot = replace(
            self._snapshot,
            annotations=tuple(
                item
                for item in self._annotations.annotations
                if selected is not None and item.sequence == selected.key
            ),
        )
        return self._snapshot

    def save_annotations(
        self, annotations: tuple[RegionalMotionAnnotation, ...]
    ) -> ViewerSessionSnapshot:
        selected = self._snapshot.index
        all_items = (
            tuple(
                item
                for item in self._annotations.annotations
                if selected is None or item.sequence != selected.key
            )
            + annotations
        )
        document = AnnotationDocument(all_items)
        if self._annotation_output is not None:
            save_annotation_document(self._annotation_output, document, self.indices)
        self._annotations = document
        self._snapshot = replace(self._snapshot, annotations=annotations)
        return self._snapshot
