"""Validated, versioned regional-motion annotation documents."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from meviewer.assets import SequenceIndex

ROIS = ("left_brow", "right_brow", "left_mouth_corner", "right_mouth_corner")
BROW_DIRECTIONS = frozenset({"raise", "lower", "not_observable"})
MOUTH_DIRECTIONS = frozenset({"up", "down", "not_observable"})


@dataclass(frozen=True)
class RegionalMotionAnnotation:
    sequence: str
    roi_name: str
    start_frame: int
    end_frame: int
    direction: str
    confidence: int
    note: str = ""


@dataclass(frozen=True)
class AnnotationDocument:
    annotations: tuple[RegionalMotionAnnotation, ...] = ()


def _validate(document: AnnotationDocument, indices: Mapping[str, SequenceIndex]) -> None:
    seen: list[tuple[str, str, int, int]] = []
    for item in document.annotations:
        if item.sequence not in indices:
            raise ValueError(f"Unknown sequence: {item.sequence}")
        if item.roi_name not in ROIS:
            raise ValueError(f"Unknown RoI: {item.roi_name}")
        if type(item.start_frame) is not int or type(item.end_frame) is not int:
            raise ValueError("Frame numbers must be integers")
        if item.start_frame > item.end_frame:
            raise ValueError("Annotation interval is reversed")
        index = indices[item.sequence]
        if not index.has_frame(item.start_frame) or not index.has_frame(item.end_frame):
            raise ValueError("Annotation frame is not present in sequence")
        directions = BROW_DIRECTIONS if "brow" in item.roi_name else MOUTH_DIRECTIONS
        if item.direction not in directions:
            raise ValueError(f"Invalid direction for {item.roi_name}: {item.direction}")
        if type(item.confidence) is not int or item.confidence not in (1, 2, 3):
            raise ValueError("Confidence must be an integer from 1 to 3")
        if not isinstance(item.note, str):
            raise ValueError("Note must be a string")
        key = (item.sequence, item.roi_name)
        for old_sequence, old_roi, start, end in seen:
            if (
                (old_sequence, old_roi) == key
                and item.start_frame <= end
                and start <= item.end_frame
            ):
                raise ValueError("Annotation intervals overlap")
        seen.append((*key, item.start_frame, item.end_frame))


def load_annotation_document(
    path: Path, indices: Mapping[str, SequenceIndex]
) -> AnnotationDocument:
    if not path.is_file():
        return AnnotationDocument()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or type(raw.get("version")) is not int or raw["version"] != 1:
            raise ValueError("Annotation document must have version 1")
        entries = raw.get("annotations")
        if not isinstance(entries, list):
            raise ValueError("annotations must be a list")
        items = tuple(
            RegionalMotionAnnotation(**entry) for entry in entries if isinstance(entry, dict)
        )
        if len(items) != len(entries):
            raise ValueError("Each annotation must be an object")
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Invalid annotation document: {path}: {exc}") from exc
    document = AnnotationDocument(items)
    _validate(document, indices)
    return document


def save_annotation_document(
    path: Path, document: AnnotationDocument, indices: Mapping[str, SequenceIndex]
) -> None:
    _validate(document, indices)
    payload: dict[str, object] = {"version": 1, "annotations": []}
    for item in document.annotations:
        row = {
            "sequence": item.sequence,
            "roi_name": item.roi_name,
            "start_frame": item.start_frame,
            "end_frame": item.end_frame,
            "direction": item.direction,
            "confidence": item.confidence,
        }
        if item.note:
            row["note"] = item.note
        cast_annotations = payload["annotations"]
        assert isinstance(cast_annotations, list)
        cast_annotations.append(row)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
