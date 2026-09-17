import json
from pathlib import Path

import pytest

from meviewer.annotations import (
    AnnotationDocument,
    RegionalMotionAnnotation,
    load_annotation_document,
    save_annotation_document,
)
from meviewer.assets import FrameAssets, SequenceIndex


def index() -> SequenceIndex:
    return SequenceIndex(
        "lfann-v3",
        "sub01",
        "01",
        Path("."),
        {frame: FrameAssets(None, None, None) for frame in (48, 49, 63, 64)},
    )


def annotation(**kwargs: object) -> RegionalMotionAnnotation:
    values = dict(
        sequence="lfann-v3/sub01/01",
        roi_name="left_brow",
        start_frame=48,
        end_frame=63,
        direction="raise",
        confidence=3,
        note="seen",
    )
    values.update(kwargs)
    return RegionalMotionAnnotation(**values)


def test_round_trip_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    document = AnnotationDocument((annotation(),))
    save_annotation_document(path, document, {index().key: index()})
    assert json.loads(path.read_text()) == {
        "version": 1,
        "annotations": [
            {
                "sequence": index().key,
                "roi_name": "left_brow",
                "start_frame": 48,
                "end_frame": 63,
                "direction": "raise",
                "confidence": 3,
                "note": "seen",
            }
        ],
    }
    assert load_annotation_document(path, {index().key: index()}) == document
    assert load_annotation_document(
        tmp_path / "missing.json", {index().key: index()}
    ) == AnnotationDocument(())


@pytest.mark.parametrize(
    "changes",
    [
        {"sequence": "unknown"},
        {"start_frame": 47},
        {"end_frame": 50},
        {"roi_name": "nose"},
        {"direction": "up"},
        {"start_frame": 63, "end_frame": 48},
        {"confidence": 0},
        {"confidence": 4},
        {"confidence": True},
    ],
)
def test_invalid_annotation_rejected(tmp_path: Path, changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        save_annotation_document(
            tmp_path / "a.json",
            AnnotationDocument((annotation(**changes),)),
            {index().key: index()},
        )


def test_overlap_rejected_and_adjacent_allowed(tmp_path: Path) -> None:
    indices = {index().key: index()}
    path = tmp_path / "a.json"
    save_annotation_document(
        path,
        AnnotationDocument((annotation(end_frame=63), annotation(start_frame=64, end_frame=64))),
        indices,
    )
    with pytest.raises(ValueError):
        save_annotation_document(
            path,
            AnnotationDocument((annotation(), annotation(start_frame=63, end_frame=64))),
            indices,
        )
