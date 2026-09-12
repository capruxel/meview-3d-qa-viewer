#!/usr/bin/env python3
"""Local PySide6/PyVista QA viewer for MEVIEW dense-mesh sequences."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from pyvistaqt import QtInteractor
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QColorDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QStyle,
    QStyleOptionSlider,
    QWidget,
)

from mesh_data import FrameCache, SequenceIndex, load_active_frames, scan_sequences
from mediapipe_data import MediaPipeFrame, load_mediapipe_frame, mediapipe_frame_path
from viewer_config import add_config_argument, apply_config_defaults
from viewer_manifest import load as load_viewer_manifest

LABELS = {0: "Positive", 1: "Negative", 2: "Surprise"}
MOTION_CLIM = (0.0, 1.0)  # Fixed QA legend; this is not the 3456-D feature scale.



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
    except (IndexError, KeyError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid raw-video metadata: {path}") from exc
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Invalid raw-video timing: {path}")
    return RawVideo(path, frame_count, fps)


@lru_cache(maxsize=3)
def load_vertex_colors(path: Path, vertex_count: int) -> np.ndarray:
    with path.open(encoding="utf-8", errors="replace") as obj_file:
        values = np.fromstring(
            " ".join(line[2:] for line in obj_file if line.startswith("v ")),
            sep=" ",
            dtype=float,
        )
    if values.size != vertex_count * 6:
        raise ValueError(f"Expected {vertex_count} colored vertices in {path}")
    return np.clip(np.rint(values.reshape(vertex_count, 6)[:, 3:] * 255), 0, 255).astype(np.uint8)


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
        if not isinstance(name, str) or not name or not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"Invalid landmark entry {name!r}: {index!r}: {path}")
        if not 0 <= index < vertex_count:
            raise ValueError(f"Landmark index out of bounds for {name!r}: {index}: {path}")
    return LandmarkMapping(variant, vertex_count, landmarks)


def face_indices_are_valid(mesh: pv.PolyData) -> bool:
    faces = mesh.faces
    position = 0
    while position < len(faces):
        size = int(faces[position])
        if size < 3 or position + size >= len(faces):
            return False
        indices = faces[position + 1 : position + size + 1]
        if len(indices) != size or indices.min(initial=0) < 0 or indices.max(initial=-1) >= mesh.n_points:
            return False
        position += size + 1
    return position == len(faces)


class MeshSequence:
    """One fixed topology plus a bounded cache of NPY vertex coordinates."""

    def __init__(self, index: SequenceIndex, mesh: pv.PolyData, cache: FrameCache):
        self.index = index
        self.mesh = mesh
        self.cache = cache
        self.frames = index.frame_numbers
        self.current_frame = self.frames[0]
        self.reference_frame = self.frames[0]

    @classmethod
    def load(cls, index: SequenceIndex) -> "MeshSequence":
        if not index.frames:
            raise ValueError(f"No numbered mesh frames: {index.directory}")
        missing = [
            f"{frame:03d} ({', '.join(asset for asset in ('npy', 'obj', 'jpg') if asset not in index.frames[frame])})"
            for frame in index.frame_numbers
            if any(asset not in index.frames[frame] for asset in ("npy", "obj", "jpg"))
        ]
        if missing:
            raise ValueError(f"Missing paired assets in {index.key}: {'; '.join(missing)}")
        first_frame = index.frame_numbers[0]
        first_obj = index.frames[first_frame]["obj"]
        try:
            mesh = pv.read(first_obj)
        except Exception as exc:  # PyVista exposes VTK reader failures as several exception types.
            raise ValueError(f"Cannot read OBJ topology {first_obj}: {exc}") from exc
        if not isinstance(mesh, pv.PolyData) or mesh.n_points == 0 or not face_indices_are_valid(mesh):
            raise ValueError(f"Invalid OBJ topology: {first_obj}")
        cache = FrameCache(index, mesh.n_points)
        # Validate the first displayable frame before the mesh enters the viewport.
        cache.get(first_frame)
        return cls(index, mesh, cache)

    def set_frame(self, frame: int) -> np.ndarray:
        if frame not in self.index.frames:
            raise ValueError(f"Frame {frame} not present in {self.index.key}")
        points = self.cache.get(frame)
        self.mesh.points[:] = points
        colors = load_vertex_colors(self.index.frames[frame]["obj"], self.mesh.n_points)
        if "vertex_colors" in self.mesh.point_data:
            self.mesh.point_data["vertex_colors"][:] = colors
        else:
            self.mesh.point_data["vertex_colors"] = colors
        self.mesh.GetPoints().Modified()
        self.mesh.GetPointData().GetArray("vertex_colors").Modified()
        self.mesh.Modified()
        self.current_frame = frame
        self.cache.preload_neighbors(frame)
        return points

    def set_reference(self, frame: int) -> None:
        if frame not in self.index.frames:
            raise ValueError(f"Reference frame {frame} not present in {self.index.key}")
        self.cache.get(frame)
        self.reference_frame = frame

    def previous_frames(self) -> tuple[int | None, int | None]:
        current_index = self.frames.index(self.current_frame)
        previous = self.frames[current_index - 1] if current_index >= 1 else None
        previous_previous = self.frames[current_index - 2] if current_index >= 2 else None
        return previous, previous_previous

    def diagnostics(self) -> dict[str, np.ndarray | None]:
        current = self.cache.get(self.current_frame)
        reference = self.cache.get(self.reference_frame)
        previous, previous_previous = self.previous_frames()
        velocity = None if previous is None else np.linalg.norm(current - self.cache.get(previous), axis=1)
        acceleration = (
            None
            if previous is None or previous_previous is None
            else np.linalg.norm(current - 2 * self.cache.get(previous) + self.cache.get(previous_previous), axis=1)
        )
        return {
            "displacement": np.linalg.norm(current - reference, axis=1),
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def pooling_bin(self, bin_number: int) -> tuple[np.ndarray, float]:
        current = self.cache.get(self.current_frame)
        reference = self.cache.get(self.reference_frame)
        motion = np.ascontiguousarray((current - reference).T * 100.0, dtype=np.float32)
        pooled = F.adaptive_max_pool1d(torch.from_numpy(motion).unsqueeze(0), 64)[0, :, bin_number].numpy()
        start = (bin_number * len(current)) // 64
        end = ((bin_number + 1) * len(current) + 63) // 64
        return np.arange(start, min(end, len(current))), float(np.linalg.norm(pooled))


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
            metadata[key] = {"label": LABELS.get(int(label), str(label)), "subject": str(subject), "video": str(video).zfill(2)}
    return metadata


def prediction_metadata(data_root: Path, results_dir: Path | None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
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
        variant_rows = sorted((row for row in rows if row.get("method", "").lower() == variant), key=lambda row: int(row["sample_order"]))
        if len(variant_rows) != len(order):
            warnings.append(f"Prediction mapping unavailable for {variant}: expected {len(order)} rows, got {len(variant_rows)}")
            continue
        for row, sample_index in zip(variant_rows, order, strict=True):
            subject, video, label = str(groups[sample_index]), str(videos[sample_index]).zfill(2), LABELS.get(int(labels[sample_index]), str(labels[sample_index]))
            if row.get("video_id", "").zfill(2) != video or row.get("true_label") != label:
                warnings.append(f"Prediction mismatch for {variant}/{subject}/{video}; row {row.get('sample_order')} ignored")
                continue
            predictions[f"{variant}/{subject}/{video}"] = row
    return predictions, metrics, warnings


class ActiveFrameSlider(QSlider):
    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.active_start: int | None = None
        self.active_end: int | None = None
        self.reference: int | None = None

    def set_markers(self, start: int | None, end: int | None, reference: int | None) -> None:
        self.active_start, self.active_end, self.reference = start, end, reference
        self.update()

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if self.maximum() <= self.minimum():
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self
        )
        scale = groove.width() / (self.maximum() - self.minimum())

        def position(frame: int) -> int:
            return groove.x() + round((frame - self.minimum()) * scale)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        marker_top, marker_bottom = groove.y() - 5, groove.bottom() + 5
        if self.active_start is not None and self.active_end is not None:
            start, end = position(self.active_start), position(self.active_end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e6a23c80"))
            painter.drawRoundedRect(start, groove.y() - 2, max(3, end - start + 1), 4, 2, 2)
            painter.setPen(QColor("#e6a23c"))
            painter.drawLine(start, marker_top, start, marker_bottom)
            painter.drawLine(end, marker_top, end, marker_bottom)
        if self.reference is not None:
            reference = position(self.reference)
            painter.setPen(QColor("#5dade2"))
            painter.drawLine(reference, marker_top - 2, reference, marker_bottom + 2)
        painter.end()


class LandmarkImageWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.image = QPixmap()
        self.frame: Any = None
        self.status = "No MediaPipe record"
        self.show_478 = True
        self.show_20 = True
        self.show_rois = True
        self.point_size = 10

    def set_frame(self, image_path: Path, frame: Any) -> None:
        self.image = QPixmap(str(image_path)) if image_path.is_file() else QPixmap()
        self.frame = frame
        self.status = "No illustration image" if self.image.isNull() else (
            "No face" if frame is not None and frame.status == "no_face" else
            "Ready" if frame is not None else "No MediaPipe record"
        )
        self.update()

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202124"))
        if self.image.isNull():
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.status)
            painter.end()
            return
        target = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target_rect = target.toRect()
        target_rect.moveTo(
            (self.width() - target_rect.width()) // 2,
            (self.height() - target_rect.height()) // 2,
        )
        painter.drawPixmap(target_rect, self.image)
        if self.frame is not None and self.frame.status == "ok":
            def draw(points: np.ndarray, color: str) -> None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(color))
                radius = max(2, self.point_size / 2)
                for x, y, _ in points:
                    if not np.isfinite((x, y)).all():
                        continue
                    px = target_rect.left() + float(np.clip(x, 0, 1)) * target_rect.width()
                    py = target_rect.top() + float(np.clip(y, 0, 1)) * target_rect.height()
                    painter.drawEllipse(round(px - radius), round(py - radius), round(2 * radius), round(2 * radius))

            if self.show_478:
                draw(self.frame.landmarks478, "#44aaff")
            if self.show_20 and len(self.frame.landmarks20):
                draw(self.frame.landmarks20, "#ffcc00")
            if self.show_rois and len(self.frame.roi_centers):
                draw(self.frame.roi_centers, "#ff5555")
        painter.setPen(QColor("white"))
        painter.drawText(8, 20, self.status)
        painter.end()




class MeshViewer(QMainWindow):
    def __init__(
        self,
        data_root: Path,
        mesh_root: Path,
        raw_root: Path | None,
        active_frames_path: Path,
        results_dir: Path | None,
        landmarks_root: Path | None,
        mediapipe_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MEVIEW 3D QA Viewer")
        self.raw_root = raw_root
        self.active_frames = load_active_frames(active_frames_path)
        self.indices = {sequence.key: sequence for sequence in scan_sequences(mesh_root, {"lfann-v3": mesh_root})}
        self.metadata = sequence_metadata(data_root)
        self.predictions, self.metrics, self.prediction_warnings = prediction_metadata(data_root, results_dir)
        self.landmarks_root = landmarks_root
        self.mediapipe_root = mediapipe_root
        self.sequence: MeshSequence | None = None
        self.raw_video: RawVideo | None = None
        self.landmark_mapping: LandmarkMapping | None = None
        self.landmark_color = QColor("#ffcc00")
        self.motion_layer: str | None = None
        self._advancing = False
        self._mesh_style: tuple[bool, bool, bool] | None = None
        self._wire_visible: bool | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.raw_player = QMediaPlayer(self)
        self._build_ui()
        self._populate_sequences()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._browser())
        splitter.addWidget(self._viewport())
        splitter.addWidget(self._inspector())
        self.solid_toggle.toggled.connect(lambda checked: self._mirror_viewport_toggle(self.mesh_layer, checked))
        self.wire_toggle.toggled.connect(lambda checked: self._mirror_viewport_toggle(self.wire_layer, checked))
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self._player())
        self.setCentralWidget(root)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.previous_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.next_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self.first_frame)
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self.last_frame)

    def _browser(self) -> QWidget:
        panel = QGroupBox("Sequence browser")
        layout = QVBoxLayout(panel)
        self.sequence_tree = QTreeWidget()
        self.sequence_tree.setHeaderHidden(True)
        self.sequence_tree.itemSelectionChanged.connect(self.load_selected_sequence)
        self.sequence_status = QLabel("No sequence")
        self.sequence_details = QLabel()
        self.sequence_details.setWordWrap(True)
        layout.addWidget(self.sequence_tree, 1)
        layout.addWidget(self.sequence_status)
        layout.addWidget(self.sequence_details)
        return panel

    def _viewport(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        controls = QHBoxLayout()
        self.solid_toggle = QCheckBox("Solid mesh")
        self.solid_toggle.setChecked(True)
        self.original_color_toggle = QCheckBox("Original colors")
        self.original_color_toggle.setChecked(True)
        self.wire_toggle = QCheckBox("Wireframe")
        self.axes_toggle = QCheckBox("Axes")
        self.axes_toggle.setChecked(True)
        for toggle in (self.solid_toggle, self.original_color_toggle, self.wire_toggle, self.axes_toggle):
            toggle.toggled.connect(self.refresh_view)
            controls.addWidget(toggle)
        for label, callback in (("Reset", self.reset_camera), ("Front", lambda: self.plotter.view_xy()), ("Side", lambda: self.plotter.view_yz()), ("Top", lambda: self.plotter.view_xz()), ("Screenshot", self.screenshot)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        comparison = QSplitter(Qt.Orientation.Horizontal)
        raw_panel = QGroupBox("Raw video (mesh-locked)")
        raw_layout = QVBoxLayout(raw_panel)
        self.raw_video_widget = QVideoWidget()
        self.raw_player.setVideoOutput(self.raw_video_widget)
        self.raw_status = QLabel("No raw video")
        self.raw_status.setWordWrap(True)
        raw_layout.addWidget(self.raw_video_widget, 1)
        raw_layout.addWidget(self.raw_status)
        self.mediapipe_image = LandmarkImageWidget()
        image_tabs = QTabWidget()
        image_tabs.addTab(raw_panel, "Raw video")
        image_tabs.addTab(self.mediapipe_image, "MediaPipe")
        comparison.addWidget(image_tabs)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#202124")
        self.plotter.add_axes()
        comparison.addWidget(self.plotter)
        comparison.setStretchFactor(0, 1)
        comparison.setStretchFactor(1, 2)
        layout.addWidget(comparison, 1)
        return panel

    def _inspector(self) -> QWidget:
        panel = QGroupBox("Inspector")
        layout = QVBoxLayout(panel)
        self.info = QLabel("Select a sequence.")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info)


        layers = QGroupBox("Diagnostic layers (not 3456-D features)")
        layers_layout = QGridLayout(layers)
        self.mesh_layer = QCheckBox()
        self.mesh_layer.setChecked(True)
        self.wire_layer = QCheckBox()
        self.displacement_toggle = QCheckBox("Displacement")
        self.velocity_toggle = QCheckBox("Velocity")
        self.acceleration_toggle = QCheckBox("Acceleration")
        self.pooling_toggle = QCheckBox("Pooling debug")
        self.pool_bin = QSpinBox()
        self.pool_bin.setRange(0, 63)
        for row, (name, control) in enumerate((("Mesh", self.mesh_layer), ("Wireframe", self.wire_layer), ("Displacement", self.displacement_toggle), ("Velocity", self.velocity_toggle), ("Acceleration", self.acceleration_toggle), ("Pooling debug", self.pooling_toggle))):
            layers_layout.addWidget(QLabel(name), row, 0)
            layers_layout.addWidget(control, row, 1)
        layers_layout.addWidget(QLabel("Pool bin"), 6, 0)
        layers_layout.addWidget(self.pool_bin, 6, 1)
        for layer, toggle in (("displacement", self.displacement_toggle), ("velocity", self.velocity_toggle), ("acceleration", self.acceleration_toggle)):
            toggle.toggled.connect(lambda checked, selected=layer: self.set_motion_layer(selected, checked))
        self.pooling_toggle.toggled.connect(self.refresh_view)
        self.pool_bin.valueChanged.connect(self.refresh_view)
        self.mesh_layer.toggled.connect(lambda checked: self._mirror_inspector_toggle(self.solid_toggle, checked))
        self.wire_layer.toggled.connect(lambda checked: self._mirror_inspector_toggle(self.wire_toggle, checked))
        layout.addWidget(layers)

        landmark_group = QGroupBox("Landmarks")
        landmark_layout = QFormLayout(landmark_group)
        self.landmark_status = QLabel("No landmark mapping found")
        self.landmark_toggle = QCheckBox("Markers")
        self.landmark_labels_toggle = QCheckBox("Labels")
        self.landmark_choice = QComboBox()
        self.landmark_size = QSpinBox()
        self.landmark_size.setRange(1, 30)
        self.landmark_size.setValue(10)
        color = QPushButton("Color")
        color.clicked.connect(self.choose_landmark_color)
        for control in (self.landmark_toggle, self.landmark_labels_toggle, self.landmark_choice, self.landmark_size):
            if hasattr(control, "toggled"):
                control.toggled.connect(self.refresh_view)
            else:
                control.currentTextChanged.connect(self.refresh_view) if isinstance(control, QComboBox) else control.valueChanged.connect(self.refresh_view)
        landmark_layout.addRow(self.landmark_status)
        landmark_layout.addRow(self.landmark_toggle)
        landmark_layout.addRow(self.landmark_labels_toggle)
        landmark_layout.addRow("Selection", self.landmark_choice)
        landmark_layout.addRow("Marker size", self.landmark_size)
        landmark_layout.addRow(color)
        layout.addWidget(landmark_group)

        mediapipe_group = QGroupBox("MediaPipe illustration")
        mediapipe_layout = QFormLayout(mediapipe_group)
        self.mediapipe_478_toggle = QCheckBox("478 points")
        self.mediapipe_478_toggle.setChecked(True)
        self.mediapipe_20_toggle = QCheckBox("Configured 20")
        self.mediapipe_20_toggle.setChecked(True)
        self.mediapipe_roi_toggle = QCheckBox("ROI centers")
        self.mediapipe_roi_toggle.setChecked(True)
        for control in (self.mediapipe_478_toggle, self.mediapipe_20_toggle, self.mediapipe_roi_toggle):
            control.toggled.connect(self.refresh_view)
        mediapipe_layout.addRow(self.mediapipe_478_toggle)
        mediapipe_layout.addRow(self.mediapipe_20_toggle)
        mediapipe_layout.addRow(self.mediapipe_roi_toggle)
        layout.addWidget(mediapipe_group)
        layout.addStretch()
        return panel

    def _player(self) -> QWidget:
        panel = QFrame()
        layout = QHBoxLayout(panel)
        for label, callback in (("|<", self.first_frame), ("<", self.previous_frame), (">", self.toggle_play), (">|", self.next_frame), (">|>", self.last_frame)):
            button = QToolButton()
            button.setText(label)
            button.clicked.connect(callback)
            if label == ">":
                self.play_button = button
                self.play_button.setToolTip("Play")
            layout.addWidget(button)
        self.frame_label = QLabel("0 / 0")
        self.reference_label = QLabel("Ref —")
        self.active_label = QLabel("Onset–offset —")
        reference_button = QToolButton()
        reference_button.setText("Set ref")
        reference_button.clicked.connect(self.set_reference_to_current)
        self.timeline = ActiveFrameSlider()
        self.timeline.setMinimumHeight(28)
        self.timeline.valueChanged.connect(self.set_frame_by_index)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.update_timer_interval)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 2.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix("×")
        self.speed_spin.valueChanged.connect(self.update_timer_interval)
        self.loop_toggle = QCheckBox("Loop")
        layout.addWidget(self.timeline, 1)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.reference_label)
        layout.addWidget(reference_button)
        layout.addWidget(self.active_label)
        layout.addWidget(QLabel("FPS"))
        layout.addWidget(self.fps_spin)
        layout.addWidget(QLabel("Speed"))
        layout.addWidget(self.speed_spin)
        layout.addWidget(self.loop_toggle)
        return panel

    def _populate_sequences(self) -> None:
        first_video: QTreeWidgetItem | None = None
        with QSignalBlocker(self.sequence_tree):
            self.sequence_tree.clear()
            for variant in sorted({sequence.variant for sequence in self.indices.values()}):
                variant_item = QTreeWidgetItem([variant.upper()])
                self.sequence_tree.addTopLevelItem(variant_item)
                for subject in sorted({sequence.subject for sequence in self.indices.values() if sequence.variant == variant}):
                    subject_item = QTreeWidgetItem([subject])
                    variant_item.addChild(subject_item)
                    for sequence in sorted(
                        (sequence for sequence in self.indices.values() if sequence.variant == variant and sequence.subject == subject),
                        key=lambda sequence: sequence.video,
                    ):
                        active = self.active_frames.get(f"{sequence.subject}_{sequence.video}")
                        timing = f" · onset F{active[0]}–offset F{active[1]}" if active else " · onset–offset unavailable"
                        video_item = QTreeWidgetItem([f"Video {sequence.video}{timing}"])
                        video_item.setData(0, Qt.ItemDataRole.UserRole, sequence.key)
                        subject_item.addChild(video_item)
                        first_video = first_video or video_item
        self.sequence_tree.expandToDepth(1)
        if first_video is not None:
            self.sequence_tree.setCurrentItem(first_video)

    def selected_index(self) -> SequenceIndex | None:
        item = self.sequence_tree.currentItem()
        key = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        return self.indices.get(key) if isinstance(key, str) else None

    def load_selected_sequence(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        self.timer.stop()
        self._set_playing(False)
        try:
            sequence = MeshSequence.load(index)
        except ValueError as exc:
            self.sequence = None
            self.sequence_status.setText(f"Invalid: {exc}")
            self.info.setText(str(exc))
            self.timeline.setRange(0, 0)
            self.plotter.clear()
            return
        self.sequence = sequence
        self._mesh_style = None
        self._wire_visible = None
        try:
            self.raw_video = probe_raw_video(self.raw_root, index)
        except ValueError as exc:
            self.raw_video = None
            self.raw_player.stop()
            self.raw_player.setSource(QUrl())
            self.raw_status.setText(str(exc))
        else:
            self.raw_player.stop()
            self.raw_player.setSource(QUrl.fromLocalFile(str(self.raw_video.path)))
            synced = min(len(sequence.frames), self.raw_video.frame_count)
            if self.raw_video.frame_count > len(sequence.frames):
                note = f"; raw tail F{synced + 1}–{self.raw_video.frame_count} unavailable in mesh"
            elif self.raw_video.frame_count < len(sequence.frames):
                note = f"; mesh F{synced + 1}–{len(sequence.frames)} has no raw frame"
            else:
                note = ""
            self.raw_status.setText(f"Locked: mesh F1–{synced} ↔ raw F1–{synced} at {self.raw_video.fps:g} FPS{note}")
        active = self.active_frames.get(f"{index.subject}_{index.video}")
        if active and active[0] in index.frames:
            sequence.set_reference(active[0])
        self._load_landmarks(sequence)
        self.sequence_status.setText(f"Ready: {len(sequence.frames)} frames")
        self.sequence_details.setText(self._metadata_text(index, active))
        self.timeline.setRange(0, len(sequence.frames) - 1)
        maximum_label = f"{len(sequence.frames)} / {len(sequence.frames)} (frame {sequence.frames[-1]})"
        self.frame_label.setFixedWidth(self.frame_label.fontMetrics().horizontalAdvance(maximum_label) + 8)
        active_start = sequence.frames.index(active[0]) if active and active[0] in sequence.frames else None
        active_end = sequence.frames.index(active[1]) if active and active[1] in sequence.frames else None
        self.timeline.set_markers(active_start, active_end, sequence.frames.index(sequence.reference_frame))
        self.active_label.setText(f"Onset–offset F{active[0]}–F{active[1]}" if active else "Onset–offset —")
        self.reference_label.setText(f"Ref F{sequence.reference_frame}")
        self.set_frame_by_index(0)
        self.reset_camera()

    def _metadata_text(self, index: SequenceIndex, active: tuple[int, int] | None) -> str:
        metadata = self.metadata.get(index.key, {})
        return "\n".join((f"Label: {metadata.get('label', 'unknown')}", f"Subject/video: {index.subject}/{index.video}", f"Onset–offset: F{active[0]}–F{active[1]}" if active else "Onset–offset: unavailable"))

    def _mirror_viewport_toggle(self, inspector_toggle: QCheckBox, checked: bool) -> None:
        with QSignalBlocker(inspector_toggle):
            inspector_toggle.setChecked(checked)

    def _mirror_inspector_toggle(self, viewport_toggle: QCheckBox, checked: bool) -> None:
        with QSignalBlocker(viewport_toggle):
            viewport_toggle.setChecked(checked)
        self.refresh_view()

    def _load_landmarks(self, sequence: MeshSequence) -> None:
        mapping_path = None if self.landmarks_root is None else self.landmarks_root / f"{sequence.index.variant}.json"
        try:
            self.landmark_mapping = load_landmarks(self.landmarks_root, sequence.index.variant, sequence.mesh.n_points)
        except ValueError as exc:
            self.landmark_mapping = None
            self.landmark_status.setText(str(exc))
        if self.landmark_mapping is None:
            if mapping_path is None or not mapping_path.is_file():
                self.landmark_status.setText(f"No landmark mapping found for {sequence.index.variant}")
            for control in (self.landmark_toggle, self.landmark_labels_toggle, self.landmark_choice, self.landmark_size):
                control.setEnabled(False)
            return
        for control in (self.landmark_toggle, self.landmark_labels_toggle, self.landmark_choice, self.landmark_size):
            control.setEnabled(True)
        self.landmark_status.setText(f"{len(self.landmark_mapping.landmarks)} mapped landmarks")
        with QSignalBlocker(self.landmark_choice):
            self.landmark_choice.clear()
            self.landmark_choice.addItem("All landmarks")
            self.landmark_choice.addItems(sorted(self.landmark_mapping.landmarks))

    def _update_mediapipe(self) -> None:
        if self.sequence is None:
            return
        frame = self.sequence.current_frame
        image_path = self.sequence.index.frames[frame].get("jpg", Path())
        record_path = mediapipe_frame_path(self.mediapipe_root, self.sequence.index, frame)
        record: MediaPipeFrame | None = None
        error: str | None = None
        try:
            record = load_mediapipe_frame(record_path)
        except FileNotFoundError:
            pass
        except ValueError as exc:
            error = str(exc)
        self.mediapipe_image.show_478 = self.mediapipe_478_toggle.isChecked()
        self.mediapipe_image.show_20 = self.mediapipe_20_toggle.isChecked()
        self.mediapipe_image.show_rois = self.mediapipe_roi_toggle.isChecked()
        self.mediapipe_image.point_size = self.landmark_size.value()
        self.mediapipe_image.set_frame(image_path, record)
        if error:
            self.mediapipe_image.status = error
            self.mediapipe_image.update()


    def set_reference_to_current(self) -> None:
        if self.sequence is None:
            return
        self.sequence.set_reference(self.sequence.current_frame)
        self.reference_label.setText(f"Ref F{self.sequence.reference_frame}")
        self.timeline.set_markers(self.timeline.active_start, self.timeline.active_end, self.timeline.value())
        self.refresh_view()

    def set_motion_layer(self, layer: str, checked: bool) -> None:
        if checked:
            self.motion_layer = layer
            for other, toggle in (("displacement", self.displacement_toggle), ("velocity", self.velocity_toggle), ("acceleration", self.acceleration_toggle)):
                if other != layer:
                    with QSignalBlocker(toggle):
                        toggle.setChecked(False)
        elif self.motion_layer == layer:
            self.motion_layer = None
        self.refresh_view()

    def set_frame_by_index(self, frame_index: int) -> None:
        if self.sequence is None:
            return
        try:
            self.sequence.set_frame(self.sequence.frames[frame_index])
        except ValueError as exc:
            self.sequence_status.setText(f"Invalid: {exc}")
            self.timer.stop()
            self._set_playing(False)
            return
        if not self._advancing:
            self._seek_raw_frame(resume=self.timer.isActive())
        self.frame_label.setText(f"{frame_index + 1} / {len(self.sequence.frames)} (frame {self.sequence.current_frame})")
        self.refresh_view()

    def _seek_raw_frame(self, *, resume: bool = False) -> None:
        if self.sequence is None or self.raw_video is None or self.sequence.current_frame > self.raw_video.frame_count:
            return
        self.raw_player.pause()
        self.raw_player.setPosition(self.raw_video.position_ms(self.sequence.current_frame))
        if resume:
            self.raw_player.setPlaybackRate(self._raw_playback_rate())
            self.raw_player.play()


    def refresh_view(self) -> None:
        if self.sequence is None:
            return
        mesh = self.sequence.mesh
        diagnostics = self.sequence.diagnostics()
        scalars = diagnostics.get(self.motion_layer) if self.motion_layer else None
        if scalars is not None:
            if "diagnostic_motion" in mesh.point_data:
                mesh.point_data["diagnostic_motion"][:] = scalars
            else:
                mesh.point_data["diagnostic_motion"] = scalars
            mesh.GetPointData().GetArray("diagnostic_motion").Modified()
            mesh.Modified()
        mesh_style = (self.solid_toggle.isChecked(), scalars is not None, self.original_color_toggle.isChecked())
        if mesh_style != self._mesh_style:
            self.plotter.remove_actor("mesh", render=False)
            if mesh_style[0] or mesh_style[1]:
                mesh_args: dict[str, Any] = {
                    "name": "mesh",
                    "smooth_shading": True,
                    "lighting": True,
                    "specular": 0.15,
                }
                if mesh_style[1]:
                    mesh_args.update(scalars="diagnostic_motion", cmap="turbo", clim=MOTION_CLIM, show_scalar_bar=True)
                elif mesh_style[2]:
                    mesh_args.update(scalars="vertex_colors", rgb=True)
                else:
                    mesh_args.update(color="#d0d0d0")
                self.plotter.add_mesh(mesh, **mesh_args)
            self._mesh_style = mesh_style
        wire_visible = self.wire_toggle.isChecked()
        if wire_visible != self._wire_visible:
            self.plotter.remove_actor("wireframe", render=False)
            if wire_visible:
                self.plotter.add_mesh(mesh, name="wireframe", style="wireframe", color="#111111", line_width=1)
            self._wire_visible = wire_visible
        self._update_pooling(mesh)
        self._update_landmarks(mesh)
        self._update_mediapipe()
        if self.axes_toggle.isChecked():
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self._update_inspector(diagnostics, scalars)
        self.plotter.render()

    def _update_pooling(self, mesh: pv.PolyData) -> None:
        self.plotter.remove_actor("pooling-bin", render=False)
        if not self.pooling_toggle.isChecked() or self.sequence is None:
            return
        indices, response = self.sequence.pooling_bin(self.pool_bin.value())
        self.plotter.add_mesh(pv.PolyData(mesh.points[indices]), name="pooling-bin", color="#ff00ff", point_size=5, render_points_as_spheres=True)
        self._pooling_response = response

    def _update_landmarks(self, mesh: pv.PolyData) -> None:
        for name in ("landmarks", "landmark-labels", "landmark-trajectories"):
            self.plotter.remove_actor(name, render=False)
        if self.landmark_mapping is None or not self.landmark_toggle.isChecked() or self.sequence is None:
            return
        selection = self.landmark_choice.currentText()
        items = list(self.landmark_mapping.landmarks.items()) if selection == "All landmarks" else [(selection, self.landmark_mapping.landmarks[selection])]
        names, indices = zip(*items, strict=True)
        points = mesh.points[list(indices)]
        self.plotter.add_mesh(pv.PolyData(points), name="landmarks", color=self.landmark_color.name(), point_size=self.landmark_size.value(), render_points_as_spheres=True)
        reference_points = self.sequence.cache.get(self.sequence.reference_frame)[list(indices)]
        line_cells = np.concatenate([np.array([2, 2 * number, 2 * number + 1]) for number in range(len(indices))])
        paths = pv.PolyData(np.vstack((reference_points, points)), lines=line_cells)
        self.plotter.add_mesh(paths, name="landmark-trajectories", color=self.landmark_color.name(), line_width=2)
        if self.landmark_labels_toggle.isChecked():
            self.plotter.add_point_labels(points, list(names), name="landmark-labels", font_size=10, text_color="white", shape=None, always_visible=True)

    def _update_inspector(self, diagnostics: dict[str, np.ndarray | None], scalars: np.ndarray | None) -> None:
        assert self.sequence is not None
        points = self.sequence.mesh.points
        active = self.active_frames.get(f"{self.sequence.index.subject}_{self.sequence.index.video}")
        prediction = self.predictions.get(self.sequence.index.key)
        motion = []
        for name, values in diagnostics.items():
            motion.append(f"{name}: unavailable" if values is None else f"{name}: max={values.max():.6g}, mean={values.mean():.6g}")
        if self.pooling_toggle.isChecked() and hasattr(self, "_pooling_response"):
            motion.append(f"pooling debug bin {self.pool_bin.value()}: max response={self._pooling_response:.6g}")
        result = "Prediction mapping unavailable" if prediction is None else f"Prediction: {prediction['true_label']} → {prediction['predicted_label']} ({'correct' if prediction['correct'] == 'True' else 'wrong'})"
        warning = "\n".join(self.prediction_warnings)
        metric = self.metrics.get(self.sequence.index.variant, {})
        metric_text = f"Metrics: accuracy={metric.get('accuracy', 'n/a')}, UAR={metric.get('uar', 'n/a')}, UF1={metric.get('uf1', 'n/a')}"
        self.info.setText("\n".join((
            f"Frame: {self.sequence.current_frame}; reference: {self.sequence.reference_frame}",
            f"File: {self.sequence.index.frames[self.sequence.current_frame]['npy']}",
            f"Vertices/faces: {self.sequence.mesh.n_points}/{self.sequence.mesh.n_cells}",
            f"Bounds: min={points.min(axis=0).round(5).tolist()} max={points.max(axis=0).round(5).tolist()}",
            f"Active window: {active[0]}–{active[1]}" if active else "Active window: unavailable",
            *motion,
            result,
            metric_text,
            warning,
        )))

    def first_frame(self) -> None:
        self.timeline.setValue(self.timeline.minimum())

    def last_frame(self) -> None:
        self.timeline.setValue(self.timeline.maximum())

    def previous_frame(self) -> None:
        self.timeline.setValue(max(self.timeline.minimum(), self.timeline.value() - 1))

    def next_frame(self) -> None:
        if self.timeline.value() == self.timeline.maximum():
            if self.loop_toggle.isChecked():
                self.first_frame()
                self._seek_raw_frame(resume=True)
            else:
                self.timer.stop()
                self.raw_player.pause()
                self._set_playing(False)
            return
        self.timeline.setValue(self.timeline.value() + 1)

    def advance_frame(self) -> None:
        self._advancing = True
        try:
            self.next_frame()
        finally:
            self._advancing = False

    def _raw_playback_rate(self) -> float:
        return 1.0 if self.raw_video is None else self.fps_spin.value() * self.speed_spin.value() / self.raw_video.fps

    def _set_playing(self, playing: bool) -> None:
        self.play_button.setText("||" if playing else ">")
        self.play_button.setToolTip("Pause" if playing else "Play")

    def toggle_play(self) -> None:
        if self.sequence is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.raw_player.pause()
            self._set_playing(False)
        else:
            self._seek_raw_frame()
            self.update_timer_interval()
            if self.raw_video is not None:
                self.raw_player.setPlaybackRate(self._raw_playback_rate())
                self.raw_player.play()
            self.timer.start()
            self._set_playing(True)

    def update_timer_interval(self) -> None:
        self.timer.setInterval(round(1000 / (self.fps_spin.value() * self.speed_spin.value())))
        if self.timer.isActive() and self.raw_video is not None:
            self.raw_player.setPlaybackRate(self._raw_playback_rate())

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "mesh-qa.png", "PNG image (*.png)")
        if path:
            self.plotter.screenshot(path)

    def choose_landmark_color(self) -> None:
        color = QColorDialog.getColor(self.landmark_color, self, "Landmark color")
        if color.isValid():
            self.landmark_color = color
            self.refresh_view()

    def closeEvent(self, event: Any) -> None:
        self.timer.stop()
        self.raw_player.stop()
        self.plotter.close()
        super().closeEvent(event)


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
