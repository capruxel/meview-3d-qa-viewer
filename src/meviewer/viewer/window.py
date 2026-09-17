"""Local PySide6/PyVista QA viewer for MEVIEW dense-mesh sequences."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from meviewer.assets import SequenceIndex
from meviewer.viewer.session import (
    LandmarkMapping,
    MeshSequence,
    RawVideo,
    ViewerSession,
)
from meviewer.viewer.widgets import ActiveFrameSlider, LandmarkImageWidget, RegionalMotionChart

FIXED_MOTION_CLIM = (0.0, 1.0)


def diagnostic_color_limits(maximum: float | None, *, auto: bool) -> tuple[float, float]:
    """Return the absolute scale or a stable whole-sequence contrast scale."""
    if not auto or maximum is None or maximum <= np.finfo(float).eps:
        return FIXED_MOTION_CLIM
    return (0.0, maximum)


def diagnostic_scalar_bar_args(
    metric: str, clim: tuple[float, float], *, auto: bool
) -> dict[str, Any]:
    return {
        "title": f"{metric} (P99 {clim[1]:.3g})" if auto else f"{metric} (fixed 0–1)",
        "vertical": True,
        "width": 0.08,
        "height": 0.55,
        "position_x": 0.88,
        "position_y": 0.22,
        "title_font_size": 10,
        "label_font_size": 8,
    }


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
        self.session = ViewerSession(
            data_root,
            mesh_root,
            raw_root,
            active_frames_path,
            results_dir,
            landmarks_root,
            mediapipe_root,
        )
        self.snapshot = self.session.snapshot
        self.landmark_color = QColor("#ffcc00")
        self.motion_layer: str | None = None
        self._advancing = False
        self._mesh_style: tuple[object, ...] | None = None
        self._wire_visible: bool | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.raw_player = QMediaPlayer(self)
        self._build_ui()
        self._populate_sequences()

    @property
    def sequence(self) -> MeshSequence | None:
        return self.snapshot.sequence

    @property
    def raw_video(self) -> RawVideo | None:
        return self.snapshot.raw_video

    @property
    def landmark_mapping(self) -> LandmarkMapping | None:
        return self.snapshot.landmark_mapping

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._browser())
        splitter.addWidget(self._viewport())
        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setWidget(self._inspector())
        splitter.addWidget(inspector_scroll)
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
        self.auto_contrast_toggle = QCheckBox("Auto contrast (P99)")
        self.auto_contrast_toggle.setChecked(True)
        self.auto_contrast_toggle.setToolTip(
            "Scale diagnostic colors to the selected sequence's 99th percentile. "
            "Turn off for fixed 0–1 values."
        )
        self.axes_toggle = QCheckBox("Axes")
        self.axes_toggle.setChecked(True)
        for toggle in (
            self.solid_toggle,
            self.original_color_toggle,
            self.wire_toggle,
            self.auto_contrast_toggle,
            self.axes_toggle,
        ):
            toggle.toggled.connect(self.refresh_view)
            controls.addWidget(toggle)
        for label, callback in (
            ("Reset", self.reset_camera),
            ("Front", lambda: self.plotter.view_xy()),
            ("Side", lambda: self.plotter.view_yz()),
            ("Top", lambda: self.plotter.view_xz()),
            ("Screenshot", self.screenshot),
            ("Export review PNG…", self.export_review),
        ):
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
        self.review_pane = QFrame()
        review_layout = QVBoxLayout(self.review_pane)
        self.review_context = QLabel("Review context: no mesh sequence selected")
        self.review_context.setWordWrap(True)
        self.motion_metric = QComboBox()
        self.motion_metric.addItems(("Displacement", "Velocity", "Acceleration"))
        self.motion_metric.setToolTip("Choose the regional-motion evidence shown in the chart.")
        self.motion_metric.setAccessibleName("Motion evidence")
        self.motion_metric.currentTextChanged.connect(self._render_review_chart)
        self.chart_status = QLabel()
        self.chart_status.setWordWrap(True)
        metric_controls = QHBoxLayout()
        metric_controls.addWidget(QLabel("Motion evidence"))
        metric_controls.addWidget(self.motion_metric)
        metric_controls.addStretch()
        review_layout.addLayout(metric_controls)
        review_layout.addWidget(self.chart_status)
        review_layout.addWidget(self.review_context)
        self.review_chart = RegionalMotionChart()
        review_layout.addWidget(self.review_chart)
        layout.addWidget(self.review_pane)
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

        diagnostic = QGroupBox("Diagnostic overlay")
        diagnostic_layout = QFormLayout(diagnostic)
        self.diagnostic_overlay = QComboBox()
        self.diagnostic_overlay.addItems(("None", "Displacement", "Velocity", "Acceleration"))
        self.diagnostic_overlay.currentTextChanged.connect(self.set_motion_layer)
        diagnostic_layout.addRow("Diagnostic overlay", self.diagnostic_overlay)
        layout.addWidget(diagnostic)

        self.advanced_debug = QGroupBox("Advanced debug")
        self.advanced_debug.setCheckable(True)
        self.advanced_debug.setChecked(False)
        advanced_body = QWidget()
        advanced_layout = QVBoxLayout(advanced_body)
        self.advanced_debug.toggled.connect(advanced_body.setVisible)
        advanced_body.setVisible(False)

        pooling = QGroupBox("Pooling")
        pooling_layout = QFormLayout(pooling)
        self.pooling_toggle = QCheckBox("Pooling debug")
        self.pool_bin = QSpinBox()
        self.pool_bin.setRange(0, 63)
        self.pooling_toggle.toggled.connect(self.refresh_view)
        self.pool_bin.valueChanged.connect(self.refresh_view)
        pooling_layout.addRow(self.pooling_toggle)
        pooling_layout.addRow("Pool bin", self.pool_bin)
        advanced_layout.addWidget(pooling)

        landmark_group = QGroupBox("3D landmarks")
        landmark_layout = QFormLayout(landmark_group)
        self.landmark_status = QLabel("No landmark mapping found")
        self.landmark_status.setWordWrap(True)
        self.landmark_toggle = QCheckBox("Markers")
        self.landmark_labels_toggle = QCheckBox("Labels")
        self.landmark_choice = QComboBox()
        self.landmark_size = QSpinBox()
        self.landmark_size.setRange(1, 30)
        self.landmark_size.setValue(10)
        color = QPushButton("Color")
        color.clicked.connect(self.choose_landmark_color)
        for control in (
            self.landmark_toggle,
            self.landmark_labels_toggle,
            self.landmark_choice,
            self.landmark_size,
        ):
            if hasattr(control, "toggled"):
                control.toggled.connect(self.refresh_view)
            else:
                control.currentTextChanged.connect(self.refresh_view) if isinstance(
                    control, QComboBox
                ) else control.valueChanged.connect(self.refresh_view)
        landmark_layout.addRow(self.landmark_status)
        landmark_layout.addRow(self.landmark_toggle)
        landmark_layout.addRow(self.landmark_labels_toggle)
        landmark_layout.addRow("Selection", self.landmark_choice)
        landmark_layout.addRow("Marker size", self.landmark_size)
        landmark_layout.addRow(color)
        advanced_layout.addWidget(landmark_group)

        self.mediapipe_group = QGroupBox("MediaPipe illustration")
        mediapipe_layout = QFormLayout(self.mediapipe_group)
        self.mediapipe_478_toggle = QCheckBox("478 points")
        self.mediapipe_478_toggle.setChecked(True)
        self.mediapipe_20_toggle = QCheckBox("Configured 20")
        self.mediapipe_20_toggle.setChecked(True)
        self.mediapipe_roi_toggle = QCheckBox("ROI centers")
        self.mediapipe_roi_toggle.setChecked(True)
        for control in (
            self.mediapipe_478_toggle,
            self.mediapipe_20_toggle,
            self.mediapipe_roi_toggle,
        ):
            control.toggled.connect(self.refresh_view)
        mediapipe_layout.addRow(self.mediapipe_478_toggle)
        mediapipe_layout.addRow(self.mediapipe_20_toggle)
        mediapipe_layout.addRow(self.mediapipe_roi_toggle)
        advanced_layout.addWidget(self.mediapipe_group)
        self.advanced_debug.setLayout(QVBoxLayout())
        self.advanced_debug.layout().addWidget(advanced_body)
        layout.addWidget(self.advanced_debug)

        return panel

    def _player(self) -> QWidget:
        panel = QFrame()
        layout = QHBoxLayout(panel)
        self.timeline = ActiveFrameSlider()
        self.timeline.setMinimumHeight(28)
        self.timeline.valueChanged.connect(self.set_frame_by_index)
        self.review_chart.frame_selected.connect(self.set_frame_by_index)
        for icon, name, callback in (
            (QStyle.StandardPixmap.SP_MediaSkipBackward, "First frame", self.first_frame),
            (QStyle.StandardPixmap.SP_MediaSeekBackward, "Previous frame", self.previous_frame),
            (QStyle.StandardPixmap.SP_MediaPlay, "Play", self.toggle_play),
            (QStyle.StandardPixmap.SP_MediaSeekForward, "Next frame", self.next_frame),
            (QStyle.StandardPixmap.SP_MediaSkipForward, "Last frame", self.last_frame),
        ):
            button = QToolButton()
            button.setIcon(self.style().standardIcon(icon))
            button.setToolTip(name)
            button.setAccessibleName(name)
            button.clicked.connect(callback)
            if name == "Play":
                self.play_button = button
            layout.addWidget(button)
        self.frame_label = QLabel("0 / 0")
        self.reference_label = QLabel("Ref —")
        self.active_label = QLabel("Onset–offset —")
        reference_button = QToolButton()
        reference_button.setText("Set ref")
        reference_button.clicked.connect(self.set_reference_to_current)
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
            for variant in sorted({sequence.variant for sequence in self.session.indices.values()}):
                variant_item = QTreeWidgetItem([variant.upper()])
                self.sequence_tree.addTopLevelItem(variant_item)
                for subject in sorted(
                    {
                        sequence.subject
                        for sequence in self.session.indices.values()
                        if sequence.variant == variant
                    }
                ):
                    subject_item = QTreeWidgetItem([subject])
                    variant_item.addChild(subject_item)
                    for sequence in sorted(
                        (
                            sequence
                            for sequence in self.session.indices.values()
                            if sequence.variant == variant and sequence.subject == subject
                        ),
                        key=lambda sequence: sequence.video,
                    ):
                        active = self.session.active_window(sequence)
                        timing = (
                            f" · onset F{active[0]}–offset F{active[1]}"
                            if active
                            else " · onset–offset unavailable"
                        )
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
        return self.session.indices.get(key) if isinstance(key, str) else None

    def load_selected_sequence(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        self.timer.stop()
        self._set_playing(False)
        self.snapshot = self.session.select(index.key)
        self._mesh_style = None
        self._wire_visible = None
        if self.snapshot.sequence is None:
            error = self.snapshot.sequence_error or "Unknown sequence error"
            self.sequence_status.setText(f"Invalid: {error}")
            self.info.setText(error)
            self.timeline.setRange(0, 0)
            self.plotter.clear()
            return
        self._apply_snapshot()
        self.reset_camera()

    def _apply_snapshot(self) -> None:
        sequence = self.snapshot.sequence
        index = self.snapshot.index
        assert sequence is not None and index is not None
        self._set_raw_video()
        self._set_landmarks()
        active = self.snapshot.active_frames
        self.sequence_status.clear()
        self.sequence_details.setText(self._metadata_text(index, active))
        self.timeline.setRange(0, len(sequence.frames) - 1)
        maximum_label = (
            f"{len(sequence.frames)} / {len(sequence.frames)} (frame {sequence.frames[-1]})"
        )
        self.frame_label.setFixedWidth(
            self.frame_label.fontMetrics().horizontalAdvance(maximum_label) + 8
        )
        active_start = (
            sequence.frames.index(active[0]) if active and active[0] in sequence.frames else None
        )
        active_end = (
            sequence.frames.index(active[1]) if active and active[1] in sequence.frames else None
        )
        self.timeline.set_markers(
            active_start, active_end, sequence.frames.index(sequence.reference_frame)
        )
        self.active_label.setText(
            f"Onset–offset F{active[0]}–F{active[1]}" if active else "Onset–offset —"
        )
        self.reference_label.setText(f"Ref F{sequence.reference_frame}")
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(sequence.frames.index(sequence.current_frame))
        self.frame_label.setText(
            f"{sequence.frames.index(sequence.current_frame) + 1} / {len(sequence.frames)} "
            f"(frame {sequence.current_frame})"
        )
        self._set_mediapipe()
        self._sync_review()
        self.refresh_view()

    def _metadata_text(self, index: SequenceIndex, active: tuple[int, int] | None) -> str:
        return "\n".join(
            (
                f"Label: {self.snapshot.metadata.get('label', 'unknown')}",
                f"Subject/video: {index.subject}/{index.video}",
                f"Onset–offset: F{active[0]}–F{active[1]}"
                if active
                else "Onset–offset: unavailable",
            )
        )

    # Solid mesh and wireframe have one control each, in the viewport toolbar.

    def _set_raw_video(self) -> None:
        sequence = self.snapshot.sequence
        raw_video = self.snapshot.raw_video
        assert sequence is not None
        self.raw_player.stop()
        if raw_video is None:
            self.raw_player.setSource(QUrl())
            self.raw_status.setText(self.snapshot.raw_video_error or "No raw video")
            return
        self.raw_player.setSource(QUrl.fromLocalFile(str(raw_video.path)))
        synced = min(len(sequence.frames), raw_video.frame_count)
        if raw_video.frame_count > len(sequence.frames):
            note = f"; raw tail F{synced + 1}–{raw_video.frame_count} unavailable in mesh"
        elif raw_video.frame_count < len(sequence.frames):
            note = f"; mesh F{synced + 1}–{len(sequence.frames)} has no raw frame"
        else:
            note = ""
        self.raw_status.setText(
            f"Locked: mesh F1–{synced} ↔ raw F1–{synced} at {raw_video.fps:g} FPS{note}"
        )

    def _set_landmarks(self) -> None:
        sequence = self.snapshot.sequence
        mapping = self.snapshot.landmark_mapping
        assert sequence is not None
        if mapping is None:
            self.landmark_status.setText(
                self.snapshot.landmark_error
                or f"No landmark mapping found for {sequence.index.variant}"
            )
            for control in (
                self.landmark_toggle,
                self.landmark_labels_toggle,
                self.landmark_choice,
                self.landmark_size,
            ):
                control.setEnabled(False)
            return
        for control in (
            self.landmark_toggle,
            self.landmark_labels_toggle,
            self.landmark_choice,
            self.landmark_size,
        ):
            control.setEnabled(True)
        self.landmark_status.setText(f"{len(mapping.landmarks)} mapped landmarks")
        with QSignalBlocker(self.landmark_choice):
            self.landmark_choice.clear()
            self.landmark_choice.addItem("All landmarks")
            self.landmark_choice.addItems(sorted(mapping.landmarks))

    def _set_mediapipe(self) -> None:
        self.mediapipe_group.setVisible(self.snapshot.mediapipe_record is not None)
        self.mediapipe_image.show_478 = self.mediapipe_478_toggle.isChecked()
        self.mediapipe_image.show_20 = self.mediapipe_20_toggle.isChecked()
        self.mediapipe_image.show_rois = self.mediapipe_roi_toggle.isChecked()
        self.mediapipe_image.point_size = self.landmark_size.value()
        self.mediapipe_image.set_frame(
            self.snapshot.mediapipe_image, self.snapshot.mediapipe_record
        )
        if self.snapshot.mediapipe_error:
            self.mediapipe_image.status = self.snapshot.mediapipe_error
            self.mediapipe_image.update()

    def _sync_review(self) -> None:
        sequence = self.snapshot.sequence
        if sequence is None:
            return
        self._render_review_chart()
        active = self.snapshot.active_frames
        active_text = f"F{active[0]}–F{active[1]}" if active else "unavailable"
        self.chart_status.setText(self.snapshot.chart_status or "")
        self.review_context.setText(
            f"{sequence.index.key} · current F{sequence.current_frame} · active window {active_text}"
        )

    def _render_review_chart(self) -> None:
        sequence = self.snapshot.sequence
        if sequence is None:
            return
        self.review_chart.set_data(
            tuple(sequence.frames),
            self.snapshot.chart_data,
            self.snapshot.active_frames,
            sequence.frames.index(sequence.current_frame),
            metric=self.motion_metric.currentText().lower(),
            status=self.snapshot.chart_status,
        )

    def set_reference_to_current(self) -> None:
        sequence = self.snapshot.sequence
        if sequence is None:
            return
        self.snapshot = self.session.set_reference(sequence.current_frame)
        self.reference_label.setText(f"Ref F{sequence.reference_frame}")
        self.timeline.set_markers(
            self.timeline.active_start, self.timeline.active_end, self.timeline.value()
        )
        self.refresh_view()

    def set_motion_layer(self, label: str) -> None:
        self.motion_layer = None if label == "None" else label.lower()
        self.refresh_view()

    def set_frame_by_index(self, frame_index: int) -> None:
        sequence = self.snapshot.sequence
        if sequence is None:
            return
        try:
            self.snapshot = self.session.set_frame(sequence.frames[frame_index])
        except ValueError as exc:
            self.sequence_status.setText(f"Invalid: {exc}")
            self.timer.stop()
            self._set_playing(False)
            return
        sequence = self.snapshot.sequence
        assert sequence is not None
        if not self._advancing:
            self._seek_raw_frame(resume=self.timer.isActive())
        self.frame_label.setText(
            f"{frame_index + 1} / {len(sequence.frames)} (frame {sequence.current_frame})"
        )
        self._set_mediapipe()
        self._sync_review()
        self.refresh_view()

    def _seek_raw_frame(self, *, resume: bool = False) -> None:
        sequence = self.snapshot.sequence
        raw_video = self.snapshot.raw_video
        if sequence is None or raw_video is None or sequence.current_frame > raw_video.frame_count:
            return
        self.raw_player.pause()
        self.raw_player.setPosition(raw_video.position_ms(sequence.current_frame))
        if resume:
            self.raw_player.setPlaybackRate(self._raw_playback_rate())
            self.raw_player.play()

    def refresh_view(self) -> None:
        if self.sequence is None:
            return
        mesh = self.sequence.mesh
        diagnostics = self.sequence.diagnostics()
        scalars = diagnostics.get(self.motion_layer) if self.motion_layer else None
        clim = (
            diagnostic_color_limits(
                self.sequence.diagnostic_percentile(self.motion_layer),
                auto=self.auto_contrast_toggle.isChecked(),
            )
            if scalars is not None and self.motion_layer is not None
            else None
        )
        if scalars is not None:
            if "diagnostic_motion" in mesh.point_data:
                mesh.point_data["diagnostic_motion"][:] = scalars
            else:
                mesh.point_data["diagnostic_motion"] = scalars
            mesh.GetPointData().GetArray("diagnostic_motion").Modified()
            mesh.Modified()
        mesh_style = (
            self.solid_toggle.isChecked(),
            scalars is not None,
            self.original_color_toggle.isChecked(),
            clim,
        )
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
                    assert clim is not None
                    mesh_args.update(
                        scalars="diagnostic_motion",
                        cmap="turbo",
                        clim=clim,
                        show_scalar_bar=True,
                        scalar_bar_args=diagnostic_scalar_bar_args(
                            self.motion_layer.title(),
                            clim,
                            auto=self.auto_contrast_toggle.isChecked(),
                        ),
                    )
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
                self.plotter.add_mesh(
                    mesh,
                    name="wireframe",
                    style="wireframe",
                    color="#111111",
                    line_width=1,
                )
            self._wire_visible = wire_visible
        self._update_pooling(mesh)
        self._update_landmarks(mesh)
        self._set_mediapipe()
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
        self.plotter.add_mesh(
            pv.PolyData(mesh.points[indices]),
            name="pooling-bin",
            color="#ff00ff",
            point_size=5,
            render_points_as_spheres=True,
        )
        self._pooling_response = response

    def _update_landmarks(self, mesh: pv.PolyData) -> None:
        for name in ("landmarks", "landmark-labels", "landmark-trajectories"):
            self.plotter.remove_actor(name, render=False)
        if (
            self.landmark_mapping is None
            or not self.landmark_toggle.isChecked()
            or self.sequence is None
        ):
            return
        selection = self.landmark_choice.currentText()
        items = (
            list(self.landmark_mapping.landmarks.items())
            if selection == "All landmarks"
            else [(selection, self.landmark_mapping.landmarks[selection])]
        )
        names, indices = zip(*items, strict=True)
        points = mesh.points[list(indices)]
        reference_points = self.sequence.reference_points(indices)
        self.plotter.add_mesh(
            pv.PolyData(points),
            name="landmarks",
            color=self.landmark_color.name(),
            point_size=self.landmark_size.value(),
            render_points_as_spheres=True,
        )
        line_cells = np.concatenate(
            [np.array([2, 2 * number, 2 * number + 1]) for number in range(len(indices))]
        )
        paths = pv.PolyData(np.vstack((reference_points, points)), lines=line_cells)
        self.plotter.add_mesh(
            paths,
            name="landmark-trajectories",
            color=self.landmark_color.name(),
            line_width=2,
        )
        if self.landmark_labels_toggle.isChecked():
            self.plotter.add_point_labels(
                points,
                list(names),
                name="landmark-labels",
                font_size=10,
                text_color="white",
                shape=None,
                always_visible=True,
            )

    def _update_inspector(
        self, diagnostics: dict[str, np.ndarray | None], scalars: np.ndarray | None
    ) -> None:
        sequence = self.snapshot.sequence
        assert sequence is not None
        points = sequence.mesh.points
        active = self.snapshot.active_frames
        prediction = self.snapshot.prediction
        motion = [
            f"{name}: unavailable"
            if values is None
            else f"{name}: max={values.max():.6g}, mean={values.mean():.6g}"
            for name, values in diagnostics.items()
        ]
        if self.pooling_toggle.isChecked() and hasattr(self, "_pooling_response"):
            motion.append(
                f"pooling debug bin {self.pool_bin.value()}: max response={self._pooling_response:.6g}"
            )
        summary = (
            "Prediction mapping unavailable"
            if prediction is None
            else f"Prediction: {prediction['true_label']} → {prediction['predicted_label']} "
            f"({'correct' if prediction['correct'] == 'True' else 'wrong'})"
        )
        warning = "\n".join(self.snapshot.prediction_warnings)
        metric = self.snapshot.metrics
        metric_text = (
            f"Metrics: accuracy={metric.get('accuracy', 'n/a')}, "
            f"UAR={metric.get('uar', 'n/a')}, UF1={metric.get('uf1', 'n/a')}"
        )
        self.info.setText(
            "\n".join(
                (
                    f"Sequence: {sequence.index.key}",
                    f"Current: F{sequence.current_frame}; reference: F{sequence.reference_frame}",
                    f"File: {sequence.index.asset_path(sequence.current_frame, 'vertices')}",
                    f"Vertices/faces: {sequence.mesh.n_points}/{sequence.mesh.n_cells}",
                    f"Bounds: min={points.min(axis=0).round(5).tolist()} "
                    f"max={points.max(axis=0).round(5).tolist()}",
                    f"Active window: F{active[0]}–F{active[1]}"
                    if active
                    else "Active window: unavailable",
                    *motion,
                    summary,
                    metric_text,
                    warning,
                )
            )
        )

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
        return (
            1.0
            if self.raw_video is None
            else self.fps_spin.value() * self.speed_spin.value() / self.raw_video.fps
        )

    def update_timer_interval(self) -> None:
        self.timer.setInterval(round(1000 / (self.fps_spin.value() * self.speed_spin.value())))
        if self.timer.isActive() and self.raw_video is not None:
            self.raw_player.setPlaybackRate(self._raw_playback_rate())

    def _set_playing(self, playing: bool) -> None:
        icon = (
            QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        )
        name = "Pause" if playing else "Play"
        self.play_button.setIcon(self.style().standardIcon(icon))
        self.play_button.setToolTip(name)
        self.play_button.setAccessibleName(name)

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

    def _snapshot_paths(self, suffix: str = "") -> tuple[Path, Path]:
        sequence = self.sequence
        if sequence is None:
            base = Path.cwd() / "meviewer.png"
            return base, base.with_name(f"meviewer{suffix}.png")
        directory = Path.cwd() / ".snapshot" / sequence.index.key.replace("/", "-")
        stem = f"{sequence.index.subject}-{sequence.index.video}_frame{sequence.current_frame:03d}_meviewer"
        return directory / f"{stem}.png", directory / f"{stem}{suffix}.png"

    def _save_snapshot(self, title: str, suffix: str, save: Any) -> None:
        base, _ = self._snapshot_paths(suffix)
        path, _ = QFileDialog.getSaveFileName(self, title, str(base), "PNG image (*.png)")
        if path:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if save(output):
                self._export_raw_snapshot(output.with_name(f"{output.stem}-raw.png"))

    def _export_raw_snapshot(self, output: Path) -> None:
        raw, sequence = self.raw_video, self.sequence
        if raw is None or sequence is None or not 1 <= sequence.current_frame <= raw.frame_count:
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.sequence_status.setText("Paired raw PNG not written: ffmpeg is unavailable")
            return
        try:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(raw.path),
                    "-vf",
                    f"select=eq(n\\,{sequence.current_frame - 1})",
                    "-frames:v",
                    "1",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            self.sequence_status.setText(f"Paired raw PNG not written: cannot run ffmpeg ({exc})")
            return
        if result.returncode:
            self.sequence_status.setText(
                f"Paired raw PNG not written: ffmpeg failed ({result.returncode})"
            )

    def screenshot(self) -> None:
        self._save_snapshot(
            "Save screenshot", "", lambda path: self.plotter.screenshot(str(path)) is not None
        )

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
        self.plotter.render()

    def choose_landmark_color(self) -> None:
        color = QColorDialog.getColor(self.landmark_color, self, "Landmark color")
        if color.isValid():
            self.landmark_color = color
            self.refresh_view()

    def export_review(self) -> None:
        self._save_snapshot(
            "Export review PNG", "-review", lambda path: self.review_pane.grab().save(str(path))
        )

    def closeEvent(self, event: Any) -> None:
        self.timer.stop()
        self.raw_player.stop()
        self.plotter.close()
        super().closeEvent(event)
