# MEVIEW 3D QA Viewer

Standalone dataset audit and PySide6/PyVista viewer for MEVIEW V2/V3 dense-mesh sequences.

The viewer does not run 3DDFA reconstruction and does not regenerate
`kinematic_features_v2.npy` or `kinematic_features_v3.npy`. `*_frontal_vertices.npy`
files provide mesh coordinates, OBJ files provide topology and per-vertex color, and
illustration JPG files are checked/displayed but never enter kinematics. The
`active_frames.json` file remains external metadata owned by the parent experiment
repository; its onset and offset values are inclusive.

Diagnostic displacement, velocity, acceleration, and pooling layers are QA views,
not the Notebook's 3456-dimensional feature vector. Raw-video timing requires the
system `ffprobe` executable. Dataset files, results, raw videos, and release archives
are intentionally not vendored here.

## Usage

From a parent repository containing `data/` and optionally `results/`/`landmarks/`:
```bash
uv sync
uv run python audit_dataset.py \
  --data-root ../data/dataset \
  --active-frames ../data/active_frames.json \
  --output /tmp/meview-dataset-audit.json
uv run python mesh_viewer.py \
  --data-root ../data/dataset \
  --active-frames ../data/active_frames.json \
  --results-dir ../results \
  --landmarks-root ../landmarks
```

The exact paths are caller-owned. Results and landmarks arguments are optional; the
viewer remains usable as mesh-only QA when either source is absent. `--self-test`
runs the data-path checks without opening the UI.

## Parent repository integration

When mounted as `kafe3d/viewer/`, run from the parent root:

```bash
uv sync --project viewer
uv run --project viewer python viewer/audit_dataset.py \
  --data-root data/dataset \
  --active-frames data/active_frames.json \
  --output /tmp/meview-dataset-audit.json
uv run --project viewer python viewer/mesh_viewer.py \
  --data-root data/dataset \
  --active-frames data/active_frames.json \
  --results-dir results \
  --landmarks-root landmarks
```

The parent repository pins a specific viewer commit through its Git submodule. To
update the viewer, commit and push changes in this repository first, then update the
parent repository's `viewer` gitlink.
