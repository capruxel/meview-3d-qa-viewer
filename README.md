# MEVIEW 3D QA Viewer

Standalone dataset audit and PySide6/PyVista viewer for MEVIEW V2/V3 dense-mesh sequences.

The viewer does not run 3DDFA reconstruction and does not regenerate
`kinematic_features_v2.npy` or `kinematic_features_v3.npy`. `*_frontal_vertices.npy`
files provide mesh coordinates, OBJ files provide topology and per-vertex color, and
illustration JPG files are checked/displayed but never enter kinematics. The
`active_frames.json` is external metadata supplied by the caller; its onset and
offset values are inclusive.

Diagnostic displacement, velocity, acceleration, and pooling layers are QA views,
not the Notebook's 3456-dimensional feature vector.

## Usage

This repository runs independently. It has no dependency on a specific host
repository; provide the dataset and metadata paths explicitly.

```bash
uv sync
uv run python audit_dataset.py \
  --data-root /path/to/data/dataset \
  --active-frames /path/to/data/active_frames.json \
  --output /tmp/meview-dataset-audit.json
uv run python mesh_viewer.py \
  --data-root /path/to/data/dataset \
  --active-frames /path/to/data/active_frames.json \
  --results-dir /path/to/results \
  --landmarks-root /path/to/landmarks
```

`--results-dir` and `--landmarks-root` are optional. The viewer remains usable
as mesh-only QA when either source is absent. Use `--self-test` to run the
data-path checks without opening the UI:

```bash
uv run python mesh_viewer.py \
  --data-root /path/to/data/dataset \
  --active-frames /path/to/data/active_frames.json \
  --self-test
```

The dataset must contain the MEVIEW V2/V3 mesh assets expected by the audit
tool. Raw-video timing additionally requires the system `ffprobe` executable.
Dataset files, results, raw videos, and release archives are intentionally not
vendored in this repository.
