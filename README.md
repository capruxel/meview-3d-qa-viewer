# MEVIEW 3D QA Viewer

Standalone dataset audit and PySide6/PyVista viewer for MEVIEW V2/V3 and
LFANN V3 dense-mesh sequences.

The viewer does not run 3DDFA reconstruction. `*_frontal_vertices.npy`
files provide mesh coordinates, OBJ files provide topology and required per-vertex
color, and illustration JPG files are displayed in the MediaPipe tab. The
`active_frames.json` is external metadata supplied by the caller; its onset
and offset values are inclusive. The teacher's `frontal_meshes_MEVIEW_v3` and
LFANN's `frontal_meshes_LFANN_v3` are separate dataset variants.

Diagnostic displacement, velocity, acceleration, and pooling layers are QA views,
not the Notebook's 3456-dimensional feature vector.

## Interface preview

The screenshot below shows the viewer with the sequence browser, mesh-locked raw
video, 3D mesh viewport, active-frame timeline, and playback controls.

![MEVIEW 3D QA Viewer interface](assets/meview-3d-qa-viewer.webp)

## Usage

This repository runs independently. It has no dependency on a specific host
repository; provide the dataset and metadata paths explicitly. Run the commands
from this repository root. Requires Python 3.12.x and `uv`.

The unified CLI is the supported entry point:

```bash
meviewer audit --data-root /path/to/data/dataset \
  --active-frames /path/to/data/active_frames.json \
  --output /tmp/meview-dataset-audit.json
meviewer extract --data-root /path/to/data/dataset \
  --output-root /path/to/records \
  --model /path/to/face_landmarker.task
meviewer viewer --data-root /path/to/data/dataset \
  --active-frames /path/to/data/active_frames.json \
  --annotation-output /path/to/review/annotations.json
```

For LFANN, extract viewer-format regional-motion evidence from the mesh root,
then supply the same output root to the viewer:

```bash
uv run meviewer extract \
  --data-root "$LFANN_MESH_ROOT" \
  --variant lfann-v3 \
  --output-root "$MEDIAPIPE_ROOT" \
  --model "$MEDIAPIPE_MODEL" \
  --config assets/lfann-regional-motion-landmarks.json

uv run meviewer viewer \
  --data-root /path/to/data/dataset \
  --mediapipe-root "$MEDIAPIPE_ROOT" \
  --active-frames /path/to/data/active_frames.json \
  --annotation-output /path/to/review/annotations.json
```

Regional-motion annotations are read from and written only to the explicit
`--annotation-output` path. Review PNGs are written only after selecting an
exact path in the export dialog.

| Input | Required | Resolution |
|---|---|---|
| `--data-root` | Yes | Dataset root. |
| `--mesh-root`, `--active-frames` | Active frames: yes | Direct argument overrides `[viewer]`; otherwise manifest values apply. `--mesh-root` falls back to `--data-root`. |
| `--raw-root` | No | Enables mesh-locked raw-video playback when matching MP4s and `ffprobe` exist. |
| `--mediapipe-root` | No | Enables MediaPipe illustrations and regional-motion chart evidence. |
| `--results-dir`, `--landmarks-root` | No | Add prediction and landmark diagnostics; mesh-only review remains available. |
| `--annotation-output` | Interactive mode | The sole annotation read/write path; `--self-test` neither opens Qt nor requires it. |

Direct arguments override `[viewer]` TOML defaults. A manifest supplies `mesh_root` and
`active_frames` only when their direct arguments are omitted.

The root launcher files were removed. Use the installed `meviewer` command;
see `docs/breakingchange-unified-entry.md` for the migration contract.

## Development checks

Ruff is pinned as a development dependency and configured in `pyproject.toml`.
Run the checks manually with:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run the test suite with pytest:

```bash
uv run pytest
```

This repository uses `prek` for Git hooks. It is included in the development
dependencies, so `uv sync` installs it with Ruff:

```bash
uv sync
uv run prek install --prepare-hooks
uv run prek run --all-files
```

The hook runs merge-conflict, whitespace, EOF, Ruff lint, and Ruff format
checks before commits. Use `uv run ruff format .` to apply formatting changes.

`prek` is also available directly for manual checks:

```bash
uv run prek validate-config
```



The dataset must contain the MEVIEW V2/V3 mesh assets expected by the audit
tool. Dataset files, results, raw videos, and release archives are intentionally
not vendored in this repository.
