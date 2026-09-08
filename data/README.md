# Dataset layout

The viewer reads an external dataset through `--data-root`. The repository does
not include mesh, video, result, or landmark data.

## Expected layout

```text
data-root/
├── frontal_meshes_MEVIEW_v2/
│   └── sub01/
│       └── 01/
│           ├── 001_frontal_vertices.npy
│           ├── 001_frontal_mesh.obj
│           ├── 001_frontal_illustration.jpg
│           └── ...
├── frontal_meshes_MEVIEW_v3/
│   └── sub01/01/...
├── groups_v2.npy                 # optional sequence metadata
├── video_ids_v2.npy              # optional sequence metadata
├── labels_v2.npy                 # optional sequence metadata
├── groups_v3.npy
├── video_ids_v3.npy
├── labels_v3.npy
└── raw/
    └── me-cuts/
        └── cuts/
            ├── sub01-1.mp4
            └── ...
```

Each mesh sequence uses `variant/subject/video` as its identity. The viewer
recognizes these filename patterns inside each sequence directory:

- `<frame>_frontal_vertices.npy`: numeric vertex coordinates with shape `(3, N)`.
- `<frame>_frontal_mesh.obj`: fixed topology and optional per-vertex color.
- `<frame>_frontal_illustration.jpg`: frame illustration used for asset checks.

The OBJ topology must remain fixed within a sequence. The first OBJ establishes
the topology; each NPY updates only the mesh coordinates. The audit command
reports missing or inconsistent assets instead of repairing them.

## Active-frame metadata

`active_frames.json` is supplied separately with `--active-frames`. Its keys are
`<subject>_<video>` and its frame range is inclusive:

```json
{
  "sub01_01": {"onset": 48, "offset": 63}
}
```

The JSON file may live beside `data-root` or anywhere else. The viewer does not
infer active ranges from filenames.

## Optional inputs

- `groups_v2.npy`, `video_ids_v2.npy`, and `labels_v2.npy` (and the V3 equivalents)
  provide sequence labels in the inspector.
- `raw/me-cuts/cuts/<subject>-<video>.mp4` enables mesh-locked raw-video playback.
  The system `ffprobe` executable must be available.
- `results/metrics.csv` and `results/predictions.csv` can be passed with
  `--results-dir`.
- `landmarks/v2.json` and `landmarks/v3.json` can be passed with
  `--landmarks-root` after a validated vertex mapping exists.

All optional inputs can be omitted. Mesh audit and playback remain available
when results, raw video, or landmark mappings are not present.
