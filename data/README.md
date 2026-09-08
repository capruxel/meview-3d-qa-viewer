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

- `<frame>_frontal_vertices.npy`: non-empty numeric vertex coordinates with
  shape `(3, N)`. Values must be finite; `N` must stay constant within a
  sequence and match the OBJ vertex count.
- `<frame>_frontal_mesh.obj`: fixed topology with one `v` line per vertex in
  `x y z r g b` format. The first OBJ establishes the topology; each frame's OBJ
  supplies the colors used by the viewer.
- `<frame>_frontal_illustration.jpg`: required for frame completeness checks;
  the viewer does not display this image.

Every numbered frame must contain all three assets. Missing any one of them
makes the sequence invalid for viewer loading. The audit command reports
missing or inconsistent assets instead of repairing them.

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
- `raw/me-cuts/cuts/<subject>-<integer-video>.mp4` enables mesh-locked raw-video
  playback. The video directory `01` maps to a file such as `sub01-1.mp4`.
  The system `ffprobe` executable must be available on `PATH`.
- `results/metrics.csv` and `results/predictions.csv` can be passed with
  `--results-dir`.
- `landmarks/v2.json` and `landmarks/v3.json` can be passed with
  `--landmarks-root` after a validated vertex mapping exists.

## Offline MediaPipe records

The optional landmarks extra can create normalized image-space records without
running detection during playback:

```bash
uv run --extra landmarks python extract_mediapipe.py \
  --data-root /path/to/data/dataset \
  --output-root /path/to/data/dataset/mediapipe \
  --model /path/to/face_landmarker.task \
  --variant v3 \
  --config /path/to/lfann_landmarks20.json
```

Records live under `mediapipe/<variant>/<subject>/<video>/<frame>.npz` and
contain `landmarks478`, optional configured `landmarks20`, `roi_centers`,
`roi_names`, and `status`. MediaPipe coordinates remain 2D illustration
coordinates; they are not mesh vertex indices.

All optional inputs can be omitted. Mesh audit and mesh-only playback remain
available when results, raw video, or landmark mappings are not present. Raw
video playback requires the matching MP4 and `ffprobe`.
