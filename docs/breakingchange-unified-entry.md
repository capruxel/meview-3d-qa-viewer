# Breaking Change: Unified `meviewer` Entry Point

## Decision

Replace the three root launchers with one package entry point:

```text
meviewer audit ...
meviewer extract ...
meviewer viewer ...
```

The package entry point dispatches lazily to the command implementations. Lazy
dispatch is required because `audit` and `extract` must remain usable without
importing Qt, PyVista, or optional MediaPipe runtime code. No root launcher
files are retained.

## Why unify

The former interface exposed three filenames as separate commands:

- `audit_dataset.py`
- `extract_mediapipe.py`
- `mesh_viewer.py`

Those files were removed rather than retained as aliases, because aliases keep
the fragmented command surface discoverable and make accidental use likely.
The single `meviewer` command gives users one discoverable namespace while
preserving separate domain modules internally.

This is a CLI consolidation, not a runtime consolidation. The audit, extraction, and viewer seams remain separate because they have different dependencies and failure modes.

Subcommand options remain unchanged after the subcommand name. Exit codes, JSON
fields, stdout text, and stderr text remain unchanged within each command.

## Breaking changes

1. The three root script filenames are removed.
2. Direct execution of the former root script paths is unsupported because the
   files no longer exist.
3. Shell scripts and CI jobs must use the corresponding `meviewer` subcommand.
4. Any code importing former root modules must import package modules instead.
   In particular, `viewer_session.py` is removed; use
   `meviewer.viewer.session`.
5. The new command requires an installed package entry point. Running source
   files directly from an unpackaged checkout is not a supported workflow.

## Non-breaking guarantees

The following behavior is intentionally retained per subcommand:

- required and optional flags;
- `--variant` filtering;
- `--overwrite` behavior;
- audit JSON schema;
- success and error messages;
- exit codes `0`, `1`, and `2` where currently defined;
- optional MediaPipe import timing;
- viewer `--self-test` behavior;
- no `QApplication` creation for help or self-test argument handling.

## Migration examples

```bash
# Audit
meviewer audit \
  --data-root /path/to/data \
  --active-frames /path/to/active_frames.json \
  --output /tmp/audit.json

# MediaPipe extraction
uv run --extra landmarks meviewer extract \
  --data-root /path/to/data \
  --output-root /path/to/records \
  --model /path/to/face_landmarker.task

# Viewer
meviewer viewer \
  --data-root /path/to/data \
  --active-frames /path/to/active_frames.json
```

## Implementation constraints

- Add one `project.scripts` entry: `meviewer = "meviewer.cli.main:main"`.
- Dispatch subcommands before importing their implementation modules.
- Keep `meviewer.audit`, `meviewer.assets`, `meviewer.mediapipe`, and `meviewer.viewer.session` as internal package seams.
- Do not add a second copy of command parsing logic.
- Remove the root launchers after downstream migration.
- Update README, ADRs, tests, and hook commands in the same breaking-change commit.


## Rejected alternatives

### Keep three commands permanently

This avoids migration but retains the fragmented public surface and makes future packaging less coherent.

### One command with global flags

This mixes unrelated options and makes invalid combinations easier to express. Subcommands keep each contract local.

### Move all implementation into one module

This would unify files by increasing coupling. The correct boundary is one CLI namespace over independent domain and adapter modules.
