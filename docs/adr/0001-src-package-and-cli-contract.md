# Source package and CLI contract

**Status:** accepted

Production code lives in `src/meviewer/`: domain modules at the package root, `viewer/` for mesh runtime, session, Qt widgets, and Qt window adapter, and `cli/` for command orchestration. The project is an installable editable package; root `audit_dataset.py`, `extract_mediapipe.py`, and `mesh_viewer.py` remain thin compatibility launchers. The CLI parameters, exit codes, stdout/stderr, and audit JSON schema are the stable external interface because dependent projects invoke these commands; Python imports remain internal implementation details. This rejects `sys.path` launchers and a concurrent CLI redesign, which would mix package migration risk with an external-contract change.
