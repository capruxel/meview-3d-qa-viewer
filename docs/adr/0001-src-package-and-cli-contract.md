# Source package and CLI contract

**Status:** accepted

Production code lives in `src/meviewer/`: domain modules at the package root, `viewer/` for mesh runtime, session, Qt widgets, and Qt window adapter, and `cli/` for command orchestration. The project is an installable package with one supported `meviewer` entry point; the former root launcher files are removed and must not be recreated as compatibility aliases.

The supported commands are:

```text
meviewer audit ...
meviewer extract ...
meviewer viewer ...
```

Subcommand parameters, exit codes, stdout/stderr, and the audit JSON schema remain stable. Dispatch is lazy: audit and extraction do not import Qt, PyVista, or optional MediaPipe runtime code. The viewer CLI only parses arguments and delegates to `meviewer.viewer.window`; viewer widgets and mesh/session logic have one implementation each.

Python imports remain internal implementation details, except for the documented package seams `meviewer.audit`, `meviewer.assets`, `meviewer.mediapipe`, and `meviewer.viewer.session`. This supersedes the original launcher-stability wording and makes the unified installed entry point the only public CLI contract.
