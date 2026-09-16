"""Command-line audit for MEVIEW mesh assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meviewer.assets import load_active_frames, scan_sequences
from meviewer.audit import audit_sequence, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--active-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks-root", type=Path)
    args = parser.parse_args()
    try:
        active_frames = load_active_frames(args.active_frames)
        records = [
            audit_sequence(sequence, active_frames, args.landmarks_root)
            for sequence in scan_sequences(args.data_root)
        ]
    except (OSError, ValueError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    summary, baseline_errors = summarize(records)
    payload = {"summary": summary, "baseline_errors": baseline_errors, "sequences": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {summary['sequence_count']} sequences, valid={summary['valid']}")
    return 0 if summary["valid"] else 1
