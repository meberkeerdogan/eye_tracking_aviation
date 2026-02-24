"""Safe, buffered writer for session data files."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

from domain.models import GazeSample, SessionMarker, SessionMeta, StateEvent

logger = logging.getLogger(__name__)

_SAMPLE_FIELDS = [
    "timestamp_mono", "timestamp_wall",
    "gaze_x_norm", "gaze_y_norm",
    "confidence", "state",
]
_EVENT_FIELDS = ["from_state", "to_state", "start_time", "end_time", "duration_ms"]
_MARKER_FIELDS = ["timestamp_mono", "timestamp_wall", "label"]


class SessionWriter:
    """Creates a session directory and writes CSV/JSON files with line-buffering
    so data is not lost if the process crashes."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        session_dir.mkdir(parents=True, exist_ok=True)

        # Open files in line-buffered mode (buffering=1 applies to text mode)
        self._sf = open(session_dir / "samples.csv", "w", newline="", buffering=1, encoding="utf-8")
        self._ef = open(session_dir / "events.csv", "w", newline="", buffering=1, encoding="utf-8")
        self._mf = open(session_dir / "markers.csv", "w", newline="", buffering=1, encoding="utf-8")

        self._sw = csv.DictWriter(self._sf, fieldnames=_SAMPLE_FIELDS)
        self._ew = csv.DictWriter(self._ef, fieldnames=_EVENT_FIELDS)
        self._mw = csv.DictWriter(self._mf, fieldnames=_MARKER_FIELDS)

        self._sw.writeheader()
        self._ew.writeheader()
        self._mw.writeheader()

        self._closed = False
        logger.info("SessionWriter opened at %s", session_dir)

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def write_sample(self, s: GazeSample) -> None:
        if self._closed:
            return
        self._sw.writerow(
            {
                "timestamp_mono": f"{s.timestamp_mono:.6f}",
                "timestamp_wall": f"{s.timestamp_wall:.6f}",
                "gaze_x_norm": f"{s.gaze_x_norm:.6f}",
                "gaze_y_norm": f"{s.gaze_y_norm:.6f}",
                "confidence": f"{s.confidence:.4f}",
                "state": s.state.value,
            }
        )

    def write_event(self, ev: StateEvent) -> None:
        if self._closed:
            return
        self._ew.writerow(
            {
                "from_state": ev.from_state.value,
                "to_state": ev.to_state.value,
                "start_time": f"{ev.start_time:.6f}",
                "end_time": f"{ev.end_time:.6f}",
                "duration_ms": f"{ev.duration_ms:.2f}",
            }
        )

    def write_marker(self, m: SessionMarker) -> None:
        if self._closed:
            return
        self._mw.writerow(
            {
                "timestamp_mono": f"{m.timestamp_mono:.6f}",
                "timestamp_wall": f"{m.timestamp_wall:.6f}",
                "label": m.label,
            }
        )

    def write_meta(self, meta: SessionMeta) -> None:
        _write_json(self.session_dir / "session_meta.json", meta.to_dict())

    def write_debrief(self, debrief: dict[str, Any]) -> None:
        _write_json(self.session_dir / "debrief.json", debrief)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sf.close()
        self._ef.close()
        self._mf.close()
        logger.info("SessionWriter closed.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except OSError as exc:
        logger.error("Failed to write %s: %s", path, exc)
