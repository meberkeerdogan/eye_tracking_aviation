"""Compute debrief statistics from session data."""

from __future__ import annotations

import statistics
from typing import Any

from domain.models import GazeState, GazeSample, StateEvent


def compute_debrief(
    samples: list[GazeSample],
    events: list[StateEvent],
    session_duration_s: float,
) -> dict[str, Any]:
    """Return a flat dict of debrief metrics suitable for JSON serialisation."""

    # ── Time per state (from committed transition events) ──────────────────
    # Each event records the duration spent *in* event.from_state.
    # The last (still-open) segment is NOT represented in events; it is
    # closed by StateMachine.force_end_segment before this function is called.
    state_durations_s: dict[GazeState, float] = {s: 0.0 for s in GazeState}
    for ev in events:
        state_durations_s[ev.from_state] += ev.duration_ms / 1000.0

    in_s = state_durations_s[GazeState.IN_COCKPIT]
    out_s = state_durations_s[GazeState.OUT_OF_COCKPIT]
    unk_s = state_durations_s[GazeState.UNKNOWN]

    total = session_duration_s if session_duration_s > 0 else 1.0

    # ── Out-of-cockpit glance segments ────────────────────────────────────
    # Duration of each segment spent IN OUT_OF_COCKPIT state (from_state=OUT_OF_COCKPIT)
    out_segments_ms: list[float] = [
        ev.duration_ms for ev in events if ev.from_state == GazeState.OUT_OF_COCKPIT
    ]
    # Number of times we *entered* OUT_OF_COCKPIT
    n_out_glances = sum(1 for ev in events if ev.to_state == GazeState.OUT_OF_COCKPIT)

    # ── Per-sample confidence stats ────────────────────────────────────────
    conf_values = [s.confidence for s in samples if s.confidence > 0]
    avg_conf = statistics.mean(conf_values) if conf_values else 0.0

    # ── Timeline for replay / charts (downsampled to ~10 Hz) ──────────────
    timeline: list[dict[str, Any]] = []
    if samples:
        t0 = samples[0].timestamp_mono
        timeline = [
            {
                "t_s": round(s.timestamp_mono - t0, 3),
                "gx": round(s.gaze_x_norm, 4),
                "gy": round(s.gaze_y_norm, 4),
                "state": s.state.value,
            }
            for s in samples[::3]  # every 3rd sample ≈ 10 Hz at 30 fps
        ]

    return {
        "total_duration_s": round(session_duration_s, 3),
        "in_cockpit_s": round(in_s, 3),
        "out_cockpit_s": round(out_s, 3),
        "unknown_s": round(unk_s, 3),
        "in_cockpit_pct": round(in_s / total * 100, 1),
        "out_cockpit_pct": round(out_s / total * 100, 1),
        "unknown_pct": round(unk_s / total * 100, 1),
        "n_out_glances": n_out_glances,
        "out_durations_ms": [round(d, 1) for d in out_segments_ms],
        "avg_out_ms": round(statistics.mean(out_segments_ms), 1) if out_segments_ms else 0.0,
        "median_out_ms": round(statistics.median(out_segments_ms), 1) if out_segments_ms else 0.0,
        "max_out_ms": round(max(out_segments_ms), 1) if out_segments_ms else 0.0,
        "total_samples": len(samples),
        "avg_confidence": round(avg_conf, 3),
        "timeline": timeline,
    }
