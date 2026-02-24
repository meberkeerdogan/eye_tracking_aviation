"""Tests for debrief metrics computation."""

import pytest

from domain.metrics import compute_debrief
from domain.models import GazeState, GazeSample, StateEvent


def _sample(t: float, state: GazeState) -> GazeSample:
    return GazeSample(
        timestamp_mono=t,
        timestamp_wall=t + 1000,
        gaze_x_norm=0.5,
        gaze_y_norm=0.5,
        confidence=0.9,
        state=state,
    )


def test_empty_session():
    result = compute_debrief([], [], 0.0)
    assert result["total_duration_s"] == 0.0
    assert result["n_out_glances"] == 0


def test_all_in_cockpit():
    # One event: from UNKNOWN → IN_COCKPIT, lasting 10 s
    events = [
        StateEvent(
            from_state=GazeState.UNKNOWN,
            to_state=GazeState.IN_COCKPIT,
            start_time=0.0,
            end_time=1.0,
        ),
        StateEvent(
            from_state=GazeState.IN_COCKPIT,
            to_state=GazeState.IN_COCKPIT,  # closing segment
            start_time=1.0,
            end_time=11.0,
        ),
    ]
    samples = [_sample(float(i), GazeState.IN_COCKPIT) for i in range(11)]
    result = compute_debrief(samples, events, session_duration_s=11.0)

    assert result["in_cockpit_s"] == pytest.approx(10.0, abs=0.01)
    assert result["n_out_glances"] == 0
    assert result["out_durations_ms"] == []


def test_out_glance_counted():
    events = [
        # 5 s in cockpit
        StateEvent(GazeState.UNKNOWN, GazeState.IN_COCKPIT, 0.0, 0.5),
        StateEvent(GazeState.IN_COCKPIT, GazeState.OUT_OF_COCKPIT, 0.5, 5.5),
        # 2 s out
        StateEvent(GazeState.OUT_OF_COCKPIT, GazeState.IN_COCKPIT, 5.5, 7.5),
        # closing
        StateEvent(GazeState.IN_COCKPIT, GazeState.IN_COCKPIT, 7.5, 10.0),
    ]
    samples = [_sample(float(i), GazeState.IN_COCKPIT) for i in range(11)]
    result = compute_debrief(samples, events, session_duration_s=10.0)

    assert result["n_out_glances"] == 1
    assert result["out_durations_ms"] == pytest.approx([2000.0], abs=1.0)
    assert result["avg_out_ms"] == pytest.approx(2000.0, abs=1.0)
    assert result["max_out_ms"] == pytest.approx(2000.0, abs=1.0)
