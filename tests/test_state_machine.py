"""Tests for the debounce state machine."""

import pytest

from domain.models import GazeState
from domain.state_machine import StateMachine


def test_initial_state():
    sm = StateMachine(stable_ms=200.0)
    assert sm.current_state == GazeState.UNKNOWN


def test_stable_transition_commits():
    sm = StateMachine(stable_ms=200.0)
    t = 0.0
    sm.reset(t)

    # Feed IN_COCKPIT for 300 ms – should commit after 200 ms
    result_before = sm.update(GazeState.IN_COCKPIT, t + 0.050)
    assert result_before == GazeState.UNKNOWN  # not yet stable

    result_after = sm.update(GazeState.IN_COCKPIT, t + 0.250)
    assert result_after == GazeState.IN_COCKPIT


def test_flicker_does_not_commit():
    sm = StateMachine(stable_ms=200.0)
    t = 0.0
    sm.reset(t)

    # Alternate between states faster than stable_ms
    sm.update(GazeState.IN_COCKPIT, t + 0.050)
    sm.update(GazeState.OUT_OF_COCKPIT, t + 0.100)  # reset pending
    sm.update(GazeState.IN_COCKPIT, t + 0.150)      # reset again

    # Still UNKNOWN because nothing was stable for 200 ms
    assert sm.current_state == GazeState.UNKNOWN


def test_transition_callback_fires():
    sm = StateMachine(stable_ms=100.0)
    t = 0.0
    sm.reset(t)
    fired = []
    sm.set_on_transition(lambda ev: fired.append(ev))

    # Call 1: starts the pending timer at t=0.05
    sm.update(GazeState.IN_COCKPIT, t + 0.05)
    # Call 2: 150 ms elapsed since pending started (0.20 - 0.05 = 0.15 s = 150 ms > 100 ms)
    # Using 0.20 instead of 0.15 to avoid floating-point precision edge cases.
    sm.update(GazeState.IN_COCKPIT, t + 0.20)

    assert len(fired) == 1
    assert fired[0].to_state == GazeState.IN_COCKPIT


def test_force_end_segment():
    sm = StateMachine(stable_ms=100.0)
    t = 0.0
    sm.reset(t)

    # Two calls required: first starts the pending timer, second commits it.
    sm.update(GazeState.IN_COCKPIT, t + 0.05)   # start pending
    sm.update(GazeState.IN_COCKPIT, t + 0.20)   # commit (150 ms > 100 ms)
    assert sm.current_state == GazeState.IN_COCKPIT

    # Now close the open segment; segment started when transition committed at t=0.20
    last_ev = sm.force_end_segment(t + 0.50)
    assert last_ev is not None
    assert last_ev.from_state == GazeState.IN_COCKPIT
    # Segment ran from t=0.20 to t=0.50 → 300 ms
    assert abs(last_ev.duration_ms - 300.0) < 5.0
