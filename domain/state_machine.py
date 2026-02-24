"""Debounce / hysteresis state machine for gaze classification."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from domain.models import GazeState, StateEvent

logger = logging.getLogger(__name__)


class StateMachine:
    """Only commits a state transition after the candidate has been stable for
    *stable_ms* milliseconds.  This prevents rapid flickering in the metrics."""

    def __init__(self, stable_ms: float = 200.0) -> None:
        self.stable_ms = stable_ms

        self._committed: GazeState = GazeState.UNKNOWN
        self._segment_start: float = 0.0  # monotonic

        self._pending: Optional[GazeState] = None
        self._pending_since: Optional[float] = None

        self._events: list[StateEvent] = []
        self._on_transition: Optional[Callable[[StateEvent], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, mono_time: float) -> None:
        self._committed = GazeState.UNKNOWN
        self._segment_start = mono_time
        self._pending = None
        self._pending_since = None
        self._events.clear()

    def update(self, candidate: GazeState, mono_time: float) -> GazeState:
        """Feed a new raw classification; return the *committed* state."""
        if candidate == self._committed:
            # Still in same state – reset the pending buffer
            self._pending = None
            self._pending_since = None
            return self._committed

        if candidate != self._pending:
            # New candidate starts its stability timer
            self._pending = candidate
            self._pending_since = mono_time
            return self._committed

        # Same candidate is accumulating – check stability duration
        assert self._pending_since is not None
        elapsed_ms = (mono_time - self._pending_since) * 1000.0
        if elapsed_ms >= self.stable_ms:
            self._commit(candidate, mono_time)

        return self._committed

    def force_end_segment(self, mono_time: float) -> Optional[StateEvent]:
        """Close the current open segment (call when session ends)."""
        if mono_time <= self._segment_start:
            return None
        event = StateEvent(
            from_state=self._committed,
            to_state=self._committed,  # same – just closing the segment
            start_time=self._segment_start,
            end_time=mono_time,
        )
        self._events.append(event)
        return event

    def set_on_transition(self, callback: Callable[[StateEvent], None]) -> None:
        self._on_transition = callback

    @property
    def current_state(self) -> GazeState:
        return self._committed

    @property
    def events(self) -> list[StateEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _commit(self, new_state: GazeState, mono_time: float) -> None:
        event = StateEvent(
            from_state=self._committed,
            to_state=new_state,
            start_time=self._segment_start,
            end_time=mono_time,
        )
        self._events.append(event)
        logger.debug(
            "State: %s → %s  (%.0f ms)",
            self._committed.value,
            new_state.value,
            event.duration_ms,
        )
        if self._on_transition:
            self._on_transition(event)

        self._committed = new_state
        self._segment_start = mono_time
        self._pending = None
        self._pending_since = None
