"""Core data models for the eye-tracking aviation application."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GazeState(str, Enum):
    IN_COCKPIT = "IN_COCKPIT"
    OUT_OF_COCKPIT = "OUT_OF_COCKPIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class GazeSample:
    timestamp_mono: float  # time.monotonic()
    timestamp_wall: float  # time.time()
    gaze_x_norm: float     # 0-1 normalized to session widget width
    gaze_y_norm: float     # 0-1 normalized to session widget height
    confidence: float      # 0-1
    state: GazeState


@dataclass
class StateEvent:
    """A committed state transition with timing."""

    from_state: GazeState
    to_state: GazeState
    start_time: float  # monotonic seconds (start of the from_state segment)
    end_time: float    # monotonic seconds (when transition committed)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class SessionMarker:
    timestamp_mono: float
    timestamp_wall: float
    label: str = "marker"


@dataclass
class SessionMeta:
    session_id: str
    mode: str          # "debug" or "test"
    started_at: str    # ISO8601
    ended_at: Optional[str] = None
    screen_width: int = 0
    screen_height: int = 0
    camera_index: int = 0
    camera_width: int = 0
    camera_height: int = 0
    calibration_hash: str = ""
    profile_name: str = "default"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class VisionResult:
    """Result emitted by the vision worker for each processed frame."""

    sample: GazeSample
    face_detected: bool = False
    auto_paused: bool = False
    # Optional pixel coords within the session widget (for drawing overlay)
    gaze_px_x: Optional[float] = None
    gaze_px_y: Optional[float] = None
    # Normalised iris positions (0-1 of camera frame) for debug overlay
    left_iris_norm: Optional[tuple[float, float]] = None
    right_iris_norm: Optional[tuple[float, float]] = None
