"""MediaPipe FaceMesh wrapper – returns iris positions and eye openness."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark indices (FaceMesh with refine_landmarks=True)
_LEFT_IRIS = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_OUTER = 362
_RIGHT_EYE_INNER = 263
_NOSE_TIP = 1


@dataclass
class FaceResult:
    """Processed output from one camera frame."""

    landmarks: object          # raw mediapipe landmark list
    left_iris: tuple[float, float]   # (x, y) normalised to frame 0-1
    right_iris: tuple[float, float]
    left_openness: float       # rough eye-openness ratio
    right_openness: float
    confidence: float          # derived from openness; 0-1
    nose_tip: tuple[float, float]


class FaceTracker:
    """Thin wrapper around MediaPipe FaceMesh.

    Must be used from a single thread (MediaPipe is not thread-safe).
    """

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.debug("FaceTracker initialised.")

    def process(self, frame_rgb: np.ndarray) -> Optional[FaceResult]:
        """Process a single RGB frame.  Returns ``None`` if no face found."""
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark

        def iris_center(indices: list[int]) -> tuple[float, float]:
            xs = [lms[i].x for i in indices]
            ys = [lms[i].y for i in indices]
            return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

        def openness(top_i: int, bot_i: int, outer_i: int, inner_i: int) -> float:
            vert = abs(lms[top_i].y - lms[bot_i].y)
            horiz = abs(lms[outer_i].x - lms[inner_i].x) + 1e-6
            return float(vert / horiz)

        left_iris = iris_center(_LEFT_IRIS)
        right_iris = iris_center(_RIGHT_IRIS)
        left_open = openness(_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM, _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
        right_open = openness(_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)

        # Confidence: higher when eyes are open.  Typical open-eye ratio ≈ 0.15
        avg_open = (left_open + right_open) / 2.0
        confidence = float(min(1.0, max(0.0, avg_open / 0.15)))

        nose = (float(lms[_NOSE_TIP].x), float(lms[_NOSE_TIP].y))

        return FaceResult(
            landmarks=lms,
            left_iris=left_iris,
            right_iris=right_iris,
            left_openness=left_open,
            right_openness=right_open,
            confidence=confidence,
            nose_tip=nose,
        )

    def close(self) -> None:
        self._face_mesh.close()
        logger.debug("FaceTracker closed.")
