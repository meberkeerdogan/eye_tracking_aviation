"""Extract a fixed-length feature vector from a FaceResult for gaze regression."""

from __future__ import annotations

import numpy as np

from vision.face_tracker import FaceResult

# MediaPipe landmark indices reused here for eye-relative features
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOTTOM = 145
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOTTOM = 374
_RIGHT_EYE_OUTER = 362
_RIGHT_EYE_INNER = 263
_NOSE_TIP = 1
_CHIN = 152
_FOREHEAD = 10


def extract_features(face: FaceResult) -> np.ndarray:
    """Return a 1-D float64 feature vector suitable for the gaze regressor.

    All values are in normalised frame coordinates (0-1).  The vector is
    designed to be robust to small head-position changes by including
    iris positions relative to their eye corners (gaze direction proxy).
    """
    lms = face.landmarks
    lx, ly = face.left_iris
    rx, ry = face.right_iris

    # ── Eye-relative iris position (direction proxy) ──────────────────────
    def eye_rel(iris_x: float, iris_y: float,
                top_i: int, bot_i: int,
                outer_i: int, inner_i: int) -> tuple[float, float, float, float]:
        x_min = min(lms[outer_i].x, lms[inner_i].x)
        x_max = max(lms[outer_i].x, lms[inner_i].x)
        y_top = lms[top_i].y
        y_bot = lms[bot_i].y
        ew = (x_max - x_min) + 1e-6
        eh = abs(y_bot - y_top) + 1e-6
        rel_x = (iris_x - x_min) / ew
        rel_y = (iris_y - min(y_top, y_bot)) / eh
        return float(rel_x), float(rel_y), float(ew), float(eh)

    l_rx, l_ry, l_ew, l_eh = eye_rel(lx, ly, _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM,
                                       _LEFT_EYE_OUTER, _LEFT_EYE_INNER)
    r_rx, r_ry, r_ew, r_eh = eye_rel(rx, ry, _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM,
                                       _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER)

    # ── Head position proxies ──────────────────────────────────────────────
    nose_x = float(lms[_NOSE_TIP].x)
    nose_y = float(lms[_NOSE_TIP].y)
    chin_x = float(lms[_CHIN].x)
    chin_y = float(lms[_CHIN].y)
    forehead_x = float(lms[_FOREHEAD].x)
    forehead_y = float(lms[_FOREHEAD].y)

    # ── Mean iris position ────────────────────────────────────────────────
    mean_x = (lx + rx) / 2.0
    mean_y = (ly + ry) / 2.0

    return np.array(
        [
            # Absolute iris positions
            lx, ly, rx, ry,
            # Eye-relative iris (gaze direction)
            l_rx, l_ry, r_rx, r_ry,
            # Eye geometry
            l_ew, l_eh, r_ew, r_eh,
            # Head position
            nose_x, nose_y,
            chin_x, chin_y,
            forehead_x, forehead_y,
            # Mean
            mean_x, mean_y,
        ],
        dtype=np.float64,
    )
