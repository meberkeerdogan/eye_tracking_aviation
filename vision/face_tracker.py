"""MediaPipe FaceLandmarker (Tasks API) wrapper.

MediaPipe 0.10.30+ removed the legacy ``mp.solutions`` API entirely.
This module uses the current Tasks API with a downloaded .task model file.
The public ``FaceResult`` dataclass keeps the same interface as before so
no other modules need to change.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger(__name__)

# ── Model file ────────────────────────────────────────────────────────────────
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_PATH = Path("assets") / _MODEL_FILENAME
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── Landmark indices (identical in both the old FaceMesh and new FaceLandmarker)
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


# ── Model download ─────────────────────────────────────────────────────────────

def ensure_model(model_path: Path = _MODEL_PATH) -> Path:
    """Return the model path, downloading it first if necessary.

    Raises ``RuntimeError`` on network failure so the caller can show a
    user-friendly error message.
    """
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FaceLandmarker model → %s", model_path)
    print(f"\nDownloading MediaPipe face model ({_MODEL_FILENAME}) — ~10 MB, one-time…")

    try:
        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                print(f"\r  [{bar}] {pct}%", end="", flush=True)

        urllib.request.urlretrieve(_MODEL_URL, str(model_path), reporthook=_progress)
        print("\n  Download complete.\n")
        logger.info("Model saved: %s", model_path)
    except Exception as exc:
        model_path.unlink(missing_ok=True)  # remove partial file
        raise RuntimeError(
            f"Failed to download FaceLandmarker model from:\n{_MODEL_URL}\n\n"
            f"Error: {exc}\n\n"
            "Please check your internet connection, or manually download the file\n"
            f"and place it at:  {model_path.resolve()}"
        ) from exc

    return model_path


# ── Result dataclass (interface unchanged from the old solutions API) ──────────

@dataclass
class FaceResult:
    """Processed output from one camera frame."""

    landmarks: list            # list[NormalizedLandmark]  – .x .y .z per item
    left_iris: tuple[float, float]
    right_iris: tuple[float, float]
    left_openness: float
    right_openness: float
    confidence: float
    nose_tip: tuple[float, float]


# ── Tracker class ──────────────────────────────────────────────────────────────

class FaceTracker:
    """Thin wrapper around the MediaPipe Tasks FaceLandmarker.

    Must be used from a **single thread** – MediaPipe landmarkers are not
    thread-safe.  The controller's worker thread owns this instance exclusively.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        path = model_path or ensure_model()
        base_options = BaseOptions(model_asset_path=str(path.resolve()))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode uses temporal tracking between frames (faster than IMAGE)
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start_mono = time.monotonic()
        self._last_ts_ms: int = -1
        logger.info("FaceTracker initialised (Tasks API, model=%s)", path.name)

    def process(self, frame_rgb: np.ndarray) -> Optional[FaceResult]:
        """Process one RGB frame; returns ``None`` if no face is detected."""
        # Timestamp must be strictly monotonically increasing for VIDEO mode
        ts_ms = int((time.monotonic() - self._start_mono) * 1000)
        ts_ms = max(ts_ms, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            return None

        lms = result.face_landmarks[0]  # list[NormalizedLandmark]

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

        avg_open = (left_open + right_open) / 2.0
        confidence = float(min(1.0, max(0.0, avg_open / 0.15)))

        return FaceResult(
            landmarks=lms,
            left_iris=left_iris,
            right_iris=right_iris,
            left_openness=left_open,
            right_openness=right_open,
            confidence=confidence,
            nose_tip=(float(lms[_NOSE_TIP].x), float(lms[_NOSE_TIP].y)),
        )

    def close(self) -> None:
        self._landmarker.close()
        logger.debug("FaceTracker closed.")
