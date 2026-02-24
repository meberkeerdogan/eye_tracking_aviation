"""9-point gaze calibration widget."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from vision.face_tracker import FaceTracker
from vision.camera import Camera
from vision.gaze_features import extract_features
from vision.gaze_mapper import GazeMapper

logger = logging.getLogger(__name__)

# 9-point grid in normalised [0,1] coords
_CAL_POINTS: list[tuple[float, float]] = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

_SETTLE_MS = 900    # wait before sampling (eyes settle)
_SAMPLE_MS = 1200   # collect samples for this long
_BLINK_MS = 300     # brief fade between dots


class GazeCalibrationWidget(QWidget):
    """Full-widget calibration routine.

    Emits ``calibration_done(mapper, rms)`` when complete,
    or ``calibration_failed(reason)`` on error.
    """

    calibration_done = Signal(object, float)   # mapper, rms
    calibration_failed = Signal(str)

    # -- States
    _ST_IDLE = "idle"
    _ST_SETTLE = "settle"
    _ST_SAMPLE = "sample"
    _ST_BLINK = "blink"
    _ST_FITTING = "fitting"
    _ST_DONE = "done"

    def __init__(
        self,
        camera: Camera,
        face_tracker: FaceTracker,
        degree: int = 2,
        ridge_alpha: float = 1.0,
        min_confidence: float = 0.3,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._camera = camera
        self._face_tracker = face_tracker
        self._degree = degree
        self._ridge_alpha = ridge_alpha
        self._min_conf = min_confidence

        self._point_idx = 0
        self._state = self._ST_IDLE
        self._state_start = 0.0
        self._dot_alpha = 1.0          # 0-1 for fade effect

        # Collected calibration data
        self._features_list: list[np.ndarray] = []
        self._targets_list: list[tuple[float, float]] = []

        self._current_features: list[np.ndarray] = []  # buffer for current dot

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz
        self._timer.timeout.connect(self._tick)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: #1a1a2e;")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._point_idx = 0
        self._features_list = []
        self._targets_list = []
        self._current_features = []
        self._enter_settle()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._state = self._ST_IDLE

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _enter_settle(self) -> None:
        self._state = self._ST_SETTLE
        self._state_start = time.monotonic()
        self._current_features = []
        self.update()

    def _enter_sample(self) -> None:
        self._state = self._ST_SAMPLE
        self._state_start = time.monotonic()
        self.update()

    def _enter_blink(self) -> None:
        self._state = self._ST_BLINK
        self._state_start = time.monotonic()
        self.update()

    def _tick(self) -> None:
        mono = time.monotonic()
        elapsed_ms = (mono - self._state_start) * 1000.0

        if self._state == self._ST_SETTLE:
            self._dot_alpha = min(1.0, elapsed_ms / _SETTLE_MS)
            if elapsed_ms >= _SETTLE_MS:
                self._enter_sample()

        elif self._state == self._ST_SAMPLE:
            progress = min(1.0, elapsed_ms / _SAMPLE_MS)
            self._dot_alpha = 1.0

            # Grab frame and extract features
            frame_data = self._camera.get_frame()
            if frame_data is not None:
                import cv2
                frame_bgr, _ = frame_data
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                face = self._face_tracker.process(frame_rgb)
                if face and face.confidence >= self._min_conf:
                    feats = extract_features(face)
                    self._current_features.append(feats)

            if elapsed_ms >= _SAMPLE_MS:
                self._commit_point()
                self._enter_blink()

        elif self._state == self._ST_BLINK:
            fade = 1.0 - min(1.0, elapsed_ms / _BLINK_MS)
            self._dot_alpha = fade
            if elapsed_ms >= _BLINK_MS:
                self._point_idx += 1
                if self._point_idx >= len(_CAL_POINTS):
                    self._fit_model()
                else:
                    self._enter_settle()

        self.update()

    def _commit_point(self) -> None:
        if not self._current_features:
            logger.warning("No samples collected for calibration point %d", self._point_idx)
            return
        stacked = np.stack(self._current_features, axis=0)
        mean_feat = stacked.mean(axis=0)
        self._features_list.append(mean_feat)
        target = _CAL_POINTS[self._point_idx]
        self._targets_list.append(target)
        logger.debug(
            "Cal point %d: %d samples  target=(%.2f, %.2f)",
            self._point_idx,
            len(self._current_features),
            target[0], target[1],
        )

    def _fit_model(self) -> None:
        self._state = self._ST_FITTING
        self._timer.stop()
        self.update()

        if len(self._features_list) < 5:
            self.calibration_failed.emit(
                f"Only {len(self._features_list)} calibration points collected (need ≥ 5). "
                "Make sure your face is visible."
            )
            return

        features = np.stack(self._features_list, axis=0)
        targets = np.array(self._targets_list, dtype=np.float64)

        mapper = GazeMapper(degree=self._degree, alpha=self._ridge_alpha)
        rms = mapper.fit(features, targets)

        self._state = self._ST_DONE
        self.update()

        self.calibration_done.emit(mapper, rms)
        logger.info("Calibration complete.  RMS=%.4f  N=%d", rms, len(self._features_list))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(15, 15, 30))

        w, h = self.width(), self.height()
        mid_x, mid_y = w // 2, h // 2

        if self._state == self._ST_IDLE:
            self._draw_text(painter, "Press Start to begin calibration", mid_x, mid_y)
            return

        if self._state == self._ST_FITTING:
            self._draw_text(painter, "Fitting gaze model…", mid_x, mid_y)
            return

        if self._state == self._ST_DONE:
            self._draw_text(painter, "Calibration complete!", mid_x, mid_y)
            return

        # Progress indicator (top left)
        progress_text = (
            f"Point {self._point_idx + 1} / {len(_CAL_POINTS)}"
        )
        font = QFont("Segoe UI", 11)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 200))
        painter.drawText(12, 24, progress_text)

        if self._state in (self._ST_SETTLE, self._ST_SAMPLE, self._ST_BLINK):
            if self._point_idx < len(_CAL_POINTS):
                tx, ty = _CAL_POINTS[self._point_idx]
                px, py = int(tx * w), int(ty * h)
                alpha = int(self._dot_alpha * 255)
                self._draw_dot(painter, px, py, alpha)

        # Status text
        if self._state == self._ST_SETTLE:
            self._draw_text(painter, "Look at the dot…", mid_x, h - 40, small=True)
        elif self._state == self._ST_SAMPLE:
            elapsed_ms = (time.monotonic() - self._state_start) * 1000
            pct = int(min(100, elapsed_ms / _SAMPLE_MS * 100))
            self._draw_text(painter, f"Hold still… {pct}%", mid_x, h - 40, small=True)

    def _draw_dot(self, painter: QPainter, cx: int, cy: int, alpha: int) -> None:
        r = 18
        grad = QRadialGradient(cx, cy, r)
        grad.setColorAt(0.0, QColor(255, 50, 50, alpha))
        grad.setColorAt(0.6, QColor(220, 0, 0, int(alpha * 0.7)))
        grad.setColorAt(1.0, QColor(200, 0, 0, 0))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # White crosshair
        pen = QPen(QColor(255, 255, 255, alpha), 1)
        painter.setPen(pen)
        painter.drawLine(cx - r, cy, cx + r, cy)
        painter.drawLine(cx, cy - r, cx, cy + r)

    @staticmethod
    def _draw_text(
        painter: QPainter, text: str, x: int, y: int, small: bool = False
    ) -> None:
        font = QFont("Segoe UI", 10 if small else 18)
        font.setBold(not small)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 255))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        painter.drawText(x - tw // 2, y, text)
