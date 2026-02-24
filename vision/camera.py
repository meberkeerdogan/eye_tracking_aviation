"""Threaded camera capture with timestamps."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Grabs frames from a webcam in a background thread so the main
    pipeline never blocks on I/O.  Call :meth:`get_frame` to retrieve
    the latest captured frame without waiting."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._width: int = 0
        self._height: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback: try without backend flag
            self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.index}.")

        # Try to set a reasonable resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCapture")
        self._thread.start()
        logger.info("Camera started: index=%d  res=%dx%d", self.index, self._width, self._height)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped.")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[tuple[np.ndarray, float]]:
        """Return ``(frame_bgr, monotonic_timestamp)`` or ``None`` if no
        frame has been captured yet."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy(), self._timestamp

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                ts = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._timestamp = ts
            else:
                time.sleep(0.005)
