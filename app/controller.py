"""Session lifecycle controller – runs the vision pipeline in a worker thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.config import Config
from domain.metrics import compute_debrief
from domain.models import (
    GazeSample,
    GazeState,
    SessionMarker,
    SessionMeta,
    VisionResult,
)
from domain.state_machine import StateMachine
from storage.calibration_store import calibration_hash, load_calibration, save_calibration
from storage.session_writer import SessionWriter
from vision.camera import Camera
from vision.face_tracker import FaceTracker
from vision.gaze_features import extract_features
from vision.gaze_mapper import EMAFilter, GazeMapper

logger = logging.getLogger(__name__)

_RUNS_DIR = Path("runs")


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting hit test on a normalised polygon."""
    if len(polygon) < 3:
        return False
    pts = np.array([[p[0], p[1]] for p in polygon], dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
    return result >= 0.0


class Controller:
    """Owns the camera, face tracker, gaze mapper and state machine.

    The vision pipeline runs in a background thread; results are pushed into
    a thread-safe queue that the UI polls via a Qt timer.
    """

    # Signals (set by the UI)
    on_auto_pause: Optional[Callable[[bool], None]] = None  # True=paused, False=resumed

    def __init__(self, config: Config) -> None:
        self.config = config
        self.camera = Camera(config.camera_index)
        self.face_tracker: Optional[FaceTracker] = None
        self.gaze_mapper: Optional[GazeMapper] = None
        self.ema = EMAFilter(alpha=config.ema_alpha)
        self.state_machine = StateMachine(stable_ms=config.stable_ms)
        self.aoi: Optional[list[tuple[float, float]]] = None

        # Session state
        self._session_writer: Optional[SessionWriter] = None
        self._session_dir: Optional[Path] = None
        self._session_meta: Optional[SessionMeta] = None
        self._session_start_mono: float = 0.0
        self._session_start_wall: float = 0.0
        self._samples: list[GazeSample] = []

        # Worker thread
        self._result_queue: queue.Queue[VisionResult] = queue.Queue(maxsize=5)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False

        # Auto-pause tracking
        self._face_lost_since: Optional[float] = None
        self._auto_paused = False

        # Calibration
        self._calib_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    def start_camera(self) -> None:
        self.camera.start()
        self.face_tracker = FaceTracker()
        logger.info("Controller: camera + face tracker ready.")

    def stop_camera(self) -> None:
        self._stop_worker()
        self.camera.stop()
        if self.face_tracker:
            self.face_tracker.close()
            self.face_tracker = None
        logger.info("Controller: camera stopped.")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def load_calibration(self, profile_name: Optional[str] = None) -> bool:
        name = profile_name or self.config.profile_name
        data = load_calibration(name)
        if data is None:
            return False
        self._apply_calibration(data)
        return True

    def save_calibration_data(
        self,
        mapper: GazeMapper,
        aoi: list[tuple[float, float]],
        rms: float,
        profile_name: Optional[str] = None,
    ) -> None:
        name = profile_name or self.config.profile_name
        data: dict[str, Any] = {
            "version": 1,
            "profile_name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gaze_mapper": mapper.to_dict(),
            "rms_error": rms,
            "aoi_polygon": aoi,
        }
        save_calibration(data, name)
        self._apply_calibration(data)
        logger.info("Calibration saved for profile '%s'  RMS=%.4f", name, rms)

    def _apply_calibration(self, data: dict) -> None:
        self._calib_data = data
        self.gaze_mapper = GazeMapper.from_dict(data["gaze_mapper"])
        raw_aoi = data.get("aoi_polygon", [])
        self.aoi = [(float(p[0]), float(p[1])) for p in raw_aoi]
        logger.info("Calibration applied.  AOI points: %d", len(self.aoi))

    @property
    def is_calibrated(self) -> bool:
        return (
            self.gaze_mapper is not None
            and self.gaze_mapper.is_fitted
            and self.aoi is not None
            and len(self.aoi) >= 3
        )

    @property
    def calibration_rms(self) -> float:
        if self._calib_data:
            return float(self._calib_data.get("rms_error", 0.0))
        return 0.0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, mode: str) -> Path:
        if not self.is_calibrated:
            raise RuntimeError("Cannot start session: not calibrated.")

        ts = datetime.now()
        session_id = ts.strftime("%Y-%m-%d_%H-%M-%S") + f"_{mode}"
        session_dir = _RUNS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        calib_h = calibration_hash(self._calib_data) if self._calib_data else ""
        meta = SessionMeta(
            session_id=session_id,
            mode=mode,
            started_at=ts.isoformat(),
            camera_index=self.config.camera_index,
            camera_width=self.camera.width,
            camera_height=self.camera.height,
            calibration_hash=calib_h,
            profile_name=self.config.profile_name,
        )

        self._session_writer = SessionWriter(session_dir)
        self._session_dir = session_dir
        self._session_meta = meta
        self._session_start_mono = time.monotonic()
        self._session_start_wall = time.time()
        self._samples = []
        self.ema.reset()
        self.state_machine.reset(self._session_start_mono)
        self._face_lost_since = None
        self._auto_paused = False

        self._session_writer.write_meta(meta)

        # Wire state-machine events → writer
        self.state_machine.set_on_transition(self._on_transition)

        self._start_worker()
        logger.info("Session started: %s  mode=%s", session_id, mode)
        return session_dir

    def stop_session(self) -> dict[str, Any]:
        self._stop_worker()

        end_mono = time.monotonic()
        end_wall = time.time()
        duration = end_mono - self._session_start_mono

        # Close last segment
        last_ev = self.state_machine.force_end_segment(end_mono)
        if last_ev and self._session_writer:
            self._session_writer.write_event(last_ev)

        if self._session_meta:
            self._session_meta.ended_at = datetime.fromtimestamp(end_wall).isoformat()
            if self._session_writer:
                self._session_writer.write_meta(self._session_meta)

        debrief = compute_debrief(self._samples, self.state_machine.events, duration)

        if self._session_writer:
            self._session_writer.write_debrief(debrief)
            self._session_writer.close()

        logger.info(
            "Session stopped.  Duration=%.1fs  Samples=%d",
            duration,
            len(self._samples),
        )
        return debrief

    def add_marker(self, label: str = "marker") -> None:
        if self._session_writer is None:
            return
        marker = SessionMarker(
            timestamp_mono=time.monotonic(),
            timestamp_wall=time.time(),
            label=label,
        )
        self._session_writer.write_marker(marker)
        logger.debug("Marker written: %s", label)

    # ------------------------------------------------------------------
    # Result queue (polled by UI)
    # ------------------------------------------------------------------

    def poll_result(self) -> Optional[VisionResult]:
        """Non-blocking read from the result queue."""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="VisionWorker"
        )
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        self._worker_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def _worker_loop(self) -> None:
        assert self.face_tracker is not None
        assert self.gaze_mapper is not None
        assert self.aoi is not None

        target_interval = 1.0 / self.config.fps_target
        last_tick = time.monotonic()

        while self._worker_running:
            # Throttle to target fps
            now = time.monotonic()
            elapsed = now - last_tick
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_tick = time.monotonic()

            frame_data = self.camera.get_frame()
            if frame_data is None:
                continue

            frame_bgr, cam_ts = frame_data
            mono_ts = time.monotonic()
            wall_ts = time.time()

            # Face detection
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face = self.face_tracker.process(frame_rgb)

            auto_paused = False

            if face is None or face.confidence < self.config.min_confidence:
                # Update auto-pause timer
                if self._face_lost_since is None:
                    self._face_lost_since = mono_ts
                lost_for = mono_ts - self._face_lost_since
                if lost_for >= self.config.auto_pause_seconds:
                    if not self._auto_paused:
                        self._auto_paused = True
                        if self.on_auto_pause:
                            self.on_auto_pause(True)
                auto_paused = self._auto_paused

                gx, gy, conf = 0.5, 0.5, 0.0
                raw_state = GazeState.UNKNOWN
                left_iris_norm = right_iris_norm = None
            else:
                # Face found – clear auto-pause
                if self._face_lost_since is not None:
                    self._face_lost_since = None
                if self._auto_paused:
                    self._auto_paused = False
                    if self.on_auto_pause:
                        self.on_auto_pause(False)

                features = extract_features(face)
                gx_raw, gy_raw = self.gaze_mapper.predict(features)
                gx, gy = self.ema.update(gx_raw, gy_raw)
                conf = face.confidence

                if point_in_polygon(gx, gy, self.aoi):
                    raw_state = GazeState.IN_COCKPIT
                else:
                    raw_state = GazeState.OUT_OF_COCKPIT

                left_iris_norm = face.left_iris
                right_iris_norm = face.right_iris

            committed = self.state_machine.update(raw_state, mono_ts)

            sample = GazeSample(
                timestamp_mono=mono_ts,
                timestamp_wall=wall_ts,
                gaze_x_norm=gx,
                gaze_y_norm=gy,
                confidence=conf,
                state=committed,
            )

            if self._session_writer:
                self._session_writer.write_sample(sample)
            self._samples.append(sample)

            result = VisionResult(
                sample=sample,
                face_detected=(face is not None and face.confidence >= self.config.min_confidence),
                auto_paused=auto_paused,
                gaze_px_x=None,  # filled by UI
                gaze_px_y=None,
                left_iris_norm=left_iris_norm,
                right_iris_norm=right_iris_norm,
            )

            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                pass  # drop frame – UI is slower than pipeline

    def _on_transition(self, event) -> None:  # type: ignore[override]
        if self._session_writer:
            self._session_writer.write_event(event)
