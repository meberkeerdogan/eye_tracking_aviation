"""Session recording screen – shows cockpit image, captures gaze."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.controller import Controller
from domain.models import GazeState, VisionResult
from ui.debug_overlay import DebugOverlay

logger = logging.getLogger(__name__)


class CockpitCanvas(QWidget):
    """Draws the cockpit background image, scaled to fill the widget."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(0, 0, scaled)
        else:
            painter.fillRect(self.rect(), QColor(40, 55, 70))
            painter.setPen(QColor(180, 180, 200))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Cockpit image not found")


class SessionScreen(QWidget):
    """Full session recording widget.

    In *debug* mode: gaze overlay visible + metrics panel.
    In *test* mode:  no overlay visible to participant.

    Press **M** to write a session marker.
    """

    session_ended = Signal(dict, Path)  # debrief dict, session dir

    def __init__(
        self,
        controller: Controller,
        cockpit_pixmap: QPixmap,
        mode: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._mode = mode

        self._session_dir: Optional[Path] = None
        self._running = False
        self._start_mono: float = 0.0

        # Accumulated state durations (kept in UI thread for overlay display)
        self._state_durations = {s: 0.0 for s in GazeState}
        self._last_sample_time: Optional[float] = None
        self._last_state: GazeState = GazeState.UNKNOWN

        # ── Build UI ──────────────────────────────────────────────────
        self._canvas = CockpitCanvas(cockpit_pixmap)
        self._overlay = DebugOverlay(self._canvas)
        self._overlay.setGeometry(self._canvas.rect())
        self._overlay.set_show_gaze(mode == "debug")

        # Status bar at bottom
        self._status_label = QLabel("Press End Session when done  |  M = marker")
        self._status_label.setStyleSheet(
            "background: rgba(0,0,0,180); color: #aaaacc; padding: 4px 10px; font-size: 11px;"
        )
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._elapsed_label = QLabel("00:00")
        self._elapsed_label.setStyleSheet(
            "background: rgba(0,0,0,180); color: #ffffff; padding: 4px 16px;"
            "font-size: 16px; font-weight: bold;"
        )

        self._end_btn = QPushButton("■ End Session")
        self._end_btn.setFixedHeight(36)
        self._end_btn.setStyleSheet(
            "background: #cc3333; color: white; font-weight: bold; font-size: 13px;"
            "border: none; padding: 0 20px; border-radius: 4px;"
        )
        self._end_btn.clicked.connect(self.end_session)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(8, 4, 8, 4)
        bottom.addWidget(self._status_label, 1)
        bottom.addWidget(self._elapsed_label)
        bottom.addWidget(self._end_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(bottom)

        # ── Timers ────────────────────────────────────────────────────
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(16)  # ~60 Hz
        self._poll_timer.timeout.connect(self._poll_results)

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(500)
        self._clock_timer.timeout.connect(self._update_clock)

        # ── Callbacks ────────────────────────────────────────────────
        self._ctrl.on_auto_pause = self._on_auto_pause

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        self._running = True
        self._start_mono = time.monotonic()
        self._state_durations = {s: 0.0 for s in GazeState}
        self._last_sample_time = None
        self._last_state = GazeState.UNKNOWN

        try:
            self._session_dir = self._ctrl.start_session(self._mode)
        except RuntimeError as exc:
            logger.error("Cannot start session: %s", exc)
            return

        self._poll_timer.start()
        self._clock_timer.start()
        self.setFocus()

    def end_session(self) -> None:
        if not self._running:
            return
        self._running = False
        self._poll_timer.stop()
        self._clock_timer.stop()

        debrief = self._ctrl.stop_session()
        self.session_ended.emit(debrief, self._session_dir or Path("."))

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_M and self._running:
            self._ctrl.add_marker("manual")
            self._flash_marker_indicator()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Timers / polling
    # ------------------------------------------------------------------

    def _poll_results(self) -> None:
        while True:
            result = self._ctrl.poll_result()
            if result is None:
                break
            self._process_result(result)

    def _process_result(self, result: VisionResult) -> None:
        # Accumulate state durations locally for live overlay
        now = result.sample.timestamp_mono
        if self._last_sample_time is not None:
            dt = now - self._last_sample_time
            self._state_durations[self._last_state] += dt
        self._last_sample_time = now
        self._last_state = result.sample.state

        # Convert normalised gaze to pixel coords within canvas
        cw, ch = self._canvas.width(), self._canvas.height()
        gx = result.sample.gaze_x_norm * cw
        gy = result.sample.gaze_y_norm * ch
        result.gaze_px_x = gx
        result.gaze_px_y = gy

        # Update overlay
        in_s = self._state_durations[GazeState.IN_COCKPIT]
        out_s = self._state_durations[GazeState.OUT_OF_COCKPIT]
        unk_s = self._state_durations[GazeState.UNKNOWN]
        self._overlay.set_durations(in_s, out_s, unk_s)
        self._overlay.update_result(result)

    def _update_clock(self) -> None:
        elapsed = int(time.monotonic() - self._start_mono)
        m, s = divmod(elapsed, 60)
        self._elapsed_label.setText(f"{m:02d}:{s:02d}")

    def _on_auto_pause(self, paused: bool) -> None:
        if paused:
            self._status_label.setText("⚠  Face lost – recording paused  |  M = marker")
            self._status_label.setStyleSheet(
                "background: rgba(180,0,0,160); color: #ffffff; padding: 4px 10px; font-size: 11px;"
            )
        else:
            self._status_label.setText("Press End Session when done  |  M = marker")
            self._status_label.setStyleSheet(
                "background: rgba(0,0,0,180); color: #aaaacc; padding: 4px 10px; font-size: 11px;"
            )

    def _flash_marker_indicator(self) -> None:
        orig = self._status_label.styleSheet()
        self._status_label.setText("● Marker recorded")
        self._status_label.setStyleSheet(
            "background: rgba(0,150,80,200); color: #ffffff; padding: 4px 10px; font-size: 11px;"
        )
        QTimer.singleShot(
            800,
            lambda: (
                self._status_label.setText("Press End Session when done  |  M = marker"),
                self._status_label.setStyleSheet(orig),
            ),
        )

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._canvas:
            self._overlay.setGeometry(self._canvas.rect())
