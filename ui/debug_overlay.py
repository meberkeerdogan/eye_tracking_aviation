"""Transparent overlay widget that renders the gaze dot and debug info."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from domain.models import GazeState, VisionResult


class DebugOverlay(QWidget):
    """Completely transparent child widget placed on top of the session screen.

    Call :meth:`update_result` to feed new data; the widget repaints itself.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.Widget)

        self._result: Optional[VisionResult] = None
        self._show_landmarks = False
        self._show_gaze = True
        self._paused = False

        # Running stats for display
        self._in_s: float = 0.0
        self._out_s: float = 0.0
        self._unk_s: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_result(self, result: VisionResult) -> None:
        self._result = result
        self._paused = result.auto_paused
        self.update()

    def set_durations(self, in_s: float, out_s: float, unk_s: float) -> None:
        self._in_s = in_s
        self._out_s = out_s
        self._unk_s = unk_s

    def set_show_landmarks(self, show: bool) -> None:
        self._show_landmarks = show

    def set_show_gaze(self, show: bool) -> None:
        self._show_gaze = show

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        if self._result is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        r = self._result
        w, h = self.width(), self.height()

        # ── Auto-pause warning ─────────────────────────────────────────
        if self._paused:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
            font = QFont("Segoe UI", 22, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 80, 80))
            msg = "⚠  Face not detected  ⚠"
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(msg)
            painter.drawText((w - tw) // 2, h // 2, msg)

        # ── Gaze dot ───────────────────────────────────────────────────
        if self._show_gaze and r.gaze_px_x is not None and r.gaze_px_y is not None:
            gx, gy = r.gaze_px_x, r.gaze_px_y
            state = r.sample.state

            if state == GazeState.IN_COCKPIT:
                dot_colour = QColor(0, 220, 100)
            elif state == GazeState.OUT_OF_COCKPIT:
                dot_colour = QColor(220, 50, 50)
            else:
                dot_colour = QColor(150, 150, 150)

            radius = 18
            grad = QRadialGradient(gx, gy, radius)
            grad.setColorAt(0.0, QColor(dot_colour.red(), dot_colour.green(), dot_colour.blue(), 220))
            grad.setColorAt(1.0, QColor(dot_colour.red(), dot_colour.green(), dot_colour.blue(), 0))
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(gx) - radius, int(gy) - radius, radius * 2, radius * 2)

            # Crosshair
            pen = QPen(dot_colour, 1)
            painter.setPen(pen)
            painter.drawLine(int(gx) - radius, int(gy), int(gx) + radius, int(gy))
            painter.drawLine(int(gx), int(gy) - radius, int(gx), int(gy) + radius)

        # ── Metrics panel (top-right) ──────────────────────────────────
        self._draw_metrics(painter, r)

    def _draw_metrics(self, painter: QPainter, r: VisionResult) -> None:
        font = QFont("Segoe UI", 10)
        painter.setFont(font)

        state = r.sample.state
        state_colours = {
            GazeState.IN_COCKPIT: "#00dc64",
            GazeState.OUT_OF_COCKPIT: "#dc3232",
            GazeState.UNKNOWN: "#888888",
        }
        sc = state_colours.get(state, "#ffffff")

        total = self._in_s + self._out_s + self._unk_s or 1.0
        lines = [
            f"State: <{state.value}>",
            f"Conf:  {r.sample.confidence:.2f}",
            f"IN:    {self._in_s:.1f}s  ({self._in_s/total*100:.0f}%)",
            f"OUT:   {self._out_s:.1f}s  ({self._out_s/total*100:.0f}%)",
            f"UNK:   {self._unk_s:.1f}s",
        ]

        panel_x = self.width() - 210
        panel_y = 10
        line_h = 18
        panel_h = len(lines) * line_h + 16

        # Semi-transparent background
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(panel_x - 8, panel_y - 4, 210, panel_h, 6, 6)

        painter.setPen(QColor(200, 200, 255))
        for i, line in enumerate(lines):
            y = panel_y + i * line_h + line_h
            if i == 0:
                painter.setPen(QColor(sc))
            else:
                painter.setPen(QColor(200, 200, 255))
            painter.drawText(panel_x, y, line)
