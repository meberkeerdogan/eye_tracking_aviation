"""Polygon AOI editor widget – drawn on top of the cockpit image."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

logger = logging.getLogger(__name__)


class AOIEditor(QWidget):
    """Shows the cockpit image and lets the user click to define a polygon AOI.

    Normalised vertex coordinates (0-1 relative to this widget's size) are
    accumulated in :attr:`polygon_norm`.

    Signals
    -------
    polygon_changed : emitted whenever a vertex is added or cleared.
    """

    polygon_changed = Signal()

    def __init__(self, cockpit_pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = cockpit_pixmap
        self._points_norm: list[tuple[float, float]] = []  # (x_norm, y_norm)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Left-click to add vertices.  Double-click to finish.  Right-click to remove last point.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def polygon_norm(self) -> list[tuple[float, float]]:
        return list(self._points_norm)

    @property
    def is_valid(self) -> bool:
        return len(self._points_norm) >= 3

    def clear(self) -> None:
        self._points_norm.clear()
        self.polygon_changed.emit()
        self.update()

    def set_polygon(self, pts: list[tuple[float, float]]) -> None:
        self._points_norm = list(pts)
        self.polygon_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            x_norm = event.position().x() / self.width()
            y_norm = event.position().y() / self.height()
            self._points_norm.append((x_norm, y_norm))
            self.polygon_changed.emit()
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            if self._points_norm:
                self._points_norm.pop()
                self.polygon_changed.emit()
                self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        # Remove the duplicate point added by the single-click that fires before
        # a double-click, then signal that the polygon is "closed"
        if self._points_norm and len(self._points_norm) >= 3:
            # Remove the extra single-click point from the double-click event
            self._points_norm.pop()
        self.polygon_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background image scaled to fill widget
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(0, 0, scaled)
        else:
            painter.fillRect(self.rect(), QColor(40, 40, 40))

        if not self._points_norm:
            return

        # Convert to pixel coords
        pts_px = [
            (x * self.width(), y * self.height()) for x, y in self._points_norm
        ]

        # Draw filled polygon (semi-transparent)
        if len(pts_px) >= 3:
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF

            poly = QPolygonF([QPointF(x, y) for x, y in pts_px])
            fill = QColor(0, 200, 100, 60)
            painter.setBrush(fill)
            outline = QPen(QColor(0, 255, 100), 2)
            painter.setPen(outline)
            painter.drawPolygon(poly)

        # Draw vertices
        vertex_pen = QPen(QColor(255, 200, 0), 2)
        painter.setPen(vertex_pen)
        for i, (px, py) in enumerate(pts_px):
            painter.setBrush(QColor(255, 200, 0))
            painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)
            painter.drawText(int(px) + 8, int(py) - 4, str(i + 1))

        # Draw closing line if >= 2 points
        if len(pts_px) >= 2:
            line_pen = QPen(QColor(0, 255, 100), 2, Qt.PenStyle.DashLine)
            painter.setPen(line_pen)
            # Line from last to first (closing)
            painter.drawLine(
                int(pts_px[-1][0]), int(pts_px[-1][1]),
                int(pts_px[0][0]), int(pts_px[0][1]),
            )
