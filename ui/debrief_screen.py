"""Debrief screen with metrics, charts, and replay."""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not available – charts disabled.")


class StatCard(QFrame):
    """Small card widget for displaying a key metric."""

    def __init__(self, title: str, value: str, colour: str = "#aaaacc") -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"QFrame {{ background: #1e2240; border-radius: 8px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)

        v_lbl = QLabel(value)
        v_lbl.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {colour};")
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("font-size: 10px; color: #888; text-transform: uppercase;")

        lay.addWidget(v_lbl)
        lay.addWidget(t_lbl)


class ReplayCanvas(QWidget):
    """Shows the cockpit image with a moving gaze dot for replay."""

    def __init__(self, cockpit_pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = cockpit_pixmap
        self._gx: float = 0.5
        self._gy: float = 0.5
        self._state: str = "UNKNOWN"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)

    def set_gaze(self, gx: float, gy: float, state: str) -> None:
        self._gx = gx
        self._gy = gy
        self._state = state
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(0, 0, scaled)
        else:
            painter.fillRect(self.rect(), QColor(40, 55, 70))

        cx = int(self._gx * self.width())
        cy = int(self._gy * self.height())
        r = 16

        colours = {
            "IN_COCKPIT": QColor(0, 220, 80),
            "OUT_OF_COCKPIT": QColor(220, 50, 50),
            "UNKNOWN": QColor(150, 150, 150),
        }
        c = colours.get(self._state, QColor(200, 200, 200))
        grad = QRadialGradient(cx, cy, r)
        grad.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), 200))
        grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)


class DebriefScreen(QWidget):
    """Shows summary statistics, charts, and gaze replay after a session."""

    home_requested = Signal()

    def __init__(
        self,
        debrief: dict[str, Any],
        session_dir: Path,
        cockpit_pixmap: QPixmap,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._debrief = debrief
        self._session_dir = session_dir
        self._pixmap = cockpit_pixmap

        # Replay state
        self._replay_samples: list[dict] = []
        self._replay_idx = 0
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(33)  # ~30 fps
        self._replay_timer.timeout.connect(self._replay_tick)
        self._replay_playing = False

        self._build_ui()
        self._load_replay_data()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        d = self._debrief
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        # Title
        title = QLabel("Session Debrief")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #e0e0ff;")
        root.addWidget(title)

        # ── Stat cards ────────────────────────────────────────────────
        cards_layout = QHBoxLayout()
        total_s = d.get("total_duration_s", 0)
        cards_layout.addWidget(StatCard("Total Duration", f"{total_s:.1f}s"))
        cards_layout.addWidget(StatCard(
            "In Cockpit",
            f"{d.get('in_cockpit_pct', 0):.1f}%",
            "#00dc64",
        ))
        cards_layout.addWidget(StatCard(
            "Out of Cockpit",
            f"{d.get('out_cockpit_pct', 0):.1f}%",
            "#dc3232",
        ))
        cards_layout.addWidget(StatCard(
            "Unknown",
            f"{d.get('unknown_pct', 0):.1f}%",
            "#888888",
        ))
        cards_layout.addWidget(StatCard(
            "Out Glances",
            str(d.get("n_out_glances", 0)),
            "#ffaa00",
        ))
        avg_out = d.get("avg_out_ms", 0)
        cards_layout.addWidget(StatCard("Avg OUT ms", f"{avg_out:.0f}"))
        max_out = d.get("max_out_ms", 0)
        cards_layout.addWidget(StatCard("Max OUT ms", f"{max_out:.0f}", "#ff6666"))

        root.addLayout(cards_layout)

        # ── Main splitter: charts (left) | replay (right) ─────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        if _HAS_MPL:
            charts_widget = self._build_charts()
        else:
            charts_widget = QLabel("Install matplotlib for charts.")
            charts_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(charts_widget)

        replay_widget = self._build_replay_panel()
        splitter.addWidget(replay_widget)
        splitter.setSizes([550, 450])

        root.addWidget(splitter, 1)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        btn_folder = QPushButton("📂 Open Run Folder")
        btn_folder.clicked.connect(self._open_folder)

        btn_export = QPushButton("💾 Export Summary CSV")
        btn_export.clicked.connect(self._export_csv)

        btn_home = QPushButton("← Back to Home")
        btn_home.clicked.connect(self.home_requested)
        btn_home.setStyleSheet("font-weight: bold;")

        btn_row.addWidget(btn_home)
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_folder)

        root.addLayout(btn_row)

    def _build_charts(self) -> QWidget:
        """Matplotlib charts embedded in a widget."""
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)

        d = self._debrief
        fig = Figure(figsize=(5.5, 7), facecolor="#12122a")

        # ── Pie chart ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_facecolor("#12122a")
        labels = ["In Cockpit", "Out of Cockpit", "Unknown"]
        sizes = [
            d.get("in_cockpit_s", 0),
            d.get("out_cockpit_s", 0),
            d.get("unknown_s", 0),
        ]
        colours = ["#00dc64", "#dc3232", "#666688"]
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colours) if s > 0]
        if non_zero:
            sz, lb, co = zip(*non_zero)
            wedges, texts, autotexts = ax1.pie(
                sz, labels=lb, colors=co, autopct="%1.1f%%",
                textprops={"color": "#ccccee", "fontsize": 8},
            )
            for at in autotexts:
                at.set_fontsize(7)
        ax1.set_title("Time Distribution", color="#ccccee", fontsize=10)

        # ── OUT duration histogram ─────────────────────────────────────
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.set_facecolor("#1a1a2e")
        out_ms = d.get("out_durations_ms", [])
        if out_ms:
            ax2.hist(out_ms, bins=min(20, len(out_ms)), color="#dc3232", alpha=0.8, edgecolor="#ff6666")
        ax2.set_xlabel("Duration (ms)", color="#aaaacc", fontsize=8)
        ax2.set_ylabel("Count", color="#aaaacc", fontsize=8)
        ax2.set_title("OUT Glance Durations", color="#ccccee", fontsize=10)
        ax2.tick_params(colors="#aaaacc", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#334466")

        # ── State timeline ─────────────────────────────────────────────
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.set_facecolor("#1a1a2e")
        timeline = d.get("timeline", [])
        if timeline:
            ts = [pt["t_s"] for pt in timeline]
            state_map = {"IN_COCKPIT": 2, "OUT_OF_COCKPIT": 1, "UNKNOWN": 0}
            sv = [state_map.get(pt["state"], 0) for pt in timeline]
            ax3.fill_between(ts, sv, step="post", alpha=0.8,
                             color="#334466", linewidth=0)
            # Colour by state
            in_mask = [v == 2 for v in sv]
            out_mask = [v == 1 for v in sv]
            ax3.scatter(
                [t for t, m in zip(ts, in_mask) if m],
                [2] * sum(in_mask), c="#00dc64", s=3, linewidths=0
            )
            ax3.scatter(
                [t for t, m in zip(ts, out_mask) if m],
                [1] * sum(out_mask), c="#dc3232", s=3, linewidths=0
            )
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["UNK", "OUT", "IN"], color="#aaaacc", fontsize=7)
        ax3.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
        ax3.set_title("State Timeline", color="#ccccee", fontsize=10)
        ax3.tick_params(colors="#aaaacc", labelsize=7)
        for spine in ax3.spines.values():
            spine.set_edgecolor("#334466")

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(canvas)
        return container

    def _build_replay_panel(self) -> QWidget:
        container = QGroupBox("Gaze Replay")
        container.setStyleSheet(
            "QGroupBox { font-size: 13px; font-weight: bold; color: #aaaaee; "
            "border: 1px solid #334; border-radius: 6px; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        lay = QVBoxLayout(container)

        self._replay_canvas = ReplayCanvas(self._pixmap)
        lay.addWidget(self._replay_canvas, 1)

        # Scrubber
        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(1000)
        self._scrubber.setValue(0)
        self._scrubber.valueChanged.connect(self._scrubber_changed)
        lay.addWidget(self._scrubber)

        # Playback buttons
        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._toggle_replay)
        btn_reset = QPushButton("⏮ Reset")
        btn_reset.clicked.connect(self._reset_replay)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()

        self._replay_time_label = QLabel("0.0s")
        self._replay_time_label.setStyleSheet("color: #aaaacc;")
        btn_row.addWidget(self._replay_time_label)

        lay.addLayout(btn_row)
        return container

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def _load_replay_data(self) -> None:
        samples_path = self._session_dir / "samples.csv"
        if not samples_path.exists():
            return
        try:
            with open(samples_path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self._replay_samples = rows
            if rows:
                self._scrubber.setMaximum(len(rows) - 1)
        except OSError as exc:
            logger.warning("Cannot load replay data: %s", exc)

    def _scrubber_changed(self, value: int) -> None:
        self._replay_idx = value
        self._update_replay_frame()

    def _toggle_replay(self, playing: bool) -> None:
        self._replay_playing = playing
        if playing:
            self._play_btn.setText("⏸ Pause")
            self._replay_timer.start()
        else:
            self._play_btn.setText("▶ Play")
            self._replay_timer.stop()

    def _reset_replay(self) -> None:
        self._replay_timer.stop()
        self._play_btn.setChecked(False)
        self._play_btn.setText("▶ Play")
        self._replay_idx = 0
        self._scrubber.setValue(0)
        self._update_replay_frame()

    def _replay_tick(self) -> None:
        if self._replay_idx >= len(self._replay_samples) - 1:
            self._toggle_replay(False)
            self._play_btn.setChecked(False)
            return
        self._replay_idx += 1
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(self._replay_idx)
        self._scrubber.blockSignals(False)
        self._update_replay_frame()

    def _update_replay_frame(self) -> None:
        if not self._replay_samples:
            return
        idx = min(self._replay_idx, len(self._replay_samples) - 1)
        row = self._replay_samples[idx]
        try:
            gx = float(row.get("gaze_x_norm", 0.5))
            gy = float(row.get("gaze_y_norm", 0.5))
            state = row.get("state", "UNKNOWN")
            t0 = float(self._replay_samples[0].get("timestamp_mono", 0))
            t = float(row.get("timestamp_mono", t0)) - t0
            self._replay_canvas.set_gaze(gx, gy, state)
            self._replay_time_label.setText(f"{t:.1f}s")
        except (ValueError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_folder(self) -> None:
        path = str(self._session_dir.resolve())
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)

    def _export_csv(self) -> None:
        """Write a single-row summary CSV to the session dir."""
        out_path = self._session_dir / "summary_export.csv"
        d = self._debrief
        fields = [
            "total_duration_s", "in_cockpit_s", "out_cockpit_s", "unknown_s",
            "in_cockpit_pct", "out_cockpit_pct", "unknown_pct",
            "n_out_glances", "avg_out_ms", "median_out_ms", "max_out_ms",
            "total_samples", "avg_confidence",
        ]
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow({k: d.get(k, "") for k in fields})
            logger.info("Summary CSV exported to %s", out_path)
            self._open_folder()
        except OSError as exc:
            logger.error("Export failed: %s", exc)
