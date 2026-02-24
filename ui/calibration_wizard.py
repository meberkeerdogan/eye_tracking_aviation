"""Calibration wizard – two steps: gaze mapping then AOI drawing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.controller import Controller
from calibration.aoi_editor import AOIEditor
from calibration.gaze_calibration import GazeCalibrationWidget
from vision.gaze_mapper import GazeMapper

logger = logging.getLogger(__name__)

_WARN_RMS = 0.06  # warn if RMS exceeds this


class CalibrationWizard(QWidget):
    """Two-step calibration wizard.

    Step 1 – 9-point gaze mapping calibration.
    Step 2 – AOI polygon editor.

    Emits ``calibration_saved`` when both steps complete and the calibration
    has been persisted to disk.
    """

    calibration_saved = Signal()
    cancelled = Signal()

    def __init__(
        self,
        controller: Controller,
        cockpit_pixmap: QPixmap,
        config: Config,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._pixmap = cockpit_pixmap
        self._config = config

        self._mapper: Optional[GazeMapper] = None
        self._rms: float = 0.0

        self._stack = QStackedWidget()
        self._build_step1()
        self._build_step2()

        # Navigation buttons
        self._btn_cancel = QPushButton("Cancel")
        self._btn_back = QPushButton("← Back")
        self._btn_next = QPushButton("Start Calibration")
        self._btn_next.setDefault(True)

        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_back.clicked.connect(self._on_back)
        self._btn_next.clicked.connect(self._on_next)

        nav = QHBoxLayout()
        nav.addWidget(self._btn_cancel)
        nav.addStretch()
        nav.addWidget(self._btn_back)
        nav.addWidget(self._btn_next)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack, 1)
        layout.addLayout(nav)

        self._update_nav()

    # ------------------------------------------------------------------
    # Build steps
    # ------------------------------------------------------------------

    def _build_step1(self) -> None:
        """Step 1 wrapper: instruction screen + gaze calibration widget."""
        self._step1_container = QWidget()
        layout = QVBoxLayout(self._step1_container)
        layout.setContentsMargins(24, 24, 24, 0)

        title = QLabel("Step 1 of 2 – Gaze Mapping Calibration")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #e0e0ff;")

        info = QLabel(
            "9 dots will appear one by one.  Look directly at each dot and hold still.\n"
            "The dot will turn red while sampling.  Do not move your head during calibration."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 13px; color: #aaaacc;")

        self._gaze_cal_widget = GazeCalibrationWidget(
            camera=self._ctrl.camera,
            face_tracker=self._ctrl.face_tracker,  # type: ignore[arg-type]
            degree=self._config.calib_degree,
            ridge_alpha=self._config.calib_ridge_alpha,
            min_confidence=self._config.min_confidence,
        )
        self._gaze_cal_widget.calibration_done.connect(self._on_gaze_cal_done)
        self._gaze_cal_widget.calibration_failed.connect(self._on_gaze_cal_failed)

        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(self._gaze_cal_widget, 1)

        self._step1_started = False
        self._stack.addWidget(self._step1_container)

    def _build_step2(self) -> None:
        """Step 2: AOI polygon editor."""
        self._step2_container = QWidget()
        layout = QVBoxLayout(self._step2_container)
        layout.setContentsMargins(24, 24, 24, 0)

        title = QLabel("Step 2 of 2 – Define Cockpit AOI")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #e0e0ff;")

        info = QLabel(
            "Click on the cockpit image to mark the boundary of the instrument panel (AOI).\n"
            "Right-click removes the last point.  Double-click to finish the polygon (need ≥ 3 points)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 13px; color: #aaaacc;")

        self._rms_label = QLabel("")
        self._rms_label.setStyleSheet("font-size: 12px; color: #88ff88;")

        self._aoi_editor = AOIEditor(self._pixmap)
        self._aoi_editor.polygon_changed.connect(self._update_nav)

        # If a previous AOI exists, pre-fill it
        if self._ctrl.aoi:
            self._aoi_editor.set_polygon(self._ctrl.aoi)

        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(self._rms_label)
        layout.addWidget(self._aoi_editor, 1)

        self._stack.addWidget(self._step2_container)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_gaze_cal_done(self, mapper: GazeMapper, rms: float) -> None:
        self._mapper = mapper
        self._rms = rms

        colour = "#ff6666" if rms > _WARN_RMS else "#88ff88"
        self._rms_label.setText(
            f"Gaze mapping RMS error: <b style='color:{colour}'>{rms:.4f}</b>"
            + (" (poor – recalibrate)" if rms > _WARN_RMS else " ✓")
        )
        self._btn_next.setText("Next: Draw AOI →")
        self._btn_next.setEnabled(True)
        self._update_nav()

    def _on_gaze_cal_failed(self, reason: str) -> None:
        QMessageBox.warning(self, "Calibration Failed", reason)
        self._btn_next.setText("Retry Calibration")
        self._btn_next.setEnabled(True)
        self._step1_started = False

    def _on_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx == 0:
            if not self._step1_started:
                self._step1_started = True
                self._btn_next.setEnabled(False)
                self._btn_next.setText("Calibrating…")
                self._gaze_cal_widget.start()
            elif self._mapper is not None:
                # Move to step 2
                self._stack.setCurrentIndex(1)
                self._update_nav()
        elif idx == 1:
            self._save_and_finish()

    def _on_back(self) -> None:
        if self._stack.currentIndex() == 1:
            self._stack.setCurrentIndex(0)
            self._gaze_cal_widget.stop()
            self._step1_started = False
            self._btn_next.setText("Start Calibration")
            self._btn_next.setEnabled(True)
            self._update_nav()

    def _on_cancel(self) -> None:
        self._gaze_cal_widget.stop()
        self.cancelled.emit()

    def _save_and_finish(self) -> None:
        if self._mapper is None:
            QMessageBox.warning(self, "Not calibrated", "Complete gaze calibration first.")
            return
        if not self._aoi_editor.is_valid:
            QMessageBox.warning(self, "AOI incomplete", "Draw at least 3 polygon points.")
            return

        self._ctrl.save_calibration_data(
            mapper=self._mapper,
            aoi=self._aoi_editor.polygon_norm,
            rms=self._rms,
            profile_name=self._config.profile_name,
        )
        self.calibration_saved.emit()

    def _update_nav(self) -> None:
        idx = self._stack.currentIndex()
        self._btn_back.setEnabled(idx > 0)

        if idx == 0:
            if not self._step1_started:
                self._btn_next.setText("Start Calibration")
                self._btn_next.setEnabled(True)
            elif self._mapper is not None:
                self._btn_next.setText("Next: Draw AOI →")
                self._btn_next.setEnabled(True)
            else:
                self._btn_next.setEnabled(False)
        elif idx == 1:
            self._btn_next.setText("Save Calibration")
            self._btn_next.setEnabled(self._aoi_editor.is_valid)
