"""Main application window – stacked screens for all app states."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import Config
from app.controller import Controller
from storage.calibration_store import list_profiles
from ui.calibration_wizard import CalibrationWizard
from ui.debrief_screen import DebriefScreen
from ui.session_screen import SessionScreen

logger = logging.getLogger(__name__)

# Screen indices in the stacked widget
_IDX_HOME = 0
_IDX_CALIBRATE = 1
_IDX_SESSION = 2
_IDX_DEBRIEF = 3

_ASSETS_DIR = Path("assets")
_COCKPIT_CANDIDATES = [
    _ASSETS_DIR / "cockpit.jpg",
    _ASSETS_DIR / "cockpit.png",
    _ASSETS_DIR / "cockpit.jpeg",
]


def _load_cockpit_pixmap() -> QPixmap:
    for p in _COCKPIT_CANDIDATES:
        if p.exists():
            px = QPixmap(str(p))
            if not px.isNull():
                logger.info("Cockpit image loaded: %s", p)
                return px
    logger.warning("No cockpit image found; using placeholder.")
    return QPixmap()  # empty – canvas will draw placeholder text


class ThresholdsDialog(QDialog):
    """Operator settings panel for tuning vision pipeline thresholds."""

    def __init__(self, config: Config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thresholds & Settings")
        self.setMinimumWidth(320)
        self._config = config

        form = QFormLayout()

        self._min_conf = QDoubleSpinBox()
        self._min_conf.setRange(0.0, 1.0)
        self._min_conf.setSingleStep(0.05)
        self._min_conf.setValue(config.min_confidence)
        form.addRow("Min Confidence:", self._min_conf)

        self._ema = QDoubleSpinBox()
        self._ema.setRange(0.01, 1.0)
        self._ema.setSingleStep(0.05)
        self._ema.setValue(config.ema_alpha)
        form.addRow("EMA Alpha (smoothing):", self._ema)

        self._stable = QDoubleSpinBox()
        self._stable.setRange(0, 2000)
        self._stable.setSingleStep(50)
        self._stable.setSuffix(" ms")
        self._stable.setValue(config.stable_ms)
        form.addRow("Stable ms (debounce):", self._stable)

        self._pause = QDoubleSpinBox()
        self._pause.setRange(0.5, 30.0)
        self._pause.setSingleStep(0.5)
        self._pause.setSuffix(" s")
        self._pause.setValue(config.auto_pause_seconds)
        form.addRow("Auto-pause after:", self._pause)

        self._cam_idx = QSpinBox()
        self._cam_idx.setRange(0, 9)
        self._cam_idx.setValue(config.camera_index)
        form.addRow("Camera index:", self._cam_idx)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _apply(self) -> None:
        self._config.min_confidence = self._min_conf.value()
        self._config.ema_alpha = self._ema.value()
        self._config.stable_ms = self._stable.value()
        self._config.auto_pause_seconds = self._pause.value()
        self._config.camera_index = self._cam_idx.value()
        self._config.save()
        self.accept()


class HomeScreen(QWidget):
    """Landing screen with profile selector and mode chooser."""

    start_session = Signal(str, str)   # mode, profile_name
    start_calibration = Signal(str)    # profile_name
    open_settings = Signal()

    def __init__(self, config: Config, controller: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Title
        title = QLabel("Eye Tracking Aviation")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #e8e8ff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Pilot Gaze Analysis System")
        sub.setStyleSheet("font-size: 14px; color: #8888aa;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(sub)

        # ── Profile selector ──────────────────────────────────────────
        profile_frame = QFrame()
        profile_frame.setStyleSheet("QFrame { background: #1e2240; border-radius: 10px; }")
        pfl = QFormLayout(profile_frame)
        pfl.setContentsMargins(24, 16, 24, 16)
        pfl.setSpacing(12)

        self._profile_combo = QComboBox()
        self._profile_combo.setEditable(True)
        self._profile_combo.setMinimumWidth(200)
        self._refresh_profiles()
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)

        self._calib_status = QLabel()
        self._update_calib_status()

        pfl.addRow("Profile:", self._profile_combo)
        pfl.addRow("Calibration:", self._calib_status)

        layout.addWidget(profile_frame, 0, Qt.AlignmentFlag.AlignHCenter)

        # ── Mode selector ─────────────────────────────────────────────
        mode_frame = QFrame()
        mode_frame.setStyleSheet("QFrame { background: #1e2240; border-radius: 10px; }")
        mfl = QVBoxLayout(mode_frame)
        mfl.setContentsMargins(24, 16, 24, 16)

        mode_title = QLabel("Session Mode")
        mode_title.setStyleSheet("font-weight: bold; color: #aaaaee;")

        self._radio_debug = QRadioButton("Debug  –  gaze overlay + live metrics (operator view)")
        self._radio_test = QRadioButton("Test  –  no overlay visible to participant")
        self._radio_debug.setChecked(True)

        mfl.addWidget(mode_title)
        mfl.addWidget(self._radio_debug)
        mfl.addWidget(self._radio_test)
        layout.addWidget(mode_frame, 0, Qt.AlignmentFlag.AlignHCenter)

        # ── Action buttons ────────────────────────────────────────────
        self._btn_calibrate = QPushButton("⚙  Calibrate")
        self._btn_calibrate.setFixedSize(180, 44)
        self._btn_calibrate.setStyleSheet(
            "background: #2244aa; color: white; font-size: 13px;"
            "border: none; border-radius: 6px; font-weight: bold;"
        )
        self._btn_calibrate.clicked.connect(
            lambda: self.start_calibration.emit(self._current_profile())
        )

        self._btn_start = QPushButton("▶  Start Session")
        self._btn_start.setFixedSize(200, 48)
        self._btn_start.setStyleSheet(
            "background: #228833; color: white; font-size: 14px;"
            "border: none; border-radius: 6px; font-weight: bold;"
        )
        self._btn_start.clicked.connect(self._on_start)

        btn_settings = QPushButton("⚙ Settings")
        btn_settings.setStyleSheet("color: #aaaacc; border: none; font-size: 11px;")
        btn_settings.clicked.connect(self.open_settings)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_row.setSpacing(16)
        btn_row.addWidget(self._btn_calibrate)
        btn_row.addWidget(self._btn_start)

        layout.addLayout(btn_row)
        layout.addWidget(btn_settings, 0, Qt.AlignmentFlag.AlignRight)
        layout.addStretch()

    def refresh(self) -> None:
        self._refresh_profiles()
        self._update_calib_status()

    def _refresh_profiles(self) -> None:
        current = self._profile_combo.currentText() or self._config.profile_name
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        profiles = list_profiles()
        if not profiles:
            profiles = ["default"]
        self._profile_combo.addItems(profiles)
        if current in profiles:
            self._profile_combo.setCurrentText(current)
        elif profiles:
            self._profile_combo.setCurrentText(profiles[0])
        self._profile_combo.blockSignals(False)

    def _current_profile(self) -> str:
        return self._profile_combo.currentText().strip() or "default"

    def _on_profile_changed(self, name: str) -> None:
        self._config.profile_name = name
        self._ctrl.load_calibration(name)
        self._update_calib_status()

    def _update_calib_status(self) -> None:
        if self._ctrl.is_calibrated:
            rms = self._ctrl.calibration_rms
            self._calib_status.setText(f"✓ Calibrated  (RMS {rms:.4f})")
            self._calib_status.setStyleSheet("color: #44dd88;")
            self._btn_start.setEnabled(True)
        else:
            self._calib_status.setText("✗ Not calibrated – run calibration first")
            self._calib_status.setStyleSheet("color: #dd4444;")
            self._btn_start.setEnabled(False)

    def _on_start(self) -> None:
        mode = "debug" if self._radio_debug.isChecked() else "test"
        self.start_session.emit(mode, self._current_profile())


class MainWindow(QMainWindow):
    """Root application window."""

    def __init__(self, config: Config, controller: Controller) -> None:
        super().__init__()
        self._config = config
        self._controller = controller

        self.setWindowTitle("Eye Tracking Aviation – Gaze Analysis")
        self.resize(config.window_width, config.window_height)

        self._cockpit_pixmap = _load_cockpit_pixmap()

        # ── Stacked widget ─────────────────────────────────────────────
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Build screens
        self._home = HomeScreen(config, controller)
        self._home.start_calibration.connect(self._go_calibrate)
        self._home.start_session.connect(self._go_session)
        self._home.open_settings.connect(self._open_settings)

        self._stack.addWidget(self._home)           # 0
        self._stack.addWidget(QWidget())            # 1 – calibration placeholder
        self._stack.addWidget(QWidget())            # 2 – session placeholder
        self._stack.addWidget(QWidget())            # 3 – debrief placeholder

        self._stack.setCurrentIndex(_IDX_HOME)

        self._session_screen: Optional[SessionScreen] = None
        self._debrief_screen: Optional[DebriefScreen] = None

        # Dark stylesheet
        self._apply_dark_theme()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_calibrate(self, profile_name: str) -> None:
        wizard = CalibrationWizard(
            controller=self._controller,
            cockpit_pixmap=self._cockpit_pixmap,
            config=self._config,
        )
        wizard.calibration_saved.connect(self._on_calibration_saved)
        wizard.cancelled.connect(lambda: self._stack.setCurrentIndex(_IDX_HOME))

        self._stack.removeWidget(self._stack.widget(_IDX_CALIBRATE))
        self._stack.insertWidget(_IDX_CALIBRATE, wizard)
        self._stack.setCurrentIndex(_IDX_CALIBRATE)

    def _on_calibration_saved(self) -> None:
        self._home.refresh()
        self._stack.setCurrentIndex(_IDX_HOME)

    def _go_session(self, mode: str, profile_name: str) -> None:
        self._config.profile_name = profile_name

        screen = SessionScreen(
            controller=self._controller,
            cockpit_pixmap=self._cockpit_pixmap,
            mode=mode,
        )
        screen.session_ended.connect(self._on_session_ended)

        self._stack.removeWidget(self._stack.widget(_IDX_SESSION))
        self._stack.insertWidget(_IDX_SESSION, screen)
        self._session_screen = screen
        self._stack.setCurrentIndex(_IDX_SESSION)
        screen.start_session()

    def _on_session_ended(self, debrief: dict, session_dir: Path) -> None:
        debrief_screen = DebriefScreen(
            debrief=debrief,
            session_dir=session_dir,
            cockpit_pixmap=self._cockpit_pixmap,
        )
        debrief_screen.home_requested.connect(self._go_home)

        self._stack.removeWidget(self._stack.widget(_IDX_DEBRIEF))
        self._stack.insertWidget(_IDX_DEBRIEF, debrief_screen)
        self._debrief_screen = debrief_screen
        self._stack.setCurrentIndex(_IDX_DEBRIEF)

    def _go_home(self) -> None:
        self._home.refresh()
        self._stack.setCurrentIndex(_IDX_HOME)

    def _open_settings(self) -> None:
        dlg = ThresholdsDialog(self._config, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Apply updated config to controller
            self._controller.config = self._config
            self._controller.ema.alpha = self._config.ema_alpha
            self._controller.state_machine.stable_ms = self._config.stable_ms

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0e0e1e;
                color: #d0d0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background: #1e2240;
                color: #d0d0f0;
                border: 1px solid #334466;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover { background: #2a3060; }
            QPushButton:pressed { background: #151530; }
            QPushButton:disabled { color: #555566; background: #141425; }
            QComboBox {
                background: #1e2240;
                color: #d0d0f0;
                border: 1px solid #334466;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                background: #1e2240;
                color: #d0d0f0;
                selection-background-color: #2a3060;
            }
            QSlider::groove:horizontal {
                background: #2a2a4a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4455cc;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QGroupBox {
                border: 1px solid #334466;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QLabel { color: #d0d0f0; }
            QRadioButton { color: #d0d0f0; }
            QDoubleSpinBox, QSpinBox {
                background: #1e2240;
                color: #d0d0f0;
                border: 1px solid #334466;
                border-radius: 4px;
                padding: 2px 6px;
            }
            """
        )
