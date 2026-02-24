"""Application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on the path when running as `python app/main.py`
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Change working directory to project root so relative paths (profiles/, runs/)
# resolve correctly regardless of where the script is invoked from.
import os
os.chdir(_ROOT)

from PySide6.QtWidgets import QApplication, QMessageBox

from app.config import Config
from app.controller import Controller
from ui.main_window import MainWindow


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Eye Tracking Aviation – starting up.")

    app = QApplication(sys.argv)
    app.setApplicationName("EyeTrackingAviation")

    config = Config.load()

    controller = Controller(config)

    # Try to start camera
    try:
        controller.start_camera()
    except RuntimeError as exc:
        QMessageBox.critical(
            None,
            "Camera Error",
            f"Could not open webcam (index {config.camera_index}).\n\n{exc}\n\n"
            "Check that your webcam is connected and not in use by another application.",
        )
        sys.exit(1)

    # Auto-load existing calibration for default profile
    if not controller.load_calibration(config.profile_name):
        logger.info("No calibration found for profile '%s'.", config.profile_name)

    window = MainWindow(config, controller)
    window.show()

    ret = app.exec()

    controller.stop_camera()
    logger.info("Exiting with code %d.", ret)
    sys.exit(ret)


if __name__ == "__main__":
    main()
