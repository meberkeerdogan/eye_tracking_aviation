"""Application-wide configuration with typed fields and sane defaults."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    # Camera
    camera_index: int = 0

    # Vision pipeline
    min_confidence: float = 0.30   # samples below this are UNKNOWN
    ema_alpha: float = 0.30        # EMA weight for new gaze sample (higher = more responsive)
    stable_ms: float = 200.0       # ms a state must be stable before committing

    # Auto-pause
    auto_pause_seconds: float = 3.0

    # Calibration
    calib_degree: int = 2          # polynomial degree for gaze mapper
    calib_ridge_alpha: float = 1.0
    calib_rms_warn: float = 0.05   # warn user if RMS > this value
    calib_dot_dwell_ms: int = 1500 # ms to collect samples per calibration dot
    calib_dot_settle_ms: int = 800  # ms to wait before collecting (eyes settle)

    # Session
    profile_name: str = "default"

    # UI
    window_width: int = 1280
    window_height: int = 800
    debug_show_landmarks: bool = False
    fps_target: int = 30

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        logger.debug("Config saved.")

    @classmethod
    def load(cls) -> "Config":
        if not _CONFIG_PATH.exists():
            return cls()
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = cls()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            logger.debug("Config loaded from %s", _CONFIG_PATH)
            return cfg
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load config (%s); using defaults.", exc)
            return cls()
