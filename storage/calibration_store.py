"""Load and save calibration profiles from/to disk."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path("profiles")


def profiles_dir() -> Path:
    return _PROFILES_DIR


def calibration_path(profile_name: str) -> Path:
    return _PROFILES_DIR / profile_name / "calibration.json"


def list_profiles() -> list[str]:
    """Return sorted profile names that have a calibration file."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted(
        d.name
        for d in _PROFILES_DIR.iterdir()
        if d.is_dir() and (d / "calibration.json").exists()
    )


def save_calibration(data: dict[str, Any], profile_name: str = "default") -> Path:
    path = calibration_path(profile_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Calibration saved: %s", path)
    return path


def load_calibration(profile_name: str = "default") -> Optional[dict[str, Any]]:
    path = calibration_path(profile_name)
    if not path.exists():
        logger.warning("No calibration found at %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Calibration loaded: %s", path)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load calibration %s: %s", path, exc)
        return None


def calibration_hash(data: dict[str, Any]) -> str:
    """Short SHA-256 hash of the calibration dict (for session meta)."""
    blob = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
