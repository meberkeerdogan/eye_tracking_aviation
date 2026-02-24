# CLAUDE.md – Eye Tracking Aviation

This file is read automatically by Claude Code at the start of every session.

---

## Project Overview

Desktop MVP application that tracks a pilot's gaze via webcam, classifies it as
IN_COCKPIT / OUT_OF_COCKPIT / UNKNOWN, records session data to flat files, and
generates a debrief report. Built with PySide6 + MediaPipe + scikit-learn.

---

## Python Environment

- **Always use the `.venv` in the project root** — created from Anaconda Python 3.13.9.
- Anaconda path: `C:\Users\berke\anaconda3\python.exe`
- System Python (`C:\Users\berke\AppData\Local\Programs\Python\Python313\python`) causes
  a PySide6 DLL crash — never use it for this project.
- Activate in Git Bash: `source .venv/Scripts/activate`
- Run app: `python app/main.py` (after activation) or double-click `run.bat`
- Run tests: `python -m pytest tests/ -v` or double-click `run_tests.bat`
- Install/update deps: `pip install -e ".[dev]"`

## Key Dependency Constraints

| Package | Constraint | Reason |
|---|---|---|
| PySide6 | `>=6.7,<6.10` | 6.10.x DLL crash on Windows with some runtimes |
| mediapipe | `>=0.10.30` | 0.10.30+ removed `mp.solutions` entirely |
| numpy | `>=1.26,<3` | mediapipe compatibility |

---

## Architecture

```
domain/       – Pure Python logic (no Qt, no camera). Safe to unit test.
vision/       – Camera thread + MediaPipe face tracker + gaze feature extraction + mapper
calibration/  – QWidget steps: 9-point gaze calibration + AOI polygon editor
storage/      – CSV/JSON file I/O (originally named io/ — renamed; io is a Python builtin)
app/          – Config dataclass, Controller (session lifecycle + worker thread), entry point
ui/           – QMainWindow + all screens (home, calibration wizard, session, debrief)
tests/        – pytest unit tests (no Qt, no camera required)
assets/       – cockpit.jpg (user-provided), face_landmarker.task (auto-downloaded ~10 MB)
profiles/     – Per-participant calibration.json files (git-ignored, .gitkeep present)
runs/         – Session output folders (git-ignored, .gitkeep present)
```

### Threading model
- Vision pipeline runs in a **background daemon thread** (`app/controller.py _worker_loop`).
- Results flow to the UI via `queue.Queue(maxsize=5)`, polled by a `QTimer` every 16 ms.
- Never call Qt widgets from the worker thread.

### Coordinate system
- All gaze and AOI coordinates are **normalised 0.0–1.0** relative to the session widget
  dimensions. This space is used consistently: calibration dot positions, mapper output,
  AOI polygon vertices, and replay playback.

### Screen navigation
- `QStackedWidget` in `ui/main_window.py`: index 0 = Home, 1 = Calibration, 2 = Session,
  3 = Debrief.

---

## MediaPipe – Tasks API (important)

`mp.solutions` was **completely removed** in mediapipe 0.10.30+.
`vision/face_tracker.py` uses the Tasks API:

```python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(path)),
    running_mode=vision.RunningMode.VIDEO,   # VIDEO mode needs monotonic timestamps
    ...
)
landmarker = vision.FaceLandmarker.create_from_options(options)
result = landmarker.detect_for_video(mp_image, ts_ms)   # ts_ms must be strictly increasing
```

- Model file: `assets/face_landmarker.task` (~10 MB, auto-downloaded on first run via
  `ensure_model()` in `vision/face_tracker.py`).
- `ensure_model()` is called in `app/main.py` **before** `QApplication` is created.

---

## Key Files

| File | Purpose |
|---|---|
| `app/main.py` | Entry point; calls `ensure_model()`, starts Qt app |
| `app/config.py` | Typed `Config` dataclass; persisted to `config.json` |
| `app/controller.py` | Session lifecycle + background vision worker thread |
| `domain/models.py` | `GazeState` enum, `GazeSample`, `StateEvent`, `VisionResult` dataclasses |
| `domain/state_machine.py` | Debounce logic (`stable_ms` hysteresis) |
| `domain/metrics.py` | `compute_debrief()` — aggregated session statistics |
| `vision/face_tracker.py` | MediaPipe Tasks API wrapper; `FaceResult` dataclass |
| `vision/gaze_mapper.py` | Polynomial ridge regression gaze mapper + EMA filter |
| `storage/session_writer.py` | Line-buffered CSV + JSON writer |
| `storage/calibration_store.py` | Load/save `profiles/<name>/calibration.json` |
| `ui/main_window.py` | `QMainWindow`, `QStackedWidget`, `ThresholdsDialog` |
| `ui/debrief_screen.py` | Stats cards, matplotlib charts, replay scrubber |
| `pyproject.toml` | Dependencies, build config, ruff/black/mypy/pytest settings |

---

## Config Defaults (app/config.py)

| Field | Default | Notes |
|---|---|---|
| `camera_index` | 0 | Webcam index |
| `min_confidence` | 0.30 | Below → UNKNOWN state |
| `ema_alpha` | 0.30 | Higher = more responsive, noisier |
| `stable_ms` | 200.0 | Debounce window in milliseconds |
| `auto_pause_seconds` | 3.0 | Warn if face lost this long |
| `calib_degree` | 2 | Polynomial degree for gaze regression |
| `calib_rms_warn` | 0.05 | Warn user if calibration RMS exceeds this |
| `profile_name` | "default" | Active calibration profile |
| `window_width/height` | 1280×800 | Main window size |

Config is persisted to `config.json` in the project root (git-ignored).

---

## Running Tests

```bash
source .venv/Scripts/activate
python -m pytest tests/ -v
```

Tests are in `tests/` and require **no camera, no Qt display**:
- `test_state_machine.py` – debounce/hysteresis logic
- `test_aoi.py` – polygon hit-test (cv2.pointPolygonTest)
- `test_metrics.py` – debrief statistics computation

All 13 tests should pass. If any fail, do not proceed with feature work until fixed.

---

## Common Pitfalls

1. **Never rename `storage/` back to `io/`** — `io` is a Python standard library module.
2. **Never use `mp.solutions`** — removed in mediapipe 0.10.30+; use Tasks API only.
3. **Never upgrade PySide6 to 6.10+** without testing the DLL compatibility first.
4. **Never call Qt UI methods from the vision worker thread** — use the result queue.
5. **Window must not be resized between calibration and session** — gaze coords would shift.
6. **Timestamps for FaceLandmarker VIDEO mode must be strictly monotonically increasing**
   (integer milliseconds). See `face_tracker.py:131–133`.

---

## Git Log (reference)

```
f651aff Fix: migrate to MediaPipe Tasks API (mp.solutions removed in 0.10.30+)
24b8f01 Fix PySide6 DLL crash: pin <6.10, rebuild venv from Anaconda Python
84a5835 Add LEARN.md, fix test timing precision, finalise GitHub prep
68cdcdb Initial commit: Eye Tracking Aviation MVP
```
