# LEARN.md — Complete Guide to the Eye Tracking Aviation Project

> **Who this is for:** Someone learning to code who wants to understand every
> decision made in this project — from folder structure to algorithms to why
> each library was chosen.  Read top-to-bottom, or jump to any section.

---

## Table of Contents

1. [What does this application actually do?](#1-what-does-this-application-actually-do)
2. [Big picture: how all the pieces fit together](#2-big-picture-how-all-the-pieces-fit-together)
3. [Technology stack — every library explained](#3-technology-stack--every-library-explained)
4. [Project structure — every folder explained](#4-project-structure--every-folder-explained)
5. [Layer 1 — Domain models (`domain/`)](#5-layer-1--domain-models-domain)
6. [Layer 2 — Vision pipeline (`vision/`)](#6-layer-2--vision-pipeline-vision)
7. [Layer 3 — Calibration (`calibration/`)](#7-layer-3--calibration-calibration)
8. [Layer 4 — Storage (`storage/`)](#8-layer-4--storage-storage)
9. [Layer 5 — Application logic (`app/`)](#9-layer-5--application-logic-app)
10. [Layer 6 — User interface (`ui/`)](#10-layer-6--user-interface-ui)
11. [Threading — why and how](#11-threading--why-and-how)
12. [The state machine — debounce explained](#12-the-state-machine--debounce-explained)
13. [Gaze calibration — the math](#13-gaze-calibration--the-math)
14. [Coordinate systems — a common source of confusion](#14-coordinate-systems--a-common-source-of-confusion)
15. [Tests — what they check and why](#15-tests--what-they-check-and-why)
16. [Virtual environment and packaging](#16-virtual-environment-and-packaging)
17. [Git and GitHub setup](#17-git-and-github-setup)
18. [Key patterns you'll see everywhere](#18-key-patterns-youll-see-everywhere)
19. [How data flows from camera to debrief](#19-how-data-flows-from-camera-to-debrief)
20. [Known limitations and how to improve them](#20-known-limitations-and-how-to-improve-them)

---

## 1. What does this application actually do?

Imagine a flight simulator used for pilot training.  A researcher wants to know:
*"Is the trainee looking at the instrument panel, or looking out the window?"*

This application answers that by:

1. Using the **webcam** to see the pilot's face
2. Finding the **eyes** and specifically the **iris** (the coloured part of the eye)
3. Estimating **where on the screen** the pilot is looking
4. Deciding at every moment whether the gaze is **inside** the instrument panel
   area (the "cockpit AOI") or **outside** it
5. Recording everything to files so it can be **analysed later**
6. Showing a **debrief screen** at the end with charts and statistics

### The key problem it solves

Manually reviewing hours of video footage to count where a pilot looked is
extremely tedious.  This app automates it: connect a webcam, calibrate once,
run the session, get instant statistics.

---

## 2. Big picture: how all the pieces fit together

Here is the data journey from camera pixels to debrief report:

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  WEBCAM                                                              │
 │  Captures 30 frames/second (raw BGR images, 640×480)                │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  frame (numpy array)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  FaceTracker  (vision/face_tracker.py)                              │
 │  MediaPipe FaceMesh finds 478 face landmarks.                       │
 │  We extract: left_iris (x,y), right_iris (x,y), confidence         │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  FaceResult
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  GazeFeatureExtractor  (vision/gaze_features.py)                    │
 │  Turns the landmarks into a 20-number feature vector                │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  numpy array [20 floats]
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  GazeMapper  (vision/gaze_mapper.py)                                │
 │  Machine learning model trained during calibration.                 │
 │  Maps feature vector → (gaze_x, gaze_y) in 0..1 screen coords      │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  (gaze_x_norm, gaze_y_norm)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  EMAFilter  (vision/gaze_mapper.py)                                 │
 │  Smooth out jitter with an Exponential Moving Average              │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  smoothed (gaze_x, gaze_y)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  AOI Hit Test  (app/controller.py)                                  │
 │  Is the gaze point inside the cockpit polygon?                      │
 │  → IN_COCKPIT / OUT_OF_COCKPIT / UNKNOWN                           │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  raw_state (GazeState enum)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  StateMachine  (domain/state_machine.py)                            │
 │  Debounce: only commit a state change after it's stable for 200ms   │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  committed_state (GazeState enum)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  SessionWriter  (storage/session_writer.py)                         │
 │  Writes one row to samples.csv and events.csv                       │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  (at session end)
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Metrics  (domain/metrics.py)                                       │
 │  Computes totals, percentages, histograms                           │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │  debrief dict
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  DebriefScreen  (ui/debrief_screen.py)                              │
 │  Charts, replay, export buttons                                     │
 └──────────────────────────────────────────────────────────────────────┘
```

All of the vision processing happens in a **background thread** so the UI
stays smooth and responsive.  The thread communicates with the UI via a
thread-safe **queue** (explained in Section 11).

---

## 3. Technology stack — every library explained

### Python 3.11+
The programming language.  We use several modern features:
- **Type hints** (`def foo(x: int) -> str`) — not enforced at runtime but
  make code much easier to read and catch bugs with tools like mypy
- **Dataclasses** — a concise way to define simple data-holding classes
- **f-strings** — `f"value is {x:.2f}"` for clean string formatting

### OpenCV (`opencv-python`)
**What:** Open Computer Vision library.  One of the most-used libraries in
robotics and computer vision.

**Why here:** We use it for two things:
1. Converting frames from BGR (camera format) to RGB (what MediaPipe expects)
2. `cv2.pointPolygonTest` — a single function that checks if a 2-D point is
   inside a polygon.  It uses a classical ray-casting algorithm.

**Alternative:** We could use `shapely` for hit-testing, but opencv is already
in the dependency tree for camera work, so adding a separate library for one
function is wasteful.

### MediaPipe (`mediapipe`)
**What:** Google's machine learning library for real-time body tracking.

**Why here:** The `FaceMesh` solution detects 478 face landmarks at ~30 fps
on a CPU without needing a GPU.  Crucially it supports `refine_landmarks=True`
which adds **iris tracking** — it can tell us the centre of each iris, which
is what we use as a proxy for gaze direction.

**What we do NOT use:** MediaPipe has a `GazePrediction` solution, but it
requires proprietary models not publicly available.  Instead we use the iris
positions as raw features and train our own simple regression model.

**Landmarks:** MediaPipe gives 478 (x,y,z) coordinates in normalised frame
space (0.0 to 1.0).  The iris landmarks are indices 469-477.

### scikit-learn (`scikit-learn`)
**What:** Python's most popular machine learning library.

**Why here:** We use it to fit the **gaze mapping model** — a polynomial
ridge regression that learns the relationship between eye positions and screen
positions during calibration.

**Specifically:**
- `PolynomialFeatures(degree=2)` — transforms our 20-input features into ~230
  quadratic combinations (e.g., `feature_3 × feature_7`, `feature_1²`), which
  lets the linear model capture curved relationships
- `Ridge(alpha=1.0)` — a linear regression that penalises large coefficients
  (regularisation), preventing over-fitting on our small ~270-sample calibration
- `StandardScaler` — centres each feature around 0 and scales to unit variance,
  which is required for Ridge regression to work properly
- `Pipeline` — chains scaler → poly features → ridge into a single object

**Why not a neural network?** With only 9 calibration points × ~30 samples
each = ~270 total data points, a neural network would massively over-fit.
Polynomial ridge regression is the right tool: it adds enough non-linearity to
fit the curved gaze-to-screen mapping while staying generalised.

### PySide6 (`PySide6`)
**What:** Python bindings for Qt 6 — the cross-platform GUI framework used in
applications like VLC, Autodesk tools, and many others.

**Why not tkinter?** tkinter is Python's built-in GUI library but is very
limited for custom drawing, animations, and overlay widgets.

**Why not PyQt6?** PySide6 is the official Qt binding (from Qt itself), while
PyQt6 is a third-party binding.  They are nearly identical but PySide6 has a
more permissive licence for non-commercial use.

**Key Qt concepts used:**
- `QMainWindow` — the top-level window
- `QStackedWidget` — holds multiple "pages"; only one is visible at a time
  (we use this for home / calibration / session / debrief screens)
- `QWidget` — the base class for every visual element
- `QPainter` — low-level drawing API for custom graphics (gaze dot, calibration
  dots, AOI polygon)
- `QTimer` — fires a function at a regular interval (we use this to poll the
  vision queue at ~60 Hz)
- `Signal/Slot` — Qt's event system.  A signal is like a radio broadcast; a
  slot is a function that "tunes in" to that broadcast

### Matplotlib (`matplotlib`)
**What:** Python's most popular plotting library.

**Why here:** For the debrief charts (pie chart, OUT duration histogram, state
timeline).  It integrates with Qt via `FigureCanvasQTAgg` which lets us embed
a matplotlib figure directly inside a `QWidget`.

### NumPy (`numpy`)
**What:** Numerical Python — provides fast N-dimensional arrays.

**Why here:** All feature vectors and training matrices are NumPy arrays.
scikit-learn and MediaPipe both return/accept NumPy arrays natively, so using
NumPy is the natural choice throughout the vision pipeline.

---

## 4. Project structure — every folder explained

```
eye_tracking_aviation/
│
│   app/          ← The "brain" — config, controller, entry point
│   calibration/  ← Calibration UI widgets (gaze dots + polygon editor)
│   domain/       ← Pure Python business logic (no dependencies on Qt/CV)
│   storage/      ← File reading and writing
│   tests/        ← Automated tests
│   ui/           ← All PySide6 screens
│   vision/       ← Camera and computer vision code
│
│   assets/       ← Static files (cockpit image)
│   profiles/     ← Per-participant calibration data
│   runs/         ← Session output data
│
│   pyproject.toml   ← Packaging + dependency definition
│   README.md        ← Quick-start guide
│   LEARN.md         ← This file
│   .gitignore       ← What not to commit to Git
```

### Why this layer structure?

This is called **separation of concerns** — each folder has one clear
responsibility and the layers only import "downward":

```
ui  →  app  →  domain       (no arrows going the other way)
          ↓       ↑
       vision   storage
          ↓
      calibration
```

- `domain/` knows nothing about Qt, OpenCV, or files — it is pure logic
- `vision/` knows about cameras and MediaPipe but nothing about Qt
- `storage/` knows about files but nothing about Qt
- `app/` brings everything together
- `ui/` sits at the top and depends on everything else

**Benefit:** If you wanted to replace PySide6 with a web frontend, you would
only need to rewrite the `ui/` folder.  Everything else stays the same.

---

## 5. Layer 1 — Domain models (`domain/`)

The `domain/` folder contains the core vocabulary of the application: the
data types and logic rules that exist regardless of how the UI is built or
what files are used.

### `domain/models.py` — data structures

```python
class GazeState(str, Enum):
    IN_COCKPIT = "IN_COCKPIT"
    OUT_OF_COCKPIT = "OUT_OF_COCKPIT"
    UNKNOWN = "UNKNOWN"
```

An **Enum** (enumeration) is a set of named values.  Using an Enum instead of
raw strings like `"in"` and `"out"` prevents typos and makes the IDE autocomplete.

Inheriting from both `str` and `Enum` means a `GazeState` value can be used
directly where a string is expected (useful for CSV writing).

```python
@dataclass
class GazeSample:
    timestamp_mono: float
    timestamp_wall: float
    gaze_x_norm: float
    gaze_y_norm: float
    confidence: float
    state: GazeState
```

A **dataclass** automatically generates `__init__`, `__repr__`, and `__eq__`
methods.  It is a concise way to define a class that is mainly used to hold
data.

**Why two timestamps?**
- `timestamp_mono` — from `time.monotonic()`.  This clock never goes backwards
  and is not affected by system clock changes.  It is ideal for measuring
  *durations*.
- `timestamp_wall` — from `time.time()`.  This is the real-world calendar time.
  It is ideal for showing the user "this session started at 14:32".

### `domain/state_machine.py` — debounce logic

See [Section 12](#12-the-state-machine--debounce-explained) for the detailed
explanation.

### `domain/metrics.py` — debrief statistics

This module takes the list of samples and transition events and computes
statistics like percentages and histogram data.  It is a pure function
with no side effects — it reads data and returns a dictionary.

**Why a separate module?** Keeping calculation logic separate from both the
UI and the file I/O means you can:
- Test it without starting a UI
- Run it on historical data without a camera
- Change the calculation without touching any UI code

---

## 6. Layer 2 — Vision pipeline (`vision/`)

### `vision/camera.py` — threaded frame capture

```python
class Camera:
    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        ...
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
```

`cv2.VideoCapture.read()` is a **blocking call** — it waits until the camera
delivers a frame.  If we called it from the main thread, the entire UI would
freeze while waiting.

Instead, a dedicated background thread calls `read()` in a loop and stores the
latest frame.  Any other code can call `camera.get_frame()` at any time and
get the most recent frame instantly, without waiting.

**Thread safety:** The frame is protected by a `threading.Lock()`.  The lock
ensures that if the capture thread is writing a new frame at the exact same
moment another thread is reading, they do not interfere and produce a corrupted
result.

**`daemon=True`:** When the main program exits, daemon threads are automatically
killed.  Without this, the program would hang waiting for the camera thread to
finish.

### `vision/face_tracker.py` — MediaPipe wrapper

```python
self._face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # enables iris tracking (landmarks 469-477)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

MediaPipe uses a two-stage pipeline:
1. **Detection** — finds a face in the frame from scratch (slower, run when
   tracking is lost)
2. **Tracking** — tracks the already-found face using optical flow (faster,
   run every subsequent frame)

The `min_*_confidence` values control how certain MediaPipe must be before
it returns a result.

**What we extract:**
- `left_iris` and `right_iris` — the (x,y) centre of each iris in normalised
  frame coordinates (0.0 to 1.0 relative to the frame width/height)
- `confidence` — we derive this from **eye openness ratio**.  A fully open eye
  has a vertical span ≈ 15% of the horizontal span.  If the eyes are mostly
  closed (blinking), the gaze estimate will be unreliable, so confidence drops.

### `vision/gaze_features.py` — feature vector

The raw iris positions change when you move your head without changing where
you look.  To make the mapping more robust, we extract features that capture
**gaze direction** rather than absolute eye position:

```
Feature vector (20 numbers):
  [0-3]   Absolute iris x,y for both eyes
  [4-7]   Iris position RELATIVE to the eye corner bounding box
           (this is the gaze direction proxy — how far across the eye the iris is)
  [8-11]  Eye width and height (to understand face scale)
  [12-17] Head position proxies: nose tip, chin, forehead
  [18-19] Mean iris position
```

**Eye-relative position:** If your iris is exactly centred between the eye
corners, you're looking straight ahead regardless of whether you're close or
far from the camera.  If it's shifted to the right, you're looking right.
This is the key insight that makes calibration generalisable across small head
movements.

### `vision/gaze_mapper.py` — regression model and EMA

See [Section 13](#13-gaze-calibration--the-math) for the full math explanation.

**EMAFilter — Exponential Moving Average:**

Without smoothing, the gaze dot on screen jiggles rapidly because:
1. MediaPipe landmark coordinates have small per-frame noise
2. The regression model amplifies small input changes

EMA solves this by making the current output a weighted blend of the new
measurement and the previous output:

```
smoothed_x = alpha * new_x + (1 - alpha) * previous_smoothed_x
```

With `alpha = 0.3`:
- 30% of the new measurement
- 70% of the historical smoothed value

A **smaller alpha** = more smoothing = smoother dot, but slower to react to
real eye movements.

A **larger alpha** = less smoothing = more responsive, but jittery.

The default `alpha=0.3` is a reasonable starting point; users can tune it in
the Thresholds panel.

---

## 7. Layer 3 — Calibration (`calibration/`)

### `calibration/gaze_calibration.py` — 9-point calibration widget

This is a `QWidget` that manages the calibration dot routine using a tiny
state machine of its own:

```
IDLE → SETTLE → SAMPLE → BLINK → (next point) → ... → FITTING → DONE
```

- **IDLE:** Waiting for the user to click Start
- **SETTLE (900ms):** Dot appears; we wait for the user's eyes to move to it
  and stop moving.  We do NOT collect samples yet.
- **SAMPLE (1200ms):** We grab ~36 frames from the camera, run face tracking
  on each, and collect feature vectors.  We average them to get one robust
  sample per calibration point.
- **BLINK (300ms):** Brief fade-out between dots so the user knows to look at
  the next one
- **FITTING:** After all 9 dots, train the regression model
- **DONE:** Emit the finished model back to the wizard

**Why 9 points?**  A 3×3 grid covers the corners, edges, and centre of the
screen.  This ensures the model is calibrated across the full range of head
and eye positions.  More points → better accuracy but more tiring to calibrate.

**Why average instead of using all samples?**  Each point produces ~36 raw
feature vectors.  Using all of them would make the regression model "think" the
corner points are 36× more important than the centre.  Averaging to one
representative vector per point keeps the training set balanced.

### `calibration/aoi_editor.py` — polygon editor

The AOI (Area of Interest) editor lets the operator draw a polygon directly on
the cockpit image to mark the instrument panel boundary.

**Click handling:**
- **Left click:** Add a vertex at the click position, converted to normalised
  coordinates: `x_norm = click_x / widget_width`
- **Right click:** Remove the last vertex (undo)
- **Double click:** Close the polygon (signal that drawing is complete)

**Why normalised coords?**  If we stored pixel coordinates like (640, 350),
the AOI would break if the window was resized.  Storing (0.5, 0.44) means
"50% across, 44% down" — this works at any window size.

**Custom drawing with QPainter:**
We override `paintEvent` to draw:
1. The cockpit image scaled to fill the widget
2. A semi-transparent green filled polygon
3. Yellow dots at each vertex
4. Numbered vertex labels
5. A dashed closing line from last point back to first

---

## 8. Layer 4 — Storage (`storage/`)

> **Note:** This folder was originally named `io/` but had to be renamed
> because Python has a built-in module called `io`.  Our folder would have
> shadowed it, causing import errors.  This is a real-world packaging gotcha.

### `storage/session_writer.py` — writing session files

The writer opens CSV files in **line-buffered mode** (`buffering=1`):

```python
self._sf = open(..., buffering=1)
```

In Python's `open()`, `buffering=1` means "write to disk after every newline".
This is slightly slower than the default (which batches writes), but it means
if the program crashes mid-session, you still have all samples up to the last
complete line — not a partially-written buffer that would be lost.

**CSV vs JSON for samples:**  We use CSV for `samples.csv` (many rows, tabular
data) because it's compact, fast to write, and trivially openable in Excel.
We use JSON for `debrief.json` and `session_meta.json` because they are
structured documents with nested data, not flat rows.

### `storage/calibration_store.py` — profile management

Calibration is expensive (takes ~30 seconds), so we save it to disk using
JSON and reload it automatically on startup.

The `calibration_hash()` function computes a 12-character fingerprint of the
calibration data and stores it in `session_meta.json`.  This lets you later
verify which calibration was used for which session.

---

## 9. Layer 5 — Application logic (`app/`)

### `app/config.py` — typed configuration

```python
@dataclass
class Config:
    camera_index: int = 0
    min_confidence: float = 0.30
    ema_alpha: float = 0.30
    stable_ms: float = 200.0
    ...
```

All tunable parameters are in one place with explicit types and default values.
The `Config.load()` / `Config.save()` methods persist it to `config.json`.

**Why not hard-code values directly in the code?**
If thresholds are scattered across multiple files, changing them requires
finding every location.  With a central `Config`, you change one number and
every module that uses `config.min_confidence` gets the new value.

### `app/controller.py` — session lifecycle

The `Controller` is the central coordinator of the application.  It owns:
- The `Camera`
- The `FaceTracker`
- The `GazeMapper` and `EMAFilter`
- The `StateMachine`
- The `SessionWriter`
- The background worker thread

The vision processing loop runs inside `_worker_loop()` in a background thread:

```python
def _worker_loop(self) -> None:
    while self._worker_running:
        frame_data = self.camera.get_frame()
        face = self.face_tracker.process(frame_rgb)
        features = extract_features(face)
        gx, gy = self.gaze_mapper.predict(features)
        gx, gy = self.ema.update(gx, gy)
        committed = self.state_machine.update(raw_state, mono_ts)
        self.session_writer.write_sample(sample)
        self._result_queue.put_nowait(result)
```

The `_result_queue` is a `queue.Queue(maxsize=5)`.  If the UI hasn't read
recent results yet and the queue fills up, new results are simply dropped
(`put_nowait` does not block).  A few dropped frames are invisible to the user
but prevents the queue from growing indefinitely.

### `app/main.py` — entry point

```python
def main() -> None:
    app = QApplication(sys.argv)  # required before any Qt widget is created
    config = Config.load()
    controller = Controller(config)
    controller.start_camera()     # raises RuntimeError if no camera found
    window = MainWindow(config, controller)
    window.show()
    ret = app.exec()              # enters the Qt event loop (blocks here)
    controller.stop_camera()      # cleanup when window is closed
    sys.exit(ret)
```

`app.exec()` starts Qt's **event loop** — a loop that waits for user input
(mouse clicks, key presses), timer events, and repaints, and dispatches them to
the right handlers.  Everything in the UI is driven by this loop.

---

## 10. Layer 6 — User interface (`ui/`)

### Screen navigation with `QStackedWidget`

The main window contains a `QStackedWidget` — think of it as a stack of
cards where only the top card is visible.  Switching screens means changing
which card is on top:

```
Index 0: HomeScreen         ← profile select, mode select, start buttons
Index 1: CalibrationWizard  ← 9-point gaze cal + AOI editor
Index 2: SessionScreen      ← cockpit image + recording
Index 3: DebriefScreen      ← statistics + charts + replay
```

### `ui/main_window.py` — the shell

`MainWindow` owns the stacked widget and handles navigation between screens.
It also hosts the `ThresholdsDialog` — an operator settings panel with sliders
for the key vision pipeline parameters.

### `ui/session_screen.py` — the recording screen

The session screen has two layers stacked on top of each other:
1. `CockpitCanvas` — draws the cockpit image scaled to fill the widget
2. `DebugOverlay` — a transparent widget drawn on top, showing the gaze dot and
   metrics (only in debug mode)

The transparent overlay uses:
```python
self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
```
`WA_TranslucentBackground` means the widget's background is transparent (you
can see through it to the `CockpitCanvas` behind it).  `WA_TransparentForMouseEvents`
means mouse clicks pass straight through to the `CockpitCanvas` below.

**The M key for markers:**
```python
def keyPressEvent(self, event: QKeyEvent) -> None:
    if event.key() == Qt.Key.Key_M and self._running:
        self._ctrl.add_marker("manual")
```
Qt routes keyboard events to the widget that has **focus**.  We call
`self.setFocus()` when the session starts to ensure key presses go here.

### `ui/debrief_screen.py` — the results screen

**Stat cards:** Each `StatCard` is a small `QFrame` with custom styling.
`QFrame.setStyleSheet("background: #1e2240; border-radius: 8px;")` uses a
subset of CSS to style Qt widgets — this is called **Qt Style Sheets (QSS)**.

**Matplotlib in Qt:** The trick is `FigureCanvasQTAgg` — matplotlib's Qt
backend converts a matplotlib `Figure` into a regular `QWidget` that can be
placed inside any Qt layout like any other widget.

**Replay:** The replay scrubber is a `QSlider`.  When the user drags it, we
read the corresponding row from the pre-loaded CSV data and update the `ReplayCanvas`
to show the gaze dot at that sample's position.  The Play button starts a
`QTimer` that automatically advances the index every 33ms (≈30fps).

---

## 11. Threading — why and how

### The problem: the GUI must never block

Qt's event loop runs on the **main thread**.  If any code on the main thread
takes more than ~16ms (the length of one 60fps frame), the UI freezes —
buttons stop responding, the window goes white on Windows ("Not Responding").

Processing one camera frame takes ~20-50ms (MediaPipe + regression prediction).
At 30fps we need to process a new frame every 33ms.  That's already on the
edge without any safety margin.

### The solution: worker thread + queue

```
Main thread (Qt event loop)                Background thread (VisionWorker)
───────────────────────────                ─────────────────────────────────
QTimer fires every 16ms           ←───    Puts VisionResult into queue
  while (queue.get_nowait()):
    update overlay with result

User clicks button                         Camera captures frame
  → session starts                         → face tracking
  → starts worker thread                   → feature extraction
                                           → gaze prediction
                                           → state machine
                                           → write to CSV
                                           → queue.put(result)
```

The `queue.Queue` is the safe bridge between threads.  `Queue` is
**thread-safe** — it handles all the locking internally so you never have to
worry about two threads accessing it at the same time.

### Why `QTimer` for polling?

We could use a Qt signal from the worker thread, but signals across threads in
Qt have some complexity.  Polling with a `QTimer` every 16ms is simple,
reliable, and fast enough that the user never notices the difference.

### `daemon=True` threads

All worker threads are created with `daemon=True`.  This means:
- They automatically terminate when the main program exits
- You don't need to manually signal them to stop (though we do stop them cleanly
  anyway for good practice)

---

## 12. The state machine — debounce explained

### The problem: noisy gaze signals

Eye tracking is noisy.  Even if you are clearly looking at the instrument panel,
the gaze estimate might briefly jump outside it for a single frame (due to a
blink, a head microtremor, or noise in the regression).

Without any filtering, you would see hundreds of tiny "out-of-cockpit glances"
that never actually happened.  The statistics would be meaningless.

### The solution: debounce / hysteresis

The state machine only commits a state change after the new state has been
**stable for at least `stable_ms` milliseconds** (default 200ms).

```
Time →    0ms     50ms    100ms   150ms   200ms   250ms   300ms
          ─────────────────────────────────────────────────────
Raw:      IN      IN      OUT     IN      IN      IN      IN
                          ↑
                  Brief noise spike (one frame outside AOI)

Pending:           -       OUT    -       -       -       -
                           started but then the raw went back to IN
                           so the pending timer resets

Committed:  UNKNOWN → IN_COCKPIT (committed at 200ms after first IN)
```

In this example, the brief OUT spike at 100ms never commits because it lasted
less than 200ms.

**The implementation:**

```python
def update(self, candidate: GazeState, mono_time: float) -> GazeState:
    if candidate == self._committed:
        # Still in the same state → clear any pending change
        self._pending = None
        self._pending_since = None
        return self._committed

    if candidate != self._pending:
        # New different state → start timing it
        self._pending = candidate
        self._pending_since = mono_time
        return self._committed

    # Same candidate as pending → check if stable enough
    elapsed_ms = (mono_time - self._pending_since) * 1000.0
    if elapsed_ms >= self.stable_ms:
        self._commit(candidate, mono_time)  # officially change state

    return self._committed  # still returning the OLD committed state until commit
```

The key insight: **we always return the committed (stable) state**, not the
raw noisy candidate.  The metrics are computed from committed states only.

---

## 13. Gaze calibration — the math

### The problem

The camera sees your iris at position (0.47, 0.52) in frame space.  But
where on the screen are you looking?  The camera cannot directly answer this —
it depends on:
- How far you are from the screen
- The angle of your head
- Your individual anatomy

### Calibration: building a personal map

During calibration, we show you dots at **known screen positions** and measure
your **iris positions** at each dot.  This gives us training data:

```
Input (features):  what your eyes look like when looking at a known point
Output (targets):  the screen position of that known point
```

We show 9 dots, collect ~30 samples per dot, average each to get 9 reliable
feature vectors, then fit a **polynomial ridge regression** model.

### Why polynomial?

The relationship between iris position and screen position is not perfectly
linear.  When you look to the extreme right, your iris doesn't just move
proportionally to the right — the curvature of the eyeball and perspective
distortion create slight non-linearities.

Polynomial features expand our 20 input features into ~230 features including
squares (`x²`, `y²`) and cross-terms (`x × y`).  This lets the linear
regression model capture these non-linear relationships.

### Why Ridge regularisation?

After polynomial expansion, we have ~230 features but only 9 training samples.
Ordinary least squares regression would wildly over-fit — it would perfectly
fit all 9 training points but fail completely on new data.

Ridge regression adds a penalty term: it tries to keep the coefficients small.
This "shrinks" the model towards simpler solutions that generalise better.

```
Ridge loss = (prediction error)² + alpha × (sum of squared coefficients)
```

With `alpha=1.0`, the model balances fitting the training data against keeping
the model simple.

### Serialisation with pickle

After fitting, we serialize the trained model to include it in the calibration
JSON file:

```python
"pipe_x": base64.b64encode(pickle.dumps(self._pipe_x)).decode()
```

1. `pickle.dumps()` — converts the Python sklearn Pipeline object into bytes
2. `base64.b64encode()` — converts bytes to a safe ASCII string
   (JSON cannot contain raw binary data)
3. `decode()` — converts the base64 bytes to a Python string so it can be
   stored in JSON

Loading works in reverse: `base64.b64decode()` → `pickle.loads()`.

**Security note:** Never load pickle data from untrusted sources.  Pickle can
execute arbitrary code.  In this application, calibration files are only ever
generated and loaded by the same program, so it is safe.

---

## 14. Coordinate systems — a common source of confusion

This project uses three different coordinate spaces.  Confusing them is a
common bug.

### 1. Camera frame space (0.0 to 1.0)
MediaPipe returns landmark coordinates normalised to the frame:
- `(0.0, 0.0)` = top-left corner of the camera frame
- `(1.0, 1.0)` = bottom-right corner of the camera frame

This is what `FaceResult.left_iris` and `FaceResult.right_iris` contain.

### 2. Screen (session widget) space (0.0 to 1.0)
The gaze mapper outputs coordinates normalised to the session window:
- `(0.0, 0.0)` = top-left corner of the session screen widget
- `(1.0, 1.0)` = bottom-right corner

The calibration dots are also shown in this space.  The AOI polygon is defined
and stored in this space.

**This is the same space used for:** gaze predictions, AOI hit testing, and
`samples.csv` gaze columns.

### 3. Widget pixel space
When drawing on screen (e.g., placing the gaze dot), we need pixel coordinates:
```python
gaze_px_x = gaze_x_norm * widget_width
gaze_px_y = gaze_y_norm * widget_height
```

The conversion is done in `session_screen.py` just before drawing:
```python
result.gaze_px_x = result.sample.gaze_x_norm * self._canvas.width()
result.gaze_px_y = result.sample.gaze_y_norm * self._canvas.height()
```

**Key rule:** The calibration window and session window must be the same size
(or at least the same aspect ratio) for the gaze mapping to be accurate.  If
you resize the window between calibration and session, the mapping will be off.

---

## 15. Tests — what they check and why

Tests live in the `tests/` folder and are run with:
```bash
.venv/Scripts/pytest tests/ -v
```

### `tests/test_state_machine.py` — 5 tests

Tests the debounce logic independently of any camera or UI.

| Test | What it proves |
|---|---|
| `test_initial_state` | Machine starts in UNKNOWN |
| `test_stable_transition_commits` | After `stable_ms`, the new state is committed |
| `test_flicker_does_not_commit` | Rapid alternating states never commit |
| `test_transition_callback_fires` | The callback is called exactly once on commit |
| `test_force_end_segment` | Closing a segment produces the right duration |

### `tests/test_aoi.py` — 5 tests

Tests the polygon hit-test function.

| Test | What it proves |
|---|---|
| `test_centre_inside` | A point clearly inside returns True |
| `test_corner_outside` | A point outside returns False |
| `test_edge_considered_inside` | Edge points count as inside |
| `test_clearly_outside` | Far corner returns False |
| `test_degenerate_polygon_returns_false` | Polygons with < 3 points are safe |

### `tests/test_metrics.py` — 3 tests

Tests the debrief statistics computation.

| Test | What it proves |
|---|---|
| `test_empty_session` | Zero-data session doesn't crash |
| `test_all_in_cockpit` | Correct in-cockpit time calculation |
| `test_out_glance_counted` | One OUT segment is correctly counted and timed |

### Why write tests?

1. **Confidence:** When you change code later, tests tell you immediately if
   you broke something
2. **Documentation:** Tests show exactly how a function is supposed to behave
3. **Edge cases:** Tests force you to think about what happens with empty data,
   zero durations, and degenerate inputs

### The floating-point lesson (real bug we fixed)

The original test did:
```python
sm.update(GazeState.IN_COCKPIT, t + 0.15)  # 150ms > 100ms, should commit
```

But `(0.15 - 0.05) * 1000` in Python floating point is `99.9999…`, not exactly
`100.0`.  So `99.9999 >= 100.0` was `False` and the test failed.

**Fix:** Use `t + 0.20` (200ms window, clearly over the 100ms threshold).

This is why you should always test **boundary conditions** with values that
have clear margins, not values right at the boundary.

---

## 16. Virtual environment and packaging

### What is a virtual environment?

Python packages installed with `pip install` go into a global folder on your
computer.  If two projects need different versions of the same library, they
conflict.

A **virtual environment** (venv) creates an isolated copy of Python + packages
just for your project:

```
.venv/
  Scripts/
    python.exe    ← Python interpreter for this project only
    pip.exe       ← pip for this project only
    pytest.exe    ← test runner for this project only
  Lib/
    site-packages/
      mediapipe/  ← installed only for this project
      PySide6/
      ...
```

### Creating and using the venv

```bash
# Create (done once):
python -m venv .venv

# Activate (done every time you open a new terminal):
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install the project + all dependencies:
pip install -e ".[dev]"
```

The `-e` flag means **editable install** — changes to the source files take
effect immediately without reinstalling.

### `pyproject.toml` — the project manifest

This file defines everything about the project:

```toml
[project]
name = "eye-tracking-aviation"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.10",
    "mediapipe>=0.10.30",
    "PySide6>=6.7",
    ...
]

[project.optional-dependencies]
dev = [
    "ruff>=0.5",      # linter
    "black>=24",      # formatter
    "mypy>=1.10",     # type checker
    "pytest>=8",      # test runner
]
```

`pip install -e ".[dev]"` installs both the main `dependencies` and the `dev`
optional extras.

### Real-world gotcha: Windows DLL conflicts with PySide6

When this project was first set up, running `python app/main.py` raised:
```
ImportError: DLL load failed while importing QtWidgets:
The specified procedure could not be found.
```

**What happened:**

1. The system Python at `C:\...\Python313\python` is the **bare CPython installer
   from python.org**, which has no Qt installed globally.
2. The `.venv` was created from that Python and `pip` installed **PySide6 6.10.2**
   (the latest at the time).
3. PySide6 6.10.2 ships with Qt 6.10, which requires a newer Windows API than
   what was available on this machine — specifically a procedure that is
   "exported" by a system DLL but not present in the installed version.

**The fix:**

Re-create the venv using **Anaconda Python 3.13.9** (which ships with C++
runtimes and DLL compatibility layers) and pin `PySide6 < 6.10` in
`pyproject.toml`, which installs the stable **6.9.3** instead.

```bash
# Remove the broken venv, recreate from Anaconda:
rm -rf .venv
/c/Users/<you>/anaconda3/python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
# → installs PySide6 6.9.3  ✓
```

**Lessons learned:**

- The Python *interpreter* version (`3.13.7` vs `3.13.9`) matters less than
  the *environment* (bare CPython vs Anaconda) when dealing with C extension
  DLLs on Windows.
- Library versions on PyPI are not always Windows-compatible on all builds.
  When a GUI library crashes with a DLL error, **downgrading one minor version**
  is often the fastest fix.
- Always test `from PySide6.QtWidgets import QApplication` (not just
  `import PySide6`) when verifying the installation, because the heavy DLL
  loading only happens when you actually import a Qt submodule.

### Linting and formatting tools

| Tool | Purpose |
|---|---|
| `ruff` | Fast linter — checks for common mistakes, unused imports, style issues |
| `black` | Auto-formatter — rewrites your code to a consistent style |
| `mypy` | Type checker — verifies that function signatures match their calls |

Run them:
```bash
.venv/Scripts/ruff check .       # show linting errors
.venv/Scripts/black .            # auto-format all files
.venv/Scripts/mypy app/ domain/  # type check
```

---

## 17. Git and GitHub setup

### What is Git?

Git is a **version control system** — it tracks every change ever made to your
code.  Think of it as a "save history" for your entire project.

Key concepts:
- **Repository (repo):** The project folder tracked by Git
- **Commit:** A saved snapshot of all files at a point in time
- **Branch:** A parallel line of development (we only use `main` here)
- **Remote:** A copy of the repo on a server (e.g., GitHub)

### What is GitHub?

GitHub is a website that hosts Git repositories online.  It adds:
- Backup (your code is safe even if your computer breaks)
- Collaboration (multiple people can work on the same project)
- Issues and pull requests for managing work

### The `.gitignore` file

Not everything should be committed.  We exclude:

| Pattern | Why |
|---|---|
| `.venv/` | Large, regenerable from pyproject.toml |
| `__pycache__/` | Auto-generated bytecode, not source code |
| `runs/` | Session data is user-generated, often large |
| `profiles/` | User-specific calibration, not portable |
| `config.json` | User-specific settings |
| `assets/cockpit.jpg` | Large image file; user provides their own |
| `.claude/` | Claude Code internal files |

The `.gitkeep` files in `runs/` and `profiles/` are empty files that force
Git to track the empty directory structure (Git normally ignores empty folders).

### Uploading to GitHub

```bash
# 1. Create a new repo on github.com (don't initialise with README)
# 2. Connect your local repo to it:
git remote add origin https://github.com/YOUR_USERNAME/eye-tracking-aviation.git
git branch -M main
git push -u origin main
```

### The commit workflow going forward

```bash
git add <changed files>         # stage files
git status                      # review what will be committed
git commit -m "brief message"   # create snapshot
git push                        # upload to GitHub
```

---

## 18. Key patterns you'll see everywhere

### Dataclasses
```python
@dataclass
class GazeSample:
    timestamp_mono: float
    gaze_x_norm: float
    state: GazeState
```
Automatically gives you `__init__`, `__repr__`, and comparison operators.
Use them whenever you need a simple data container.

### Type hints
```python
def compute_debrief(
    samples: list[GazeSample],
    events: list[StateEvent],
    session_duration_s: float,
) -> dict[str, Any]:
```
Types are not enforced at runtime but:
- Make the code self-documenting
- Allow IDEs to autocomplete and catch bugs before you run the code
- Allow `mypy` to find type errors statically

### `Optional[X]` means "either X or None"
```python
self._frame: Optional[np.ndarray] = None
```
This says "this attribute can be either a numpy array or None".  You must check
`if self._frame is not None:` before using it.  This is safer than just writing
`None` and forgetting to check.

### Signals and slots (Qt)
```python
# Define a signal in a QObject subclass:
calibration_saved = Signal()

# Connect it to a handler:
wizard.calibration_saved.connect(self._on_calibration_saved)

# Emit it:
self.calibration_saved.emit()
```
Signals decouple components.  The `CalibrationWizard` doesn't need to know
anything about `MainWindow` — it just emits `calibration_saved` and whoever
is listening handles it.

### Context managers (`with` statement)
```python
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh)
# File is automatically closed here, even if an exception occurs
```
Always use `with` when opening files.  It guarantees the file is closed even
if an exception is raised inside the block.

### Logging instead of print()
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Session started: %s", session_id)
logger.debug("Gaze mapper fitted.  RMS=%.4f", rms)
logger.warning("No calibration found for profile '%s'", name)
```
Unlike `print()`, logging:
- Includes timestamps and log levels automatically
- Can be turned off for the entire application with one line
- Can be directed to a file instead of the console
- Uses `%s` formatting (not f-strings) for performance — the formatting only
  happens if the message is actually going to be displayed

---

## 19. How data flows from camera to debrief

Here is a complete walkthrough of one frame being processed during a session:

**1. Camera thread** (running continuously in background)
```
CaptureLoop: cap.read() → frame_bgr (640×480 numpy array)
             Stores it with a timestamp under a threading.Lock
```

**2. Vision worker thread** (running during session)
```
_worker_loop():
  camera.get_frame()         → (frame_bgr, mono_ts)
  cv2.cvtColor(BGR → RGB)    → frame_rgb
  face_tracker.process()     → FaceResult (iris positions, confidence)
  extract_features(face)     → features (20-dim numpy array)
  gaze_mapper.predict(feat)  → (0.61, 0.48) raw gaze point
  ema.update(0.61, 0.48)     → (0.59, 0.49) smoothed gaze point

  point_in_polygon(0.59, 0.49, aoi) → True
  raw_state = GazeState.IN_COCKPIT

  state_machine.update(IN_COCKPIT, 12.450) → GazeState.IN_COCKPIT (committed)

  GazeSample(
    timestamp_mono=12.450,
    timestamp_wall=1740404330.2,
    gaze_x_norm=0.59,
    gaze_y_norm=0.49,
    confidence=0.87,
    state=GazeState.IN_COCKPIT
  )

  session_writer.write_sample(sample)  → appends to samples.csv
  result_queue.put_nowait(VisionResult(...))
```

**3. Main thread** (QTimer fires every 16ms)
```
session_screen._poll_results():
  result = controller.poll_result()
  result.gaze_px_x = 0.59 * canvas_width   (= 754px for 1280px canvas)
  result.gaze_px_y = 0.49 * canvas_height  (= 372px for 760px canvas)

  overlay.set_durations(in_s=45.2, out_s=3.1, unk_s=1.7)
  overlay.update_result(result)  → triggers repaint
```

**4. Qt paint event**
```
DebugOverlay.paintEvent():
  Draw gaze dot at (754, 372) with green colour (IN_COCKPIT)
  Draw metrics panel top-right: "State: IN_COCKPIT, Conf: 0.87, ..."
```

**5. At session end**
```
controller.stop_session():
  state_machine.force_end_segment(end_mono)  → closes last segment
  compute_debrief(samples, events, duration) → statistics dict
  session_writer.write_debrief(debrief)      → writes debrief.json
  session_writer.close()                     → flushes + closes all CSVs
  return debrief
```

**6. Debrief screen built**
```
DebriefScreen(debrief, session_dir, cockpit_pixmap):
  Shows stat cards from debrief dict
  Builds matplotlib charts from timeline + out_durations_ms
  Loads samples.csv for replay scrubber
```

---

## 20. Known limitations and how to improve them

### Gaze accuracy

**Current approach:** Polynomial ridge regression on iris positions.

**Problem:** The model is trained on 9 screen positions.  Head movement between
calibration and session degrades accuracy.

**Better approach (advanced):** Use **3-D head pose estimation** (solve the
PnP problem using the 3-D MediaPipe landmarks) to decompose the eye rotation
into a 3-D gaze vector.  Libraries like `OpenGaze` or neural models like
`L2CS-Net` can estimate this much more accurately.

### Confidence estimation

**Current approach:** Eye openness ratio as a proxy for confidence.

**Problem:** The ratio is also low during normal squinting (bright sunlight,
concentration), not just blinks or face-loss.

**Better approach:** Use the MediaPipe detection confidence score directly, or
look at the variance of iris positions across several consecutive frames (stable
= high confidence, jittery = low confidence).

### Window resize breaks calibration

**Current problem:** Gaze coordinates are normalised to the session widget's
pixel dimensions at calibration time.  If the window is resized, the coordinate
space changes.

**Fix:** Lock the window to a fixed size (`setFixedSize(width, height)`), or
re-normalise gaze predictions to the *current* window size using the recorded
calibration window dimensions.

### Single monitor only

**Problem:** If the application is dragged to a second monitor with a different
position/scale, gaze mapping breaks.

**Fix:** Store the monitor configuration at calibration time and use Qt's
`QScreen` API to remap coordinates at session time.

### No PDF export

The debrief screen currently exports CSV only.

**Fix:** Add `reportlab` or `fpdf2` to dependencies and generate a formatted
PDF with embedded charts.

---

## Quick reference: running the project

```bash
# Activate your environment:
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Mac/Linux

# Run the application:
python app/main.py

# Run all tests:
pytest tests/ -v

# Check code style:
ruff check .

# Auto-format code:
black .

# Generate a placeholder cockpit image (if you don't have a real one):
python assets/generate_placeholder.py
```

---

*This file was written to be a complete learning resource.  If something is
unclear, look at the source file mentioned in each section and trace through
the code line by line — the best way to understand code is to read it slowly
and ask "what does this line do?".*
