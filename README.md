# SevenSegmentMonitor

SevenSegmentMonitor is a computer vision and IoT monitoring tool designed to continuously capture images (from a USB/V4L2 camera
or static frames) of seven-segment displays, extract and recognize numeric readouts via OpenCV, and persist the parsed readings
into an SQLite database with automated retention/cleanup.

---

## Features

- **Multi-threaded Architecture:**
    - **Camera Producer:** Asynchronously captures frames from a V4L2 / USB camera or loads frames from disk for
      testing/simulation.
    - **Vision Worker:** Performs automatic rotation correction, color/brightness filtering, shear de-skewing, digit box slicing,
      and segment state recognition.
    - **Database Writer Worker:** Batches writes and persists timestamped readings and alert/state statuses to SQLite.
    - **Database Cleanup Worker:** Automatically purges records older than a configurable retention period.
- **Multiple Color & Mask Extraction Algorithms:**
    - `WITHOUT_GREEN`: Difference-based channel masking for red LED displays.
    - `RED`: Dual-band HSV color masking.
    - `RED_ADAPTIVE`: Adaptive thresholding based on the green-to-red pixel ratio.
- **Display Layout Customization:**
    - Configurable rows and digits per row (`NUM_DIGITS_ROWS`, `NUM_DIGITS_PER_ROW`).
    - Shear angle compensation for slanted italicized digits (`DIGIT_SHEAR_ANGLE`).
    - Digit density and segment activation threshold tuning.
- **Deployment Ready:**
    - Run directly via Python CLI (`thermometermonitor`).
    - Run containerized via Docker Compose with optional Grafana integration (using `frser-sqlite-datasource`).

---

## Architecture Overview

```
                      +-------------------+
                      |   Camera / Video  |
                      |   or Static Img   |
                      +---------+---------+
                                |
                                v
                    [ camera_producer_thread ]
                                |
                          (frame_queue)
                                |
                                v
                     [ vision_worker_thread ]
                 - Rotation / Perspective Deskew
                 - Masking (WITHOUT_GREEN / RED / RED_ADAPTIVE)
                 - Bounding box & Digit slicing
                 - 7-Segment state evaluation
                                |
                           (db_queue)
                                |
                                v
                   [ db_writer_worker_thread ]
                                |
                                v
                     [ SQLite Database DB ]
                                ^
                                |
                  [ db_cleanup_worker_thread ]
```

---

## Installation

### Prerequisites

- Python >= 3.13
- UV package manager (recommended) or pip

### Local Setup with UV

```bash
# Clone repository
git clone https://github.com/lexsysko/SevenSegmentMonitor.git
cd SevenSegmentMonitor

# Install dependencies and package in editable mode
uv sync
```

---

## Usage

### 1. Running via CLI

Once installed, you can start the application using the entry point script:

```bash
# Set your environment variables or copy .env
cp dot.env.example .env

# Run monitor
uv run thermometermonitor
# or
uv run python -m SevenSegmentMonitor.main
```

### 2. Running via Docker Compose

```bash
# Start monitor service
docker compose up -d seven-s-monitor

# Start monitor with Grafana dashboard
docker compose --profile grafana up -d
```

---

## Configuration & Environment Variables (`settings.py`)

All configuration options can be defined via environment variables or loaded automatically from a `.env` file located in the
working directory or project root.

### Core & Storage

| Environment Variable | Type             | Default                             | Description                                                                                        |
|----------------------|------------------|-------------------------------------|----------------------------------------------------------------------------------------------------|
| `CAMERA_PATH`        | `str` / `int`    | `/dev/video0`                       | Camera device path (e.g., `/dev/video0`) or video capture device index (`0`, `1`).                 |
| `DATA_PATH`          | `str`            | `./data` (or `<project_root>/data`) | Directory where SQLite database, captured frames (`img/`), and debug images (`debug/`) are stored. |
| `DB_FILE`            | `str`            | `seventsegment_data.db`             | Filename of the SQLite database inside `DATA_PATH`.                                                |
| `LOGLEVEL`           | `str`            | `INFO`                              | Application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).                           |
| `FORCE_HEADLESS`     | `bool` (`t`/`f`) | `False`                             | Forces headless mode (disables OpenCV GUI debug display windows).                                  |

### Database Maintenance & Batching

| Environment Variable     | Type  | Default       | Description                                                                             |
|--------------------------|-------|---------------|-----------------------------------------------------------------------------------------|
| `BATCH_FLUSH_DB_TIMEOUT` | `int` | `2`           | Interval in seconds between batch commits to the SQLite database.                       |
| `CLEANUP_TIMEOUT`        | `int` | `86400` (24h) | Interval in seconds between executions of the historical database cleanup worker.       |
| `CLEANUP_PERIOD_DAYS`    | `int` | `30`          | Number of days of historical records to keep. Older records are deleted during cleanup. |

### Frame Capture & Simulation

| Environment Variable | Type             | Default      | Description                                                                                                                  |
|----------------------|------------------|--------------|------------------------------------------------------------------------------------------------------------------------------|
| `FRAME_FPS_DELAY`    | `float`          | `1.0`        | Delay in seconds between consecutive frame captures.                                                                         |
| `LOAD_FRAME`         | `bool` (`t`/`f`) | `False`      | When `True`, frames are loaded cyclically from files on disk instead of the camera (useful for testing/tuning).              |
| `SAVE_FRAME`         | `bool` (`t`/`f`) | `False`      | When `True` and `LOAD_FRAME` is `False`, captured frames are saved to disk in `DATA_PATH/img`.                               |
| `FRAME_FILE_NAME`    | `str`            | `frame*.png` | Pattern or filename used for saving/loading frames in `DATA_PATH/img`. If it contains `*`, timestamping or globbing is used. |

### Vision & Preprocessing

| Environment Variable       | Type              | Default         | Description                                                                                                               |
|----------------------------|-------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| `ROTATE_FIXED_FRAME_ANGLE` | `float` \| `None` | `None`          | Fixed rotation angle (in degrees) applied to the raw frame immediately upon capture.                                      |
| `ROTATE_FRAME_ANGLE`       | `float` \| `None` | `None`          | Rotation angle override for segment alignment. If set to `None`, auto-rotation is calculated using contour min-area rect. |
| `DIMMED_BRIGHTNESS`        | `float` \| `None` | `None`          | Fixed brightness threshold value to detect whether the display is dimmed.                                                 |
| `VISION_MASK_TYPE`         | `str`             | `without_green` | Algorithm used to isolate segments. Supported values: `without_green`, `red`, `red_adaptive`.                             |
| `SMALL_COMPONENT_AREA`     | `int` \| `None`   | `None`          | Minimum area threshold for connected components filtering to remove small noise artifacts.                                |

### Color Thresholds & Ranges

| Environment Variable   | Type                   | Default           | Description                                                                                                        |
|------------------------|------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------|
| `WITHOUT_GREEN_RANGE`  | `tuple[int, int]`      | `(220, 225)`      | Low and high thresholds `(low, high)` applied to red minus green difference when `VISION_MASK_TYPE=without_green`. |
| `RED_HSV_RANGE_1_LOW`  | `tuple[int, int, int]` | `(0, 40, 180)`    | Lower HSV boundary for red color band 1 when `VISION_MASK_TYPE=red`.                                               |
| `RED_HSV_RANGE_1_HIGH` | `tuple[int, int, int]` | `(18, 255, 255)`  | Upper HSV boundary for red color band 1 when `VISION_MASK_TYPE=red`.                                               |
| `RED_HSV_RANGE_2_LOW`  | `tuple[int, int, int]` | `(160, 40, 180)`  | Lower HSV boundary for red color band 2 when `VISION_MASK_TYPE=red`.                                               |
| `RED_HSV_RANGE_2_HIGH` | `tuple[int, int, int]` | `(180, 255, 255)` | Upper HSV boundary for red color band 2 when `VISION_MASK_TYPE=red`.                                               |
| `RED_ADAPTIVE_RANGE`   | `tuple[int, int]`      | `(30, 255)`       | Lower and upper thresholds for adaptive ratio mask when `VISION_MASK_TYPE=red_adaptive`.                           |

### Digit & Segment Recognition

| Environment Variable       | Type             | Default | Description                                                                                             |
|----------------------------|------------------|---------|---------------------------------------------------------------------------------------------------------|
| `NUM_DIGITS_ROWS`          | `int`            | `2`     | Number of display digit rows.                                                                           |
| `NUM_DIGITS_PER_ROW`       | `int`            | `3`     | Number of digits per display row.                                                                       |
| `DIGIT_SHEAR_ANGLE`        | `float`          | `-7.0`  | Shear transformation angle in degrees used to deskew italicized/slanted seven-segment digits.           |
| `DIGIT_DENSITY_THRESH`     | `float`          | `0.15`  | Minimum density threshold (ratio of active pixels) required for a slice to be treated as a valid digit. |
| `THRESHOLD_ON_STATE`       | `int`            | `3`     | Number of active segments required to mark the device/display state as active (`True`).                 |
| `NORMALIZE_DIGITS_HEIGHT`  | `bool` (`t`/`f`) | `False` | Whether to resize/normalize digit bounding box heights prior to segment extraction.                     |
| `NORMALIZED_DIGITS_HEIGHT` | `int`            | `64`    | Target height in pixels when `NORMALIZE_DIGITS_HEIGHT` is enabled.                                      |

---

## Tuning & Calibration

For step-by-step guidance on calibrating shear angles, thresholding, and color masks, refer to:

- [`docs/TUNING.md`](docs/TUNING.md) — Parameter tuning guide.
- [`docs/LOGS.md`](docs/LOGS.md) — Example log output during execution.
- `tools/easy_hsv_color_picker/` — Interactive GUI tool to find optimal HSV ranges for your display.

---

## License

This project is licensed under the [MIT License](LICENSE).
