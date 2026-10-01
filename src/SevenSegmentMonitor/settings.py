from os import environ, mkdir
from pathlib import Path

from dotenv import load_dotenv

from SevenSegmentMonitor import __version__

BASE_PATH = Path(__file__).parent.parent.parent

for folder in (Path.cwd(), BASE_PATH):
    if (folder / ".env").exists():
        load_dotenv()
        break

data_path_str: str | None = environ.get("DATA_PATH")

if data_path_str:
    DATA_PATH = Path(data_path_str).expanduser().resolve()
else:
    DATA_PATH = BASE_PATH / "data"
    if not DATA_PATH.exists():
        # Safe for both CLI package execution and local development
        DATA_PATH = Path.cwd() / "data"

DB_FILE = DATA_PATH / Path(environ.get("DB_FILE", "seventsegment_data.db")).name
DB_FILE.parent.mkdir(exist_ok=True, parents=True)

APP_VERSION = __version__
LOGLEVEL: str = environ.get("LOGLEVEL", "INFO").upper()
CLEANUP_TIMEOUT: int = int(environ.get("CLEANUP_TIMEOUT", 60 * 60 * 24))
CLEANUP_PERIOD_DAYS: int = int(environ.get("CLEANUP_PERIOD_DAYS", 30))
BATCH_FLUSH_DB_TIMEOUT: int = int(environ.get("BATCH_FLUSH_DB_TIMEOUT", 2))

LOAD_FRAME: bool = environ.get("LOAD_FRAME", "f")[0].lower() == "t"
SAVE_FRAME: bool = environ.get("SAVE_FRAME", "f")[0].lower() == "t"
FRAME_FOLDERS: Path = DATA_PATH / "img"
FRAME_FOLDERS.mkdir(exist_ok=True, parents=True)
FRAME_NAME_PATH: Path = FRAME_FOLDERS / Path(environ.get("FRAME_FILE_NAME", "frame*.png")).name

FRAME_FPS_DELAY: float = float(environ.get("FRAME_FPS_DELAY", 1))

# if ROTATE_FRAME_ANGLE is None then use Autorotate
ROTATE_FRAME_ANGLE: float | None = float(_a) if (_a := environ.get("ROTATE_FRAME_ANGLE")) else None
DIMMED_BRIGHTNESS: float = float(environ.get("DIMMED_BRIGHTNESS", -10))

DIGIT_DENSITY_THRESH: float = float(environ.get("DIGIT_DENSITY_THRESH", 0.15))
DIGIT_SHEAR_ANGLE: float = float(environ.get("DIGIT_SHEAR_ANGLE", -7))

NORMALIZE_DIGITS_HEIGHT: bool = environ.get("NORMALIZE_DIGITS_HEIGHT", "f")[0].lower() == "t"
NORMALIZED_DIGITS_HEIGHT: int = int(environ.get("NORMALIZED_DIGITS_HEIGHT", 64))

THRESHOLD_ON_STATE: int = int(environ.get("THRESHOLD_ON_STATE", 3))
