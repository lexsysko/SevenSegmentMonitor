from os import environ
from pathlib import Path

from SevenSegmentMonitor.enums import MaskType
from dotenv import load_dotenv

from SevenSegmentMonitor import __version__


def str_tuple(a: str | None, default: tuple) -> tuple:
    if not a:
        return default
    try:
        return tuple([int(i.strip()) for i in a.strip().split(",")])
    except ValueError:
        return default


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

FORCE_HEADLESS: bool = environ.get("FORCE_HEADLESS", "f")[0].lower() == "t"
LOAD_FRAME: bool = environ.get("LOAD_FRAME", "f")[0].lower() == "t"
SAVE_FRAME: bool = environ.get("SAVE_FRAME", "f")[0].lower() == "t"
FRAME_FOLDERS: Path = DATA_PATH / "img"
FRAME_FOLDERS.mkdir(exist_ok=True, parents=True)
FRAME_NAME_PATH: Path = FRAME_FOLDERS / Path(environ.get("FRAME_FILE_NAME", "frame*.png")).name

DEBUG_FOLDER = DATA_PATH / "debug"
DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)


FRAME_FPS_DELAY: float = float(environ.get("FRAME_FPS_DELAY", 1))

ROTATE_FIXED_FRAME_ANGLE: float | None = float(_a) if (_a := environ.get("ROTATE_FIXED_FRAME_ANGLE")) else None
# if ROTATE_FRAME_ANGLE is None then use Autorotate
ROTATE_FRAME_ANGLE: float | None = float(_a) if (_a := environ.get("ROTATE_FRAME_ANGLE")) else None
DIMMED_BRIGHTNESS: float | None = float(_a) if (_a := environ.get("DIMMED_BRIGHTNESS")) else None

RED_HSV_RANGE_1_LOW: tuple = str_tuple(environ.get("RED_HSV_RANGE_1_LOW"), (0, 40, 180))
RED_HSV_RANGE_1_HIGH: tuple = str_tuple(environ.get("RED_HSV_RANGE_1_HIGH"), (18, 255, 255))
RED_HSV_RANGE_2_LOW: tuple = str_tuple(environ.get("RED_HSV_RANGE_2_LOW"), (160, 40, 180))
RED_HSV_RANGE_2_HIGH: tuple = str_tuple(environ.get("RED_HSV_RANGE_2_HIGH"), (180, 255, 255))

WITHOUT_GREEN_RANGE: tuple = str_tuple(environ.get("WITHOUT_GREEN_RANGE"), (220, 225))
RED_ADAPTIVE_RANGE: tuple = str_tuple(environ.get("RED_ADAPTIVE_RANGE"), (30, 255))

try:
    VISION_MASK_TYPE: MaskType = MaskType(environ.get("VISION_MASK_TYPE", MaskType.WITHOUT_GREEN.value).lower().strip())
except ValueError as e:
    print(f"VISION_MASK_TYPE value {e} of {(',').join(MaskType.__members__.keys())}")
    exit(1)

SMALL_COMPONENT_AREA: int | None = int(_a) if (_a := environ.get("SMALL_COMPONENT_AREA")) else None

DIGIT_DENSITY_THRESH: float = float(environ.get("DIGIT_DENSITY_THRESH", 0.15))
DIGIT_SHEAR_ANGLE: float = float(environ.get("DIGIT_SHEAR_ANGLE", -7))

NORMALIZE_DIGITS_HEIGHT: bool = environ.get("NORMALIZE_DIGITS_HEIGHT", "f")[0].lower() == "t"
NORMALIZED_DIGITS_HEIGHT: int = int(environ.get("NORMALIZED_DIGITS_HEIGHT", 64))

THRESHOLD_ON_STATE: int = int(environ.get("THRESHOLD_ON_STATE", 3))

NUM_DIGITS_ROWS: int = int(environ.get("NUM_DIGITS_ROWS", 2))
NUM_DIGITS_PER_ROW: int = int(environ.get("NUM_DIGITS_PER_ROW", 3))
