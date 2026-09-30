from os import environ
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

