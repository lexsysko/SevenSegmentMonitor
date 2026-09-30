from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

try:
    __version__ = version("SevenSegmentMonitor")
except PackageNotFoundError:
    try:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        if pyproject_path.is_file():
            with pyproject_path.open("rb") as f:
                __version__ = tomllib.load(f).get("project", {}).get("version", "unknown")
        else:
            __version__ = "unknown"
    except Exception:
        __version__ = "unknown"

__all__ = ["__version__"]
