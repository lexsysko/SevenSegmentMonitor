import logging
import re

import cv2
from SevenSegmentMonitor.settings import FORCE_HEADLESS

logger = logging.getLogger(__name__)

_is_headless = None


def is_headless() -> bool:
    global _is_headless

    if _is_headless is None:
        if FORCE_HEADLESS:
            _is_headless = True
            return True

        build_info = cv2.getBuildInformation()

        match = re.search(r"^\s*GUI:\s*(.*)$", build_info, re.MULTILINE)

        if match:
            gui = match.group(1).strip()
            _is_headless = gui.upper() == "NONE"
        else:
            _is_headless = True

    return _is_headless


def is_debug():
    return logger.getEffectiveLevel() == logging.DEBUG
