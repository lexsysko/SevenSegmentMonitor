import logging

import cv2
from SevenSegmentMonitor.services.vision import logger

_is_headless = None


def is_headless() -> bool:
    global _is_headless
    if _is_headless is None:
        if logger.getEffectiveLevel() != logging.DEBUG:
            _is_headless = False
            return False
        build_info = cv2.getBuildInformation()
        # Headless builds explicitly state 'GUI: NONE' under the GUI section
        _is_headless = "GUI:               NONE" in build_info or "GUI:" not in build_info
    return _is_headless
