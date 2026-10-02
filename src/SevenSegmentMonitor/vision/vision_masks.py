import cv2
import numpy as np
from SevenSegmentMonitor.enums import MaskType
from SevenSegmentMonitor.settings import (
    RED_HSV_RANGE_1_LOW,
    RED_HSV_RANGE_1_HIGH,
    RED_HSV_RANGE_2_LOW,
    RED_HSV_RANGE_2_HIGH,
    WITHOUT_GREEN_RANGE,
    VISION_MASK_TYPE,
    RED_ADAPTIVE_RANGE,
)


def make_binary_otsu_mask(frame: np.ndarray, range: tuple[int, int] = (0, 255)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, range[0], range[1], cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def make_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED_HSV_RANGE_1_LOW, RED_HSV_RANGE_1_HIGH)
    m2 = cv2.inRange(hsv, RED_HSV_RANGE_2_LOW, RED_HSV_RANGE_2_HIGH)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def make_red_mask_adaptive(frame: np.ndarray) -> np.ndarray:
    # Convert frame to signed integer to prevent negative underflow
    b, g, r = cv2.split(frame.astype(np.int16))

    # Subtract max(G, B) from R. Red pixels will produce high positive values.
    # Non-red bright pixels (white lights, green LEDs) will produce <= 0.
    red_dominance = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)

    # Threshold the dominant red pixels (adjust '30' if needed)
    _, mask = cv2.threshold(red_dominance, RED_ADAPTIVE_RANGE[0], RED_ADAPTIVE_RANGE[1], cv2.THRESH_BINARY)

    # Close small gaps in segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def make_without_green_mask(frame: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(frame)
    gray_no_green = ((r.astype(np.uint16) + b.astype(np.uint16)) // 2).astype(np.uint8)
    _, mask_bright = cv2.threshold(gray_no_green, WITHOUT_GREEN_RANGE[0], WITHOUT_GREEN_RANGE[1], cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask_bright, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def make_mask(frame: np.ndarray, mask_type: MaskType | None = None) -> np.ndarray:
    mask_type = mask_type or VISION_MASK_TYPE
    match mask_type:
        case MaskType.WITHOUT_GREEN:
            return make_without_green_mask(frame)
        case MaskType.RED:
            return make_red_mask(frame)
        case MaskType.RED_ADAPTIVE:
            return make_red_mask_adaptive(frame)
    raise NotImplemented
