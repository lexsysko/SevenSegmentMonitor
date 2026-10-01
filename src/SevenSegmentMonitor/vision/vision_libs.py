import cv2
import numpy as np
import logging

from SevenSegmentMonitor.settings import (
    ROTATE_FRAME_ANGLE,
    DIMMED_BRIGHTNESS,
    NORMALIZE_DIGITS_HEIGHT,
    NORMALIZED_DIGITS_HEIGHT,
    DIGIT_SHEAR_ANGLE,
    NUM_DIGITS_PER_ROW,
)

logger = logging.getLogger(__name__)

DIGIT_MAP = {
    (1, 1, 1, 1, 1, 1, 0): "0",
    (0, 1, 1, 0, 0, 0, 0): "1",
    (1, 1, 0, 1, 1, 0, 1): "2",
    (1, 1, 1, 1, 0, 0, 1): "3",
    (0, 1, 1, 0, 0, 1, 1): "4",
    (1, 0, 1, 1, 0, 1, 1): "5",
    (1, 0, 1, 1, 1, 1, 1): "6",
    (1, 1, 1, 0, 0, 0, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}

# Relative segment windows (x, y, w, h) in 0..1
# Order: a(top), b(top-right), c(bottom-right), d(bottom), e(bottom-left), f(top-left), g(middle)
SEGMENTS_REL = [
    (0.18, 0.02, 0.64, 0.15),  # a
    (0.62, 0.12, 0.34, 0.31),  # b
    (0.62, 0.53, 0.34, 0.31),  # c
    (0.18, 0.83, 0.64, 0.14),  # d
    (0.04, 0.53, 0.34, 0.31),  # e
    (0.04, 0.12, 0.34, 0.31),  # f
    (0.20, 0.42, 0.50, 0.15),  # g
]


def make_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def tilt_from_contour(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(make_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), ang = cv2.minAreaRect(c)
    # works for both old (-90..0] and new (0..90] OpenCV conventions
    return ((ang + 45) % 90) - 45


def rotate_frame(frame, angle: float | None = None) -> np.ndarray:
    """
    If angle is None used autorotate
    """
    if angle is None:
        angle = tilt_from_contour(frame)
        logger.debug(f"Auto rotated by {angle:.2f}deg")
    elif angle == 0:
        return frame
    (h, w) = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def deskew_centroid(mask: np.ndarray, shear_angle: float = 9.0) -> np.ndarray:
    """Constant horizontal shear to un-italicise the digits."""
    if not shear_angle:
        return mask
    h, w = mask.shape[:2]
    M = np.float32([[1, np.tan(np.radians(-shear_angle)), 0], [0, 1, 0]])
    M[0, 2] = -M[0, 1] * h / 2
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)


def deskew_anchor(mask: np.ndarray, shear_angle: float = 9.0, anchor: str = "bottom") -> np.ndarray:
    """Horizontal shear without translating the anchor row.
    anchor: 'top' | 'center' | 'bottom' -- the row that stays in place."""
    if not shear_angle:
        return mask
    h, w = mask.shape[:2]
    t = np.tan(np.radians(-shear_angle))
    y0 = {"top": 0, "center": h / 2, "bottom": h - 1}[anchor]
    # x' = x + t*(y - y0)  -> the row y0 does not move
    M = np.float32([[1, t, -t * y0], [0, 1, 0]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def make_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 40, 180), (18, 255, 255))
    m2 = cv2.inRange(hsv, (160, 40, 180), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def is_probably_one(roi: np.ndarray) -> bool:
    h, w = roi.shape[:2]
    if h < 12 or w < 5:
        return False
    if h / max(w, 1) < 2.6:
        return False
    mid = w // 2
    left_d = cv2.countNonZero(roi[:, :mid]) / max(mid * h, 1)
    right_d = cv2.countNonZero(roi[:, mid:]) / max((w - mid) * h, 1)
    return right_d > 0.28 and left_d < 0.06


def decode_digit(roi: np.ndarray, density_thresh: float = 0.13, id: int = 0, im_debug: bool = False) -> str:
    h, w = roi.shape[:2]
    if h < 10 or w < 6:
        return "?"

    if NORMALIZE_DIGITS_HEIGHT:
        # 1. normalize to a fixed height, keep the aspect ratio
        scale = NORMALIZED_DIGITS_HEIGHT / h
        roi = cv2.resize(roi, (max(1, round(w * scale)), NORMALIZED_DIGITS_HEIGHT), interpolation=cv2.INTER_LINEAR)
        _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)  # back to a clean mask
        h, w = roi.shape[:2]

    if DIGIT_SHEAR_ANGLE:
        roi = deskew_centroid(roi, shear_angle=DIGIT_SHEAR_ANGLE)

    # ignore decimal point area on the right
    usable_w = int(w * 0.90)
    roi = roi[:, :usable_w]
    h, w = roi.shape[:2]

    if h < 22:
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        h, w = roi.shape[:2]

    if is_probably_one(roi):
        return "1"

    debug: np.ndarray | None = None

    if im_debug:
        debug = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    states = []

    for i, (rx, ry, rw, rh) in enumerate(SEGMENTS_REL):
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = min(w, x1 + max(1, int(rw * w)))
        y2 = min(h, y1 + max(1, int(rh * h)))
        seg = roi[y1:y2, x1:x2]
        dens = cv2.countNonZero(seg) / float(seg.size) if seg.size else 0.0
        on = dens > density_thresh
        states.append(1 if on else 0)
        color = (0, 255, 0) if on else (0, 0, 255)
        if debug is not None:
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)

    if debug is not None:
        cv2.imshow(f"segments_{id}", debug)

    return DIGIT_MAP.get(tuple(states), "?")


def merge_vertical_fragments(
    candidates, x_tol: int = 10, y_gap_max: int = 15
) -> list[tuple[float, float, float, float]]:
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (c[0], c[1]))
    merged = []
    used = [False] * len(cands)

    for i, (x1, y1, w1, h1) in enumerate(cands):
        if used[i]:
            continue
        for j in range(i + 1, len(cands)):
            if used[j]:
                continue
            x2, y2, w2, h2 = cands[j]
            if abs(x1 - x2) < x_tol and 0 < (y2 - (y1 + h1)) < y_gap_max:
                nx = min(x1, x2)
                ny = min(y1, y2)
                nw = max(x1 + w1, x2 + w2) - nx
                nh = max(y1 + h1, y2 + h2) - ny
                merged.append((nx, ny, nw, nh))
                used[i] = used[j] = True
                break
        else:
            merged.append((x1, y1, w1, h1))
            used[i] = True
    return merged


def find_digit_candidates(mask: np.ndarray, min_area: int = 30, max_area: int = 800):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
    solid = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / max(w, 1)
        if 1.0 < aspect < 4.5:
            raw.append((x, y, w, h))

    return merge_vertical_fragments(raw, x_tol=10, y_gap_max=15)


def extract_row_from_candidates(mask, row_candidates, num_digits=3):
    """Extract one row using leftmost / rightmost + average width."""
    if len(row_candidates) < 2:
        return [], []

    row = sorted(row_candidates, key=lambda c: c[0])
    widths = [c[2] for c in row]
    avg_w = int(np.median(widths))

    x_start = row[0][0]
    x_end = row[-1][0] + row[-1][2]
    total_w = x_end - x_start
    y = min(c[1] for c in row)
    h = max(c[1] + c[3] for c in row) - y

    char_w = avg_w
    remaining = total_w - num_digits * char_w
    gap = max(0, remaining // (num_digits - 1)) if num_digits > 1 else 0

    if gap < 0 or gap > char_w * 0.6:
        char_w = total_w // num_digits
        gap = max(0, gap // 4)

    rois, boxes = [], []
    for i in range(num_digits):
        dx = max(0, x_start + i * (char_w + gap))
        roi = mask[y : y + h, dx : dx + char_w]
        rois.append(roi)
        boxes.append((dx, y, char_w, h))
    return rois, boxes


def extract_both_rows(mask, candidates, num_digits=3, row_y_tolerance=30):
    """
    Split candidates into top and bottom rows and extract both.
    Returns (rois, boxes) for all 6 digits (top then bottom).
    """
    if not candidates:
        return [], []

    # Sort by Y and split into two groups
    candidates = sorted(candidates, key=lambda c: c[1])
    ys = [c[1] for c in candidates]
    mid_y = (min(ys) + max(ys)) / 2

    #
    top_cands = [c for c in candidates if c[1] < mid_y + row_y_tolerance // 2]
    bottom_cands = [c for c in candidates if c[1] >= mid_y - row_y_tolerance // 2]

    # Avoid double-counting if tolerance is large
    # top_cands = [c for c in candidates if c[1] < mid_y]
    # bottom_cands = [c for c in candidates if c[1] >= mid_y]

    top_rois, top_boxes = extract_row_from_candidates(mask, top_cands, num_digits)
    bottom_rois, bottom_boxes = extract_row_from_candidates(mask, bottom_cands, num_digits)

    return top_rois + bottom_rois, top_boxes + bottom_boxes


def split_rows(candidates, num_rows: int | None = None, gap_factor: float = 0.6):
    """
    Group (x, y, w, h) candidates into rows, ordered top to bottom.
    num_rows: if known, split at the (num_rows - 1) largest gaps.
              If None, split wherever the gap exceeds gap_factor * median digit height.
    """
    if not candidates:
        return []

    cands = sorted(candidates, key=lambda c: c[1] + c[3] / 2)  # by Y center
    cys = np.array([c[1] + c[3] / 2 for c in cands])
    gaps = np.diff(cys)

    if num_rows is not None and num_rows > 1:
        k = min(num_rows - 1, len(gaps))
        cut_idx = np.sort(np.argsort(gaps)[-k:]) if k > 0 else []
    else:
        med_h = np.median([c[3] for c in cands])
        cut_idx = np.where(gaps > gap_factor * med_h)[0]

    rows, start = [], 0
    for i in cut_idx:
        rows.append(cands[start : i + 1])
        start = i + 1
    rows.append(cands[start:])
    return rows


def extract_all_rows(mask: np.ndarray, candidates, num_digits=3, num_rows=None, gap_factor=0.6):
    """Returns (rois, boxes) for every row, top to bottom, each row left to right."""
    all_rois, all_boxes = [], []
    for row in split_rows(candidates, num_rows, gap_factor):
        rois, boxes = extract_row_from_candidates(mask, row, num_digits)
        all_rois += rois
        all_boxes += boxes
    return all_rois, all_boxes


def preprocess_img(
    frame: np.ndarray,
    rotate_frame_angle: float | None = ROTATE_FRAME_ANGLE,
    dimmed_brightness: float = DIMMED_BRIGHTNESS,
) -> np.ndarray:
    if rotate_frame_angle != 0:
        frame = rotate_frame(frame, angle=rotate_frame_angle)
    if dimmed_brightness:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=dimmed_brightness)
    return frame
