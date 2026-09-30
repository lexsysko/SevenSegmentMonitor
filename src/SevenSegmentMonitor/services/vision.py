import cv2
import numpy as np
import logging

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
    (0.22, 0.02, 0.56, 0.15),  # a
    (0.68, 0.12, 0.28, 0.32),  # b
    (0.68, 0.52, 0.28, 0.32),  # c
    (0.22, 0.83, 0.56, 0.15),  # d
    (0.04, 0.52, 0.28, 0.32),  # e
    (0.04, 0.12, 0.28, 0.32),  # f
    (0.22, 0.42, 0.56, 0.15),  # g
]


def rotate_frame(frame, angle=5.0):
    (h, w) = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(mask, shear_angle=9.0):
    """Constant horizontal shear to un-italicise the digits."""
    h, w = mask.shape[:2]
    M = np.float32([
        [1, np.tan(np.radians(-shear_angle)), 0],
        [0, 1, 0]
    ])
    M[0, 2] = -M[0, 1] * h / 2
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)


def make_red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 40, 180), (18, 255, 255))
    m2 = cv2.inRange(hsv, (160, 40, 180), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def is_probably_one(roi: np.ndarray) -> bool:
    """Thin digit with almost all mass on the right side → '1'."""
    h, w = roi.shape[:2]
    if h < 10 or w < 4:
        return False
    if h / max(w, 1) > 2.3:
        return True
    mid = w // 2
    left_d = cv2.countNonZero(roi[:, :mid]) / max(mid * h, 1)
    right_d = cv2.countNonZero(roi[:, mid:]) / max((w - mid) * h, 1)
    return right_d > 0.22 and left_d < 0.07


def decode_digit(roi: np.ndarray, density_thresh: float = 0.18) -> str:
    h, w = roi.shape[:2]
    if h < 10 or w < 6:
        return "?"

    # ----- ignore decimal point area (right ~12 % of the slot) -----
    usable_w = int(w * 0.88)
    roi = roi[:, :usable_w]
    h, w = roi.shape[:2]

    # Optional upscale for very small digits
    if h < 22:
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        h, w = roi.shape[:2]

    if is_probably_one(roi):
        return "1"

    states = []
    for rx, ry, rw, rh in SEGMENTS_REL:
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = min(w, x1 + max(1, int(rw * w)))
        y2 = min(h, y1 + max(1, int(rh * h)))
        seg = roi[y1:y2, x1:x2]
        dens = cv2.countNonZero(seg) / float(seg.size) if seg.size else 0.0
        states.append(1 if dens > density_thresh else 0)

    return DIGIT_MAP.get(tuple(states), "?")


def find_display_roi(mask: np.ndarray, min_area=800):
    """Tight bounding box of the largest red blob (whole display)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(mask.shape[1] - x, w + 2 * pad)
    h = min(mask.shape[0] - y, h + 2 * pad)
    return x, y, w, h


def extract_digits_fixed_pitch(mask, display_roi, num_digits=6,
                               char_width_ratio=0.130,
                               gap_ratio=0.030):
    """
    Split the display into equal-width character slots.
    Decimal points fall into the gaps or are cut by the 0.88 crop inside decode_digit.
    """
    x, y, w, h = display_roi
    char_w = int(w * char_width_ratio)
    gap = int(w * gap_ratio)
    total_w = num_digits * char_w + (num_digits - 1) * gap
    start_x = x + max(0, (w - total_w) // 2)

    rois = []
    boxes = []  # for drawing
    for i in range(num_digits):
        dx = start_x + i * (char_w + gap)
        roi = mask[y:y + h, dx:dx + char_w]
        rois.append(roi)
        boxes.append((dx, y, char_w, h))
    return rois, boxes


# ====================== public API ======================

def process_and_annotate(frame):
    """Full pipeline with visualisation (GUI path)."""
    frame = rotate_frame(frame, angle=5.0)
    dimmed = cv2.convertScaleAbs(frame, alpha=1.0, beta=-20)
    mask = make_red_mask(dimmed)
    mask = deskew(mask, shear_angle=9.0)

    display_roi = find_display_roi(mask)
    readout = ""
    digit_boxes = []

    if display_roi is not None:
        NUM_DIGITS = 6
        rois, boxes = extract_digits_fixed_pitch(
            mask, display_roi,
            num_digits=NUM_DIGITS,
            char_width_ratio=0.130,
            gap_ratio=0.030
        )

        chars = []
        for i, roi in enumerate(rois):
            ch = decode_digit(roi, density_thresh=0.18)
            chars.append(ch)
            x, y, w, h = boxes[i]
            digit_boxes.append((x, y, w, h, ch))

        readout = "".join(chars)

    # ----- draw -----
    vis = frame.copy()
    if display_roi is not None:
        x, y, w, h = display_roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h, ch) in digit_boxes:
        color = (0, 255, 0) if ch != "?" else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, ch, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(vis, f"Readout: {readout or '…'}",
                (20, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return readout, (vis, mask)


def process_frame(frame):
    """Headless version – same logic, no drawing."""
    frame = rotate_frame(frame, angle=5.0)
    dimmed = cv2.convertScaleAbs(frame, alpha=1.0, beta=-20)
    mask = make_red_mask(dimmed)
    mask = deskew(mask, shear_angle=9.0)

    display_roi = find_display_roi(mask)
    if display_roi is None:
        return ""

    rois, _ = extract_digits_fixed_pitch(
        mask, display_roi,
        num_digits=6,
        char_width_ratio=0.130,
        gap_ratio=0.030
    )
    return "".join(decode_digit(r, density_thresh=0.18) for r in rois)
