import logging

import cv2
import numpy as np
from SevenSegmentMonitor.settings import DIGIT_DENSITY_THRESH
from SevenSegmentMonitor.vision.vision_libs import (
    find_digit_candidates,
    extract_both_rows,
    decode_digit,
    preprocess_img,
    make_red_mask,
)

logger = logging.getLogger(__name__)

# ====================== public API ======================


def process_and_annotate(frame: np.ndarray) -> tuple[str, tuple]:
    """Full pipeline with visualization (GUI path)."""
    frame = preprocess_img(frame)
    mask = make_red_mask(frame)

    candidates = find_digit_candidates(mask)
    rois, boxes = extract_both_rows(mask, candidates, num_digits=3)

    chars = []
    digit_boxes = []
    for i, roi in enumerate(rois):
        ch = decode_digit(roi, density_thresh=DIGIT_DENSITY_THRESH, id=i, im_debug=True)
        chars.append(ch)
        x, y, w, h = boxes[i]
        digit_boxes.append((x, y, w, h, ch))

    # Format as two groups
    if len(chars) == 6:
        readout = f"{''.join(chars[:3])} {''.join(chars[3:])}"
    else:
        readout = "".join(chars)

    # ----- draw -----
    vis = frame.copy()

    for x, y, w, h, ch in digit_boxes:
        color = (0, 255, 0) if ch != "?" else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, ch, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Optional: show candidates in yellow
    for c in candidates:
        x, y, w, h = c
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0, 0.1), 1)

    cv2.putText(
        vis, f"Readout: {readout or '…'}", (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )

    return readout, (vis, mask)


def process_frame(frame: np.ndarray) -> str:
    """Headless version – same logic, no drawing."""
    frame = preprocess_img(frame)
    mask = make_red_mask(frame)

    candidates = find_digit_candidates(mask)
    rois, _ = extract_both_rows(mask, candidates, num_digits=3)

    chars = [decode_digit(r, density_thresh=DIGIT_DENSITY_THRESH) for r in rois]

    if len(chars) == 6:
        return f"{''.join(chars[:3])} {''.join(chars[3:])}"
    return "".join(chars)
