import logging

import cv2
import numpy as np
from SevenSegmentMonitor.settings import DIGIT_DENSITY_THRESH, NUM_DIGITS_PER_ROW, NUM_DIGITS_ROWS, SMALL_COMPONENT_AREA
from SevenSegmentMonitor.vision.vision_libs import (
    find_digit_candidates,
    decode_digit,
    preprocess_img,
    make_red_mask,
    extract_all_rows,
    make_debug_grid,
    remove_small_components,
    show_or_save,
    make_without_green_mask,
)

logger = logging.getLogger(__name__)

# ====================== public API ======================


def process_and_annotate(frame: np.ndarray) -> tuple[str, tuple]:
    """Full pipeline with visualization (GUI path)."""
    frame = preprocess_img(frame)
    # mask = make_red_mask(frame)
    mask = make_without_green_mask(frame)
    if SMALL_COMPONENT_AREA is not None:
        mask = remove_small_components(mask, min_area=SMALL_COMPONENT_AREA)

    candidates = find_digit_candidates(mask)
    # rois, boxes = extract_both_rows(mask, candidates, num_digits=NUM_DIGITS_PER_ROW)
    rois, boxes = extract_all_rows(mask, candidates, num_digits=NUM_DIGITS_PER_ROW, num_rows=NUM_DIGITS_ROWS)

    chars = []
    digit_boxes = []
    im_debug = True
    debug_digit_frames = []
    usable_w_scale = 0.95 if SMALL_COMPONENT_AREA is None else 1
    for i, roi in enumerate(rois):
        ch, debug_digit_frame = decode_digit(
            roi, density_thresh=DIGIT_DENSITY_THRESH, im_debug=im_debug, usable_w_scale=usable_w_scale
        )
        chars.append(ch)
        x, y, w, h = boxes[i]
        digit_boxes.append((x, y, w, h, ch))
        if debug_digit_frame is not None:
            debug_digit_frames.append(debug_digit_frame)

    if debug_digit_frames:
        show_or_save(
            f"segments", make_debug_grid(debug_digit_frames, scale=2, rows=NUM_DIGITS_ROWS, cols=NUM_DIGITS_PER_ROW)
        )

    # Format as two groups
    if len(chars) == NUM_DIGITS_ROWS * NUM_DIGITS_PER_ROW:
        readout = f"{''.join(chars[:NUM_DIGITS_PER_ROW])} {''.join(chars[NUM_DIGITS_PER_ROW:])}"
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
    # mask = make_red_mask(frame)
    mask = make_without_green_mask(frame)

    candidates = find_digit_candidates(mask)
    rois, _ = extract_all_rows(mask, candidates, num_digits=NUM_DIGITS_PER_ROW, num_rows=NUM_DIGITS_ROWS)
    usable_w_scale = 0.95 if SMALL_COMPONENT_AREA is None else 1

    chars = [decode_digit(r, density_thresh=DIGIT_DENSITY_THRESH, usable_w_scale=usable_w_scale)[0] for r in rois]

    if len(chars) == 6:
        return f"{''.join(chars[:3])} {''.join(chars[3:])}"
    return "".join(chars)
