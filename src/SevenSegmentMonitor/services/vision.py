import logging
import queue
import time
from queue import Queue
from threading import Event

import cv2
import numpy as np

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


def rotate_frame(frame, angle=7.0):
    """
    Rotates a frame around its center by a specified angle in degrees.
    Positive angle = Counter-Clockwise rotation.
    Negative angle = Clockwise rotation.
    """
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)

    # 1. Obtain the 2D rotation matrix
    # Arguments: center point, rotation angle (degrees), scale factor
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. Perform the affine warp
    # BORDER_REPLICATE prevents dark borders at frame edges after rotation
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def deskew_crop(crop, shear_angle=12):
    """Deskews italicized 7-segment digits by shifting horizontal rows."""
    h, w = crop.shape[:2]
    M = np.float32([
        [1, np.tan(np.radians(-shear_angle)), 0],
        [0, 1, 0]
    ])
    # Shift center origin so cropping doesn't cut off edges
    M[0, 2] = -M[0, 1] * h / 2
    return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR)


def order_points(pts):
    """Sorts 4 coordinates in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect


def preprocess_overexposed_frame(frame: np.ndarray) -> np.ndarray:
    """
    Recovers segment boundaries from severely overexposed LED displays.
    """
    # Convert BGR to 16-bit to prevent arithmetic overflow/underflow
    b, g, r = cv2.split(frame.astype(np.int16))

    # --- Step 1: Red Channel Dominance ---
    # Blown-out centers have high R, G, B, but LED edges retain higher Red relative to Green/Blue.
    # Subtracting the maximum of G/B eliminates ambient white light and background glare.
    red_dominance = r - np.maximum(g, b)
    red_dominance = np.clip(red_dominance, 0, 255).astype(np.uint8)

    # --- Step 2: Non-Linear Gamma Compression ---
    # Gamma < 1.0 sharply suppresses soft halo bloom while retaining high-intensity core pixels.
    gamma = 0.35
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(red_dominance, lut)

    # --- Step 3: Gentle Gaussian Blur ---
    # Smooths high-frequency sensor noise before binarization
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

    # --- Step 4: Otsu's Automatic Thresholding ---
    # Computes optimal threshold dynamically based on histogram bimodality
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 5: Morphological Erosion (Separate Fused Segments) ---
    # Disconnects adjacent 7-segments that have bloomed together into single contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)

    return eroded_mask


def camera_producer_thread(video_src, frame_queue: Queue, stop_event: Event):
    logger.info(f"camera producer is ready. headless: {is_headless()}")

    cap = cv2.VideoCapture(video_src)
    cap.set(cv2.CAP_PROP_FPS, 1)
    # Disable Auto-Exposure (V4L2 backend flags: 1 = manual mode, 3 = auto mode)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    #
    # # Manually set exposure (Try values between -10 and -3, or raw numbers like 20-100 depending on driver)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -10)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Drop oldest frame if worker falls behind
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                ...

        frame_queue.put(frame)
        time.sleep(1)

    cap.release()


def vision_worker_thread(frame_queue: Queue, db_queue: Queue, stop_event: Event):
    logger.info("vision worker is ready")
    last_detected = None

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if not is_headless():
            readout, frames = process_and_annotate(frame)
            # Native GUI window rendering
            for i, frame in enumerate(frames):
                cv2.imshow(f"7-Segment Display Monitor ({i})", frame)

            # cv2.waitKey is REQUIRED for cv2.imshow to update the window frame
            # Pressing 'q' signals all threads to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit signal received from OpenCV GUI.")
                stop_event.set()
                break
        else:
            readout = process_frame(frame)

        logger.debug(f"{readout=}")

        if readout and "?" not in readout and readout != last_detected:
            last_detected = readout
            timestamp = time.time()
            db_queue.put((timestamp, 1, readout))

    if not is_headless():
        cv2.destroyAllWindows()


def process_and_annotate(frame):
    """Processes segments, draws bounding boxes, and decodes digits."""
    # --- 1. PRE-PROCESSING: BLUR & DECREASE BRIGHTNESS ---

    frame = rotate_frame(frame, 5)

    # # Gaussian Blur: Adjust (5, 5) to (9, 9) if you need stronger smoothing
    # # blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0)
    #
    # # Decrease Brightness: beta=-50 reduces intensity (value range: -255 to 0)
    # # alpha=1.0 keeps contrast unchanged
    dimmed_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=-25)
    #
    # --- 2. HSV THRESHOLDING ON PRE-PROCESSED FRAME ---
    hsv = cv2.cvtColor(dimmed_frame, cv2.COLOR_BGR2HSV)

    lower_orange_red = np.array([0, 40, 216])
    upper_orange_red = np.array([179, 186, 255])
    mask_orange = cv2.inRange(hsv, lower_orange_red, upper_orange_red)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    red_mask = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    # red_mask = preprocess_overexposed_frame(frame)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_data = []
    # print("\n")
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # print(i, x, y, w, h)
        if w > 8 and h > 20 and (h / float(w)) > 0.8:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype("float32")

            # 1. Properly order perspective points
            src = order_points(box)

            # Determine width and height
            (tl, tr, br, bl) = src
            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            if width < 5 or height < 5:
                continue

            # Target dimensions (normalized digit box)
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            # 2. Warp Perspective
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(red_mask, M, (width, height))

            # 3. Apply Un-shear (Deskew internal slant)
            warped = deskew_crop(warped, shear_angle=8)

            # 4. Check Segment Densities
            segments_rel = [
                (0.20, 0.00, 0.60, 0.20),  # Top
                (0.65, 0.15, 0.35, 0.35),  # Top-Right
                (0.65, 0.50, 0.35, 0.35),  # Bottom-Right
                (0.20, 0.80, 0.60, 0.20),  # Bottom
                (0.00, 0.50, 0.35, 0.35),  # Bottom-Left
                (0.00, 0.15, 0.35, 0.35),  # Top-Left
                (0.20, 0.40, 0.60, 0.20),  # Middle
            ]

            states = []
            for rx, ry, rw, rh in segments_rel:
                sx, sy = int(rx * width), int(ry * height)
                sw, sh = max(1, int(rw * width)), max(1, int(rh * height))
                seg_crop = warped[sy:sy + sh, sx:sx + sw]
                pixel_density = np.count_nonzero(seg_crop) / float(sw * sh)
                states.append(1 if pixel_density > 0.20 else 0)

            digit_char = DIGIT_MAP.get(tuple(states), "?")
            digit_data.append((x, y, w, h, digit_char))

    digit_data = sorted(digit_data, key=lambda d: (d[1] // 30, d[0]))
    readout_str = "".join([d[4] for d in digit_data])

    # Annotate frame
    for x, y, w, h, val in digit_data:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, val, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(
        frame,
        f"Readout: {readout_str if readout_str else 'Searching...'}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return readout_str, (frame, red_mask)


def process_frame(frame):
    """Heavy OpenCV deskewing and HSV segment processing."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.add(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_data = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 12 and h > 25 and (h / float(w)) > 1.0:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype("float32")

            s = box.sum(axis=1)
            diff = np.diff(box, axis=1)
            tl, br = box[np.argmin(s)], box[np.argmax(s)]
            tr, bl = box[np.argmin(diff)], box[np.argmax(diff)]

            width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            if width < 5 or height < 5:
                continue

            src = np.array([tl, tr, br, bl], dtype="float32")
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(red_mask, M, (width, height))

            segments_rel = [
                (0.20, 0.00, 0.60, 0.20),
                (0.70, 0.15, 0.30, 0.35),
                (0.70, 0.50, 0.30, 0.35),
                (0.20, 0.80, 0.60, 0.20),
                (0.00, 0.50, 0.30, 0.35),
                (0.00, 0.15, 0.30, 0.35),
                (0.20, 0.40, 0.60, 0.20),
            ]

            states = []
            for rx, ry, rw, rh in segments_rel:
                sx, sy = int(rx * width), int(ry * height)
                sw, sh = max(1, int(rw * width)), max(1, int(rh * height))
                seg_crop = warped[sy: sy + sh, sx: sx + sw]
                pixel_density = np.count_nonzero(seg_crop) / float(sw * sh)
                states.append(1 if pixel_density > 0.25 else 0)

            digit_char = DIGIT_MAP.get(tuple(states), "?")
            digit_data.append((x, digit_char))

    digit_data = sorted(digit_data, key=lambda d: d[0])
    readout_str = "".join([d[1] for d in digit_data])

    return readout_str
