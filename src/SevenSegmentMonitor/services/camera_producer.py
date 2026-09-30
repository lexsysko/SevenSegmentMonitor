import logging
import queue
import time
from queue import Queue
from threading import Event

import cv2
from SevenSegmentMonitor.settings import SAVE_FRAME, FRAME_FPS_DELAY, FRAME_NAME_PATH
from SevenSegmentMonitor.tools import is_headless

logger = logging.getLogger(__name__)


def camera_producer_thread(video_src, frame_queue: Queue, stop_event: Event):
    logger.info(f"camera producer is ready. headless: {is_headless()}")

    if LOAD_FRAME and FRAME_NAME_PATH:
        cap = cv2.imread(FRAME_NAME_PATH)
    else:
        cap = cv2.VideoCapture(video_src)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if SAVE_FRAME and FRAME_NAME_PATH and not LOAD_FRAME:
            cv2.imwrite(FRAME_NAME_PATH, frame)
            logger.debug(f"Saved frame to file: {FRAME_NAME_PATH}")

        # Drop oldest frame if worker falls behind
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                ...

        frame_queue.put(frame)

        if FRAME_FPS_DELAY:
            time.sleep(FRAME_FPS_DELAY)

    cap.release()
