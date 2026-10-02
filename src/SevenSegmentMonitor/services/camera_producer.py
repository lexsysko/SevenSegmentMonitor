import logging
import queue
import time
from itertools import cycle
from queue import Queue
from threading import Event

import cv2
from SevenSegmentMonitor.settings import SAVE_FRAME, FRAME_FPS_DELAY, FRAME_NAME_PATH, LOAD_FRAME
from SevenSegmentMonitor.tools import is_headless

logger = logging.getLogger(__name__)


def camera_producer_thread(video_src, frame_queue: Queue, stop_event: Event):
    logger.info(f"camera producer is ready. headless: {is_headless()}")
    cycled_list = None
    cap = None
    if LOAD_FRAME and FRAME_NAME_PATH:
        if FRAME_NAME_PATH.stem.find("*"):
            glob_list = FRAME_NAME_PATH.parent.glob(FRAME_NAME_PATH.stem)
        else:
            glob_list = (FRAME_NAME_PATH,)
        cycled_list = cycle(glob_list)

    while not stop_event.is_set():
        if cycled_list is not None:
            filename = next(cycled_list)
            if not filename.exists():
                logger.error(f"{filename} does not exist")
                continue
            frame = cv2.imread(filename)
            ret = True
            logger.debug(filename.name)
        else:
            cap = cv2.VideoCapture(video_src)
            ret, frame = cap.read()

        if not ret or frame is None:
            continue

        if SAVE_FRAME and FRAME_NAME_PATH and not LOAD_FRAME:
            if FRAME_NAME_PATH.stem.find("*"):
                counter = time.time_ns() // 10
                filename_stem = FRAME_NAME_PATH.stem.replace("*", f"-{counter}")
                filename = FRAME_NAME_PATH.with_stem(filename_stem)
            else:
                filename = FRAME_NAME_PATH
            cv2.imwrite(filename, frame)
            logger.debug(f"Saved frame to file: {filename}")

        # Drop oldest frame if worker falls behind
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                ...

        frame_queue.put(frame)

        if FRAME_FPS_DELAY:
            time.sleep(FRAME_FPS_DELAY)

    if cap is not None:
        cap.release()
