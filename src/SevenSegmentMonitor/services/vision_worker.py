import logging
import queue
import time
from queue import Queue
from threading import Event

import cv2
from SevenSegmentMonitor.services.vision import process_and_annotate, process_frame
from SevenSegmentMonitor.tools import is_headless

logger = logging.getLogger(__name__)


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
                logger.info("Quit signal received from OpenCV GUI.")
                stop_event.set()
                break
        else:
            readout = process_frame(frame)

        # logger.debug(f"{readout=}")

        if readout and "?" not in readout and readout != last_detected:
            last_detected = readout
            timestamp = time.time()
            db_queue.put((timestamp, 1, readout))

    if not is_headless():
        cv2.destroyAllWindows()
