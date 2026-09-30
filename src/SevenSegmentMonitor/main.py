import logging
import os
import queue
import sqlite3
import threading
from pathlib import Path

from SevenSegmentMonitor.handler_signal import setup_signal_handlers
from SevenSegmentMonitor.services.db_writer_worker import db_writer_worker_thread, db_cleanup_worker_thread
from SevenSegmentMonitor.services.vision import camera_producer_thread, vision_worker_thread
from SevenSegmentMonitor.settings import APP_VERSION, LOGLEVEL


def setup_logger(level: str | int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger."""
    if isinstance(level, str):
        try:
            resolved_level = int(level)
        except ValueError:
            resolved_level = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        resolved_level = level
    else:
        resolved_level = logging.INFO

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    log = logging.getLogger(Path(__file__).parent.name)
    log.setLevel(resolved_level)
    return log


logger = setup_logger(LOGLEVEL)


# --- 5. MAIN ENTRY POINT ---
def main():
    logger.info(f"{APP_VERSION=}")
    video_src = os.environ.get("CAMERA_PATH", "/dev/video0")
    if video_src.isdigit():
        video_src = int(video_src)

    shutdown_event = threading.Event()

    setup_signal_handlers(shutdown_event)

    # Standard thread-safe queues
    frame_queue = queue.Queue(maxsize=1)
    db_queue = queue.Queue()

    # Create workers
    t_cam = threading.Thread(target=camera_producer_thread, args=(video_src, frame_queue, shutdown_event), daemon=True)
    t_vision = threading.Thread(target=vision_worker_thread, args=(frame_queue, db_queue, shutdown_event), daemon=True)
    t_db_cleanup = threading.Thread(target=db_cleanup_worker_thread, args=(shutdown_event,), daemon=True)
    t_db_writer = threading.Thread(target=db_writer_worker_thread, args=(db_queue, shutdown_event), daemon=False)

    t_cam.start()
    t_vision.start()
    t_db_writer.start()
    t_db_cleanup.start()

    logger.info("Multithreaded Headless Monitor Running...")

    try:
        t_db_writer.join()
    except KeyboardInterrupt:
        logger.info("\nStopping threads...")
        shutdown_event.set()


if __name__ == "__main__":
    main()
