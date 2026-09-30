import datetime
import logging
import queue
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from time import sleep

from SevenSegmentMonitor.settings import DB_FILE, CLEANUP_TIMEOUT, CLEANUP_PERIOD_DAYS

logger = logging.getLogger(__name__)

INSERT_EVENTS_DATA_SQL = """ 
                    INSERT INTO events (timestamp, state, raw_data)
                    VALUES (?, ?, ?)
                    """


@contextmanager
def get_db_connection(db_path: Path | None = None):
    """Centralized database connection provider with optimized PRAGMAs."""
    path = db_path or DB_FILE
    conn = sqlite3.connect(path)
    try:
        # Standardize performance & concurrency settings across all connections
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_db_connection(db_path) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                state       INTEGER NOT NULL,
                raw_data    TEXT    NOT NULL
            )
            """
        )
        db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        db.commit()
    logger.info("[DB] init_db done")


def db_writer_worker_thread(
        db_queue: queue.Queue, shutdown_event: Event,
        db_path: Path | None = None,
) -> None:
    init_db(db_path)
    logger.info(f"[DB] Worker is ready")

    with get_db_connection(db_path) as db:
        while not (shutdown_event.is_set() and db_queue.empty()):
            try:
                timestamp, state, raw_data = db_queue.get(timeout=1.0)
                db.execute("INSERT INTO events (timestamp, state, raw_data) VALUES (?, ?, ?)", (timestamp, state, raw_data))
                db.commit()
                logger.debug(f"[{timestamp}] SQL LOGGED: {state=} {raw_data=}")
                db_queue.task_done()
            except queue.Empty:
                continue

    logger.info("[DB] Worker stopped.")


def db_cleanup(cutoff_timestamp: float, db_path: Path | None = None) -> int:
    """Deletes records older than cutoff_timestamp and returns the deleted row count."""
    with get_db_connection(db_path) as db:
        cursor = db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_timestamp,))
        db.commit()
        return cursor.rowcount


def db_cleanup_worker_thread(shutdown_event: Event, db_path: Path | None = None, cleanup_timeout=None):
    cleanup_timeout = cleanup_timeout or CLEANUP_TIMEOUT
    cleanup_period = datetime.timedelta(days=CLEANUP_PERIOD_DAYS).total_seconds()

    if not cleanup_period:
        logger.info("[DB] DB WORKER FOR CLEANUP IS DISABLED")
        return
    logger.info(f"[DB] CLEANUP initialized every {cleanup_timeout} seconds.")

    while not shutdown_event.is_set():
        try:
            cutoff_timestamp = time.time() - cleanup_period
            deleted_count = db_cleanup(cutoff_timestamp, db_path)

            if deleted_count:
                logger.info(f"[DB] Cleanup finished. Deleted {deleted_count} old sensor records.")

        except Exception as e:
            logger.error(f"[DB] Error during database cleanup: {e}", exc_info=True)

        start_sleep = time.time()
        while not shutdown_event.is_set() or time.time() - start_sleep > cleanup_timeout:
            sleep(1)

    logger.info("[DB] Cleanup worker shut down cleanly.")
