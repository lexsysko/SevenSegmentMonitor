import logging
import signal
from threading import Event

logger = logging.getLogger(__name__)


def setup_signal_handlers(shutdown_event: Event):
    """
    Cross-platform signal handler setup.
    Works on both Linux/macOS (Docker) and Windows natively.
    """

    def handle_signal(sig, frame=None):
        logger.info(f"\n[System] Received signal {sig}. Triggering graceful shutdown...")
        shutdown_event.set()

    # Standard signal bindings compatible with Windows and Linux
    signal.signal(signal.SIGINT, handle_signal)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signal)  # Docker stop / termination
