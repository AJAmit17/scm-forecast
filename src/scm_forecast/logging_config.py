"""Centralized logging setup.

Every module logs via `get_logger(__name__)` at INFO (milestones: rows/SKUs
processed, stage timings) and WARNING (a candidate model failed to fit and a
fallback was used - includes the exception type/message, not a vague "it
broke"). This module only decides how those records are ROUTED:

- CLI: `configure_console_logging()` - human-readable lines to stderr.
- Streamlit: `configure_streamlit_logging()` - an in-memory handler whose
  contents `app.py` renders in a "Run log" panel, since a Streamlit user may
  not have (or be watching) the terminal the process was launched from.

Both are idempotent-safe to call repeatedly (Streamlit reruns the whole
script on every interaction) - they reset the `scm_forecast` logger's
handlers each call rather than accumulating duplicates.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "scm_forecast"


def get_logger(name: str) -> logging.Logger:
    """`name` is typically `__name__`; namespaced under the shared root logger."""
    return logging.getLogger(name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}")


def configure_console_logging(level: int = logging.INFO) -> None:
    """CLI entrypoint: plain-text logs to stderr, one handler total."""
    root = logging.getLogger(LOGGER_NAME)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


class InMemoryLogHandler(logging.Handler):
    """Captures formatted records so a UI can render them after the run."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []
        self.max_level = logging.NOTSET

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))
        self.max_level = max(self.max_level, record.levelno)


def configure_streamlit_logging(level: int = logging.INFO) -> InMemoryLogHandler:
    """Streamlit entrypoint: fresh in-memory handler for this script run."""
    root = logging.getLogger(LOGGER_NAME)
    root.handlers.clear()
    handler = InMemoryLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return handler
