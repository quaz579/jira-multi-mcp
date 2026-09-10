"""Logging setup: stderr + a rotating file under XDG state, never stdout."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from jira_multi_mcp.model import AppConfig
from jira_multi_mcp.secrets import RedactingFilter, Secret

LOGGER_NAME = "jira_multi_mcp"

_FORMATTER = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 5


def resolve_log_dir() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return base / "jira-multi-mcp" / "logs"


def _collect_secrets(config: AppConfig) -> list[Secret]:
    secrets: list[Secret] = []
    for site in config.sites:
        if site.api_token is not None:
            secrets.append(site.api_token)
        if site.personal_token is not None:
            secrets.append(site.personal_token)
    if config.defaults.api_token is not None:
        secrets.append(config.defaults.api_token)
    if config.defaults.personal_token is not None:
        secrets.append(config.defaults.personal_token)
    return secrets


def configure_logging(config: AppConfig | None, *, verbose: bool = False) -> logging.Logger:
    """(Re)configures the package logger. Safe to call more than once (e.g. in tests)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    redact = RedactingFilter(_collect_secrets(config) if config is not None else [])

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_FORMATTER)
    stream_handler.addFilter(redact)
    logger.addHandler(stream_handler)

    log_dir = resolve_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler = RotatingFileHandler(
            log_dir / "server.log", maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
        )
        file_handler.setFormatter(_FORMATTER)
        file_handler.addFilter(redact)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("could not open log file under %s; file logging disabled", log_dir)

    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger
