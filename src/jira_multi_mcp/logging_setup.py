"""Logging setup: stderr + a rotating file under XDG state, never stdout.

Handlers are attached to the ROOT logger (not just the package logger) so
that anything a future milestone imports (httpx, fastmcp, ...) is redacted
too, not only records this package emits directly. The package logger keeps
its own level (INFO/DEBUG) and propagates up to those handlers; the root
logger's own level is set high enough to keep third-party libraries quiet by
default.
"""

from __future__ import annotations

import logging
import os
import sys
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path

from jira_multi_mcp.model import AppConfig
from jira_multi_mcp.secrets import RedactingFilter, RedactingFormatter, Secret

LOGGER_NAME = "jira_multi_mcp"

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 5
_FILE_MODE = 0o600

# Marks a handler this module installed on the root logger, so a later
# reconfigure only tears down its own handlers and leaves anything else
# (e.g. pytest's caplog handler) alone.
_OWNED_ATTR = "_jira_multi_mcp_owned"


class _SecureRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler whose log file (and each rotated backup) is 0600."""

    def _open(self) -> TextIOWrapper:
        stream = super()._open()
        os.chmod(self.baseFilename, _FILE_MODE)
        return stream


def resolve_log_dir() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return base / "jira-multi-mcp" / "logs"


def collect_secrets(config: AppConfig) -> list[Secret]:
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


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(config: AppConfig | None, *, verbose: bool = False) -> logging.Logger:
    """(Re)configures logging. Safe to call more than once (e.g. in tests)."""
    package_logger = logging.getLogger(LOGGER_NAME)
    package_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    package_logger.propagate = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
    _remove_owned_handlers(root)

    secrets = collect_secrets(config) if config is not None else []
    formatter = RedactingFormatter(_FORMAT, secrets)
    redact_filter = RedactingFilter(secrets)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redact_filter)
    setattr(stream_handler, _OWNED_ATTR, True)
    root.addHandler(stream_handler)

    log_dir = resolve_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler = _SecureRotatingFileHandler(
            str(log_dir / "server.log"), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redact_filter)
        setattr(file_handler, _OWNED_ATTR, True)
        root.addHandler(file_handler)
    except OSError:
        package_logger.warning("could not open log file under %s; file logging disabled", log_dir)

    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    return package_logger
