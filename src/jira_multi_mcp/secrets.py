"""Secret value wrapper and a logging filter that redacts secret values."""

from __future__ import annotations

import logging
from collections.abc import Iterable

_MASK = "***"
_MIN_REDACT_LEN = 4


class Secret:
    """Holds a sensitive string so it can't be printed or logged by accident.

    ``repr()`` and ``str()`` always show ``***``; the real value is only
    reachable via :meth:`get_secret_value`.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Secret({_MASK!r})"

    def __str__(self) -> str:
        return _MASK

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)


def _secret_values(secrets: Iterable[Secret]) -> tuple[str, ...]:
    """Longest-first so a token that is a substring of another is still fully masked."""
    return tuple(
        sorted(
            {s.get_secret_value() for s in secrets if len(s.get_secret_value()) >= _MIN_REDACT_LEN},
            key=len,
            reverse=True,
        )
    )


def _redact_with_values(text: str, values: tuple[str, ...]) -> str:
    redacted = text
    for value in values:
        if value in redacted:
            redacted = redacted.replace(value, _MASK)
    return redacted


def redact_text(text: str, secrets: Iterable[Secret]) -> str:
    return _redact_with_values(text, _secret_values(secrets))


class RedactingFilter(logging.Filter):
    """Scrubs known secret values out of a record's raw message/args.

    This only covers ``record.getMessage()``; it does NOT touch a formatted
    traceback (``exc_info``/``exc_text``) or ``stack_info``, which are
    rendered later by the handler's formatter. Use :class:`RedactingFormatter`
    on every handler to cover those too — this filter is kept in addition
    because it lets ``record.msg`` be scrubbed before other filters/handlers
    that don't use the formatter (e.g. anything reading ``record.msg`` directly).
    """

    def __init__(self, secrets: Iterable[Secret]) -> None:
        super().__init__()
        self._values = _secret_values(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._values:
            return True
        message = record.getMessage()
        redacted = _redact_with_values(message, self._values)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    """Scrubs known secret values out of the FULLY formatted record.

    ``RedactingFilter`` only sees ``record.getMessage()``; the traceback text
    a formatter appends for ``exc_info``/``stack_info`` bypasses it entirely,
    so a secret embedded in an exception message (e.g. ``RuntimeError(token)``)
    would otherwise reach stderr and the log file unredacted. Wrap the base
    formatter's output instead of relying on the filter alone.
    """

    def __init__(self, fmt: str, secrets: Iterable[Secret]) -> None:
        super().__init__(fmt)
        self._values = _secret_values(secrets)

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return _redact_with_values(formatted, self._values)
