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


class RedactingFilter(logging.Filter):
    """Scrubs known secret values out of every log record it sees.

    Must be attached to each handler individually: a filter on a logger does
    not apply to records a child logger propagates through it, only to
    records that logger itself emits directly.
    """

    def __init__(self, secrets: Iterable[Secret]) -> None:
        super().__init__()
        self._values = tuple(
            sorted(
                {s.get_secret_value() for s in secrets if len(s.get_secret_value()) >= _MIN_REDACT_LEN},
                key=len,
                reverse=True,
            )
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._values:
            return True
        message = record.getMessage()
        redacted = message
        for value in self._values:
            if value in redacted:
                redacted = redacted.replace(value, _MASK)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True
