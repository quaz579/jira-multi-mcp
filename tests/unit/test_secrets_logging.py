"""Secret masking and log redaction: nothing sensitive reaches stdout/stderr."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from jira_multi_mcp.cli import main
from jira_multi_mcp.secrets import RedactingFilter, Secret

TOKEN = "super-secret-token-value"


def test_secret_repr_and_str_are_masked() -> None:
    secret = Secret(TOKEN)
    assert TOKEN not in repr(secret)
    assert TOKEN not in str(secret)
    assert repr(secret) == "Secret('***')"
    assert str(secret) == "***"
    assert secret.get_secret_value() == TOKEN


def test_secret_equality_compares_underlying_value() -> None:
    assert Secret("a") == Secret("a")
    assert Secret("a") != Secret("b")


def test_redacting_filter_scrubs_secret_from_message() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=f"token is {TOKEN}",
        args=(),
        exc_info=None,
    )
    filt = RedactingFilter([Secret(TOKEN)])
    assert filt.filter(record) is True
    assert TOKEN not in record.getMessage()
    assert "***" in record.getMessage()


def test_redacting_filter_ignores_short_values() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="value is ab",
        args=(),
        exc_info=None,
    )
    filt = RedactingFilter([Secret("ab")])
    filt.filter(record)
    assert record.getMessage() == "value is ab"


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
        [defaults]
        username = "bgrossman@jumpmind.com"
        api_token = "{TOKEN}"

        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        """
    )
    return path


def test_print_config_never_shows_the_real_token(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_config(tmp_path)
    exit_code = main(["--print-config", "--config", str(config_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err
    assert "***" in captured.out
    assert "acme" in captured.out
