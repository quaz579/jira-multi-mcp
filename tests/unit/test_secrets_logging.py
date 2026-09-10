"""Secret masking and log redaction: nothing sensitive reaches stdout/stderr."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from jira_multi_mcp.cli import main
from jira_multi_mcp.config import load_config
from jira_multi_mcp.logging_setup import configure_logging, resolve_log_dir
from jira_multi_mcp.secrets import RedactingFilter, RedactingFormatter, Secret, redact_text
from jira_multi_mcp.sources import EnvOverlaySource, TomlFileConfigSource

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


def test_print_config_shows_the_env_var_name_for_an_env_sourced_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACME_TOKEN", TOKEN)
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [defaults]
        username = "bgrossman@jumpmind.com"

        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        api_token_env = "ACME_TOKEN"
        """
    )
    exit_code = main(["--print-config", "--config", str(path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "env:ACME_TOKEN" in captured.out
    assert TOKEN not in captured.out


def test_redacting_formatter_scrubs_a_formatted_traceback() -> None:
    formatter = RedactingFormatter("%(message)s", [Secret(TOKEN)])
    try:
        raise RuntimeError(f"failure near token {TOKEN}")
    except RuntimeError:
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="x",
            args=(),
            exc_info=sys.exc_info(),
        )
    formatted = formatter.format(record)
    assert TOKEN not in formatted
    assert "***" in formatted


def test_redact_text_masks_every_known_secret() -> None:
    assert redact_text(f"a={TOKEN} b=other", [Secret(TOKEN)]) == "a=*** b=other"


def test_configure_logging_redacts_exception_text_from_stderr_and_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = load_config(sources=[TomlFileConfigSource(_write_config(tmp_path)), EnvOverlaySource({})])
    logger = configure_logging(config, verbose=False)

    try:
        raise RuntimeError(f"secret leak attempt {TOKEN}")
    except RuntimeError:
        logger.error("something failed", exc_info=True)

    stderr = capsys.readouterr().err
    assert TOKEN not in stderr
    assert "***" in stderr

    log_file = resolve_log_dir() / "server.log"
    assert log_file.is_file()
    contents = log_file.read_text()
    assert TOKEN not in contents
    assert "***" in contents


def test_configure_logging_installs_redacting_filter_on_every_handler(tmp_path: Path) -> None:
    config = load_config(sources=[TomlFileConfigSource(_write_config(tmp_path)), EnvOverlaySource({})])
    configure_logging(config, verbose=False)
    root = logging.getLogger()
    owned_handlers = [h for h in root.handlers if getattr(h, "_jira_multi_mcp_owned", False)]
    assert owned_handlers
    for handler in owned_handlers:
        assert any(isinstance(f, RedactingFilter) for f in handler.filters)
        assert isinstance(handler.formatter, RedactingFormatter)


def test_log_file_is_created_with_mode_0600(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = load_config(sources=[TomlFileConfigSource(_write_config(tmp_path)), EnvOverlaySource({})])
    configure_logging(config, verbose=False)
    log_file = resolve_log_dir() / "server.log"
    assert log_file.is_file()
    assert (log_file.stat().st_mode & 0o777) == 0o600


def test_print_config_shows_the_dc_auth_branch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
        [[sites]]
        name = "onprem"
        url = "https://jira.example.com"
        key_prefixes = ["ONPREM"]
        personal_token = "{TOKEN}"
        """
    )
    exit_code = main(["--print-config", "--config", str(path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "server_dc" in captured.out
    assert TOKEN not in captured.out
