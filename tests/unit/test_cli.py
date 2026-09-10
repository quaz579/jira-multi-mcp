import subprocess
from pathlib import Path
from typing import Any

import pytest

from jira_multi_mcp import __version__
from jira_multi_mcp.cli import main


def test_version_prints_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_help_and_version_require_no_config_file() -> None:
    # Must work on a clean machine/CI runner with no config.toml anywhere.
    with pytest.raises(SystemExit):
        main(["--help"])
    with pytest.raises(SystemExit):
        main(["--version"])


def test_serve_is_not_implemented_yet(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [defaults]
        username = "bgrossman@jumpmind.com"
        api_token = "token"

        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        """
    )
    exit_code = main(["--config", str(config_path)])
    assert exit_code == 2


def test_missing_config_exits_2_with_no_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.toml"
    exit_code = main(["--check", "--config", str(missing)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_warm_inserts_refresh_right_after_uvx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [defaults]
        username = "bgrossman@jumpmind.com"
        api_token = "token"

        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        """
    )
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(["--warm", "--refresh", "--config", str(config_path)])

    assert exit_code == 0
    assert captured_command == ["uvx", "--refresh", "mcp-atlassian@latest", "--help"]


def test_warm_without_refresh_does_not_insert_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        personal_token = "token"
        """
    )
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = main(["--warm", "--config", str(config_path)])

    assert exit_code == 0
    assert captured_command == ["uvx", "mcp-atlassian@latest", "--help"]
