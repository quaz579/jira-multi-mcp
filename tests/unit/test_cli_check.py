"""`--check` against respx-mocked Jira /myself endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from jira_multi_mcp.cli import main

TOKEN = "super-secret-check-token"

CONFIG = f"""
[defaults]
username = "bgrossman@jumpmind.com"
api_token = "{TOKEN}"

[[sites]]
name = "acme"
url = "https://acme.atlassian.net"
key_prefixes = ["ACME"]

[[sites]]
name = "beta"
url = "https://beta.atlassian.net"
key_prefixes = ["BETA"]
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    return path


@respx.mock
def test_check_all_sites_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(200, json={"displayName": "Ben Grossman", "accountId": "acc-acme"})
    )
    respx.get("https://beta.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(200, json={"displayName": "Ben Grossman", "accountId": "acc-beta"})
    )

    exit_code = main(["--check", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "acme" in captured.out
    assert "beta" in captured.out
    assert "Ben Grossman" in captured.out
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


@respx.mock
def test_check_one_site_401_fails_without_leaking_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(200, json={"displayName": "Ben Grossman", "accountId": "acc-acme"})
    )
    respx.get("https://beta.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(401, json={"errorMessages": ["Unauthorized"]})
    )

    exit_code = main(["--check", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "401" in captured.out
    assert "Unauthorized" in captured.out
    assert TOKEN not in captured.out
    assert TOKEN not in captured.err


@respx.mock
def test_check_allow_partial_exits_zero_when_at_least_one_site_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(200, json={"displayName": "Ben Grossman", "accountId": "acc-acme"})
    )
    respx.get("https://beta.atlassian.net/rest/api/3/myself").mock(
        return_value=Response(401, json={"errorMessages": ["Unauthorized"]})
    )

    exit_code = main(["--check", "--allow-partial", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0


@respx.mock
def test_check_allow_partial_still_fails_when_every_site_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/myself").mock(return_value=Response(401, json={}))
    respx.get("https://beta.atlassian.net/rest/api/3/myself").mock(return_value=Response(401, json={}))

    exit_code = main(["--check", "--allow-partial", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
