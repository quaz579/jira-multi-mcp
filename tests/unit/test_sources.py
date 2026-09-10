"""ConfigSource implementations: TOML parsing, env overlay parsing, merge order."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jira_multi_mcp.errors import ConfigError
from jira_multi_mcp.sources import EnvOverlaySource, RawConfig, TomlFileConfigSource, merge_sources


def test_toml_file_source_keys_sites_by_name(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]

        [[sites]]
        name = "beta"
        url = "https://beta.atlassian.net"
        key_prefixes = ["BETA"]
        """
    )
    raw = TomlFileConfigSource(path).load()
    assert set(raw["sites"]) == {"acme", "beta"}
    assert raw["sites"]["acme"]["url"] == "https://acme.atlassian.net"


def test_toml_file_source_rejects_site_missing_name(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[[sites]]\nurl = "https://acme.atlassian.net"\nkey_prefixes = ["ACME"]\n')
    with pytest.raises(ConfigError, match="name"):
        TomlFileConfigSource(path).load()


def test_toml_file_source_rejects_duplicate_name_in_same_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]

        [[sites]]
        name = "acme"
        url = "https://acme2.atlassian.net"
        key_prefixes = ["ACME2"]
        """
    )
    with pytest.raises(ConfigError, match="duplicate"):
        TomlFileConfigSource(path).load()


def test_env_overlay_parses_site_fields() -> None:
    environ = {
        "JIRA_MULTI_SITE_ACME_URL": "https://acme.atlassian.net",
        "JIRA_MULTI_SITE_ACME_KEY_PREFIXES": "ACME, ACMEOPS",
        "JIRA_MULTI_SITE_ACME_API_TOKEN_ENV": "ACME_TOKEN",
        "JIRA_MULTI_SITE_ACME_READ_ONLY": "true",
        "UNRELATED_VAR": "ignored",
    }
    raw = EnvOverlaySource(environ).load()
    assert raw["sites"]["acme"]["url"] == "https://acme.atlassian.net"
    assert raw["sites"]["acme"]["key_prefixes"] == ["ACME", "ACMEOPS"]
    assert raw["sites"]["acme"]["api_token_env"] == "ACME_TOKEN"
    assert raw["sites"]["acme"]["read_only"] is True


def test_env_overlay_handles_underscore_in_site_name() -> None:
    environ = {"JIRA_MULTI_SITE_MY_SITE_URL": "https://my-site.atlassian.net"}
    raw = EnvOverlaySource(environ).load()
    assert raw["sites"]["my_site"]["url"] == "https://my-site.atlassian.net"


def test_env_overlay_rejects_unparseable_bool() -> None:
    environ = {"JIRA_MULTI_SITE_ACME_READ_ONLY": "maybe"}
    with pytest.raises(ConfigError, match="JIRA_MULTI_SITE_ACME_READ_ONLY"):
        EnvOverlaySource(environ).load()


def test_merge_sources_later_wins_field_by_field() -> None:
    first: RawConfig = {
        "defaults": {"username": "a"},
        "upstream": {},
        "sites": {"acme": {"url": "https://acme.atlassian.net", "key_prefixes": ["ACME"]}},
    }
    second: RawConfig = {
        "defaults": {"toolset_preset": "all"},
        "upstream": {},
        "sites": {"acme": {"key_prefixes": ["ACME", "ACMEOPS"]}},
    }
    merged = merge_sources(first, second)
    assert merged["defaults"] == {"username": "a", "toolset_preset": "all"}
    assert merged["sites"]["acme"]["url"] == "https://acme.atlassian.net"
    assert merged["sites"]["acme"]["key_prefixes"] == ["ACME", "ACMEOPS"]


def test_toml_file_source_invalid_utf8_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b'[defaults]\nusername = "\xff\xfe not valid utf-8"\n')
    with pytest.raises(ConfigError, match="UTF-8"):
        TomlFileConfigSource(path).load()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permission bits")
def test_toml_file_source_unreadable_file_is_a_config_error_not_a_raw_oserror(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[[sites]]\nname = "acme"\n')
    path.chmod(0o000)
    try:
        with pytest.raises(ConfigError, match="could not read config file"):
            TomlFileConfigSource(path).load()
    finally:
        path.chmod(0o600)


def test_merge_sources_can_add_a_new_site_without_touching_others() -> None:
    first: RawConfig = {
        "defaults": {},
        "upstream": {},
        "sites": {"acme": {"url": "https://acme.atlassian.net"}},
    }
    second: RawConfig = {
        "defaults": {},
        "upstream": {},
        "sites": {"beta": {"url": "https://beta.atlassian.net"}},
    }
    merged = merge_sources(first, second)
    assert set(merged["sites"]) == {"acme", "beta"}
