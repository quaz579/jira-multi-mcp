"""Loads, merges, and validates configuration into an :class:`AppConfig`."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jira_multi_mcp.errors import ConfigError
from jira_multi_mcp.model import AppConfig, Defaults, SiteConfig, UpstreamConfig
from jira_multi_mcp.secrets import Secret
from jira_multi_mcp.sources import (
    ConfigSource,
    EnvOverlaySource,
    RawConfig,
    TomlFileConfigSource,
    merge_sources,
)
from jira_multi_mcp.tools_meta import CURATED_TOOLS, WRAPPER_OWNED_TOOLS

_logger = logging.getLogger(__name__)

_SITE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_KEY_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")
_TOOLSET_PRESETS = ("curated", "all")

_CURATED_ALLOWLIST = CURATED_TOOLS - WRAPPER_OWNED_TOOLS


def resolve_config_path() -> Path:
    """Where ``load_config`` reads from when no explicit sources are given."""
    env_path = os.environ.get("JIRA_MULTI_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return base / "jira-multi-mcp" / "config.toml"


def load_config(sources: Sequence[ConfigSource] | None = None) -> AppConfig:
    if sources is None:
        sources = [TomlFileConfigSource(resolve_config_path()), EnvOverlaySource()]
    raw = merge_sources(*(source.load() for source in sources))
    return _build_app_config(raw)


def _build_app_config(raw: RawConfig) -> AppConfig:
    sites_raw = raw["sites"]
    if not sites_raw:
        raise ConfigError("no sites configured; add at least one [[sites]] entry")

    defaults = _parse_defaults(raw["defaults"])
    upstream = _parse_upstream(raw["upstream"])

    sites: list[SiteConfig] = []
    prefix_owners: dict[str, str] = {}
    for name, fields in sites_raw.items():
        site = _parse_site(name, fields, defaults)
        for prefix in site.key_prefixes:
            owner = prefix_owners.get(prefix)
            if owner is not None and owner != site.name:
                raise ConfigError(
                    f"key prefix '{prefix}' is used by both site '{owner}' and site '{site.name}'"
                )
            prefix_owners[prefix] = site.name
        sites.append(site)

    return AppConfig(defaults=defaults, upstream=upstream, sites=tuple(sites))


def _parse_defaults(fields: dict[str, Any]) -> Defaults:
    toolset_preset = fields.get("toolset_preset", "curated")
    if toolset_preset not in _TOOLSET_PRESETS:
        raise ConfigError(
            f"defaults.toolset_preset must be one of {_TOOLSET_PRESETS}, got {toolset_preset!r}"
        )

    api_token, api_token_env = _split_token_fields(fields, "api_token", site_label="defaults")
    personal_token, personal_token_env = _split_token_fields(fields, "personal_token", site_label="defaults")

    return Defaults(
        username=fields.get("username"),
        api_token=api_token,
        api_token_env=api_token_env,
        personal_token=personal_token,
        personal_token_env=personal_token_env,
        toolset_preset=toolset_preset,
        call_timeout_seconds=_positive_number(fields, "call_timeout_seconds", 120.0, "defaults"),
        connect_timeout_seconds=_positive_number(fields, "connect_timeout_seconds", 90.0, "defaults"),
        attachment_max_bytes=int(_positive_number(fields, "attachment_max_bytes", 104_857_600, "defaults")),
    )


def _parse_upstream(fields: dict[str, Any]) -> UpstreamConfig:
    command = fields.get("command")
    env_passthrough = fields.get("env_passthrough")
    workspace_dir = fields.get("workspace_dir")
    defaults = UpstreamConfig()
    return UpstreamConfig(
        command=tuple(command) if command is not None else defaults.command,
        env_passthrough=tuple(env_passthrough) if env_passthrough is not None else defaults.env_passthrough,
        workspace_dir=workspace_dir if workspace_dir is not None else defaults.workspace_dir,
    )


def _positive_number(fields: dict[str, Any], key: str, default: float, label: str) -> float:
    value = fields.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label}.{key} must be a number, got {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"{label}.{key} must be greater than zero, got {number}")
    return number


def _split_token_fields(
    fields: dict[str, Any], base: str, *, site_label: str
) -> tuple[Secret | None, str | None]:
    """Reads ``<base>``/``<base>_env`` from a raw dict without resolving env yet."""
    literal = fields.get(base)
    env_name = fields.get(f"{base}_env")
    if literal is not None and env_name is not None:
        raise ConfigError(f"{site_label}: specify only one of '{base}' or '{base}_env', not both")
    if literal is not None:
        return Secret(str(literal)), None
    return None, env_name


def _parse_site(name: str, fields: dict[str, Any], defaults: Defaults) -> SiteConfig:
    if not _SITE_NAME_RE.match(name):
        raise ConfigError(f"site name '{name}' is invalid; names must match {_SITE_NAME_RE.pattern}")

    url = _parse_url(name, fields.get("url"))
    key_prefixes = _parse_key_prefixes(name, fields.get("key_prefixes"))

    username = fields.get("username", defaults.username)
    read_only = bool(fields.get("read_only", False))
    enabled_tools = _parse_enabled_tools(name, fields.get("enabled_tools"), defaults.toolset_preset)

    api_token, api_token_env = _split_token_fields(fields, "api_token", site_label=f"site '{name}'")
    personal_token, personal_token_env = _split_token_fields(
        fields, "personal_token", site_label=f"site '{name}'"
    )
    site_has_cloud = api_token is not None or api_token_env is not None
    site_has_personal = personal_token is not None or personal_token_env is not None
    if site_has_cloud and site_has_personal:
        raise ConfigError(
            f"site '{name}': specify either api_token/api_token_env "
            "or personal_token/personal_token_env, not both"
        )

    if not site_has_cloud and not site_has_personal:
        # Nothing of its own: inherit whichever auth group the defaults define.
        api_token, api_token_env = defaults.api_token, defaults.api_token_env
        personal_token, personal_token_env = defaults.personal_token, defaults.personal_token_env

    resolved_api_token, resolved_api_token_env = _resolve_token(
        api_token, api_token_env, site_name=name, field="api_token"
    )
    resolved_personal_token, resolved_personal_token_env = _resolve_token(
        personal_token, personal_token_env, site_name=name, field="personal_token"
    )

    if resolved_api_token is not None and resolved_personal_token is not None:
        raise ConfigError(
            f"site '{name}': resolved both a Cloud api_token and a Server/DC personal_token; "
            "only one auth shape is allowed"
        )
    if resolved_api_token is None and resolved_personal_token is None:
        raise ConfigError(
            f"site '{name}': no credentials resolved; set api_token/api_token_env (Cloud) "
            "or personal_token/personal_token_env (Server/DC), directly or via [defaults]"
        )
    if resolved_api_token is not None and not username:
        raise ConfigError(
            f"site '{name}': Cloud auth (api_token) requires a username, set directly or via [defaults]"
        )

    return SiteConfig(
        name=name,
        url=url,
        key_prefixes=key_prefixes,
        username=username,
        api_token=resolved_api_token,
        api_token_env=resolved_api_token_env,
        personal_token=resolved_personal_token,
        personal_token_env=resolved_personal_token_env,
        read_only=read_only,
        enabled_tools=enabled_tools,
    )


def _parse_url(site_name: str, raw_url: Any) -> str:
    if not raw_url or not isinstance(raw_url, str):
        raise ConfigError(f"site '{site_name}': 'url' is required")
    parts = urlsplit(raw_url)
    if parts.scheme != "https" or not parts.netloc:
        raise ConfigError(f"site '{site_name}': 'url' must be an https URL, got {raw_url!r}")
    return raw_url.rstrip("/")


def _parse_key_prefixes(site_name: str, raw_prefixes: Any) -> tuple[str, ...]:
    if not raw_prefixes or not isinstance(raw_prefixes, list):
        raise ConfigError(f"site '{site_name}': 'key_prefixes' must be a non-empty list")
    prefixes: list[str] = []
    for prefix in raw_prefixes:
        if not isinstance(prefix, str) or not _KEY_PREFIX_RE.match(prefix):
            raise ConfigError(
                f"site '{site_name}': key prefix {prefix!r} is invalid; must match {_KEY_PREFIX_RE.pattern}"
            )
        prefixes.append(prefix)
    return tuple(prefixes)


def _parse_enabled_tools(site_name: str, raw_tools: Any, toolset_preset: str) -> frozenset[str] | None:
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list) or not all(isinstance(t, str) for t in raw_tools):
        raise ConfigError(f"site '{site_name}': 'enabled_tools' must be a list of strings")
    tools = frozenset(raw_tools)
    if toolset_preset == "curated":
        unknown = tools - _CURATED_ALLOWLIST
        if unknown:
            raise ConfigError(
                f"site '{site_name}': enabled_tools contains tools outside the curated allowlist: "
                f"{', '.join(sorted(unknown))}"
            )
    else:
        not_jira = {t for t in tools if not t.startswith("jira_")}
        if not_jira:
            raise ConfigError(
                f"site '{site_name}': enabled_tools must all be 'jira_'-prefixed tool names: "
                f"{', '.join(sorted(not_jira))}"
            )
    return tools


def _resolve_token(
    literal: Secret | None, env_name: str | None, *, site_name: str, field: str
) -> tuple[Secret | None, str | None]:
    if literal is not None:
        _logger.info(
            "site '%s': %s is set directly in the config file; consider using %s_env instead",
            site_name,
            field,
            field,
        )
        return literal, None
    if env_name is not None:
        value = os.environ.get(env_name)
        if value is None:
            raise ConfigError(
                f"site '{site_name}': environment variable '{env_name}' referenced by {field}_env is not set"
            )
        return Secret(value), env_name
    return None, None
