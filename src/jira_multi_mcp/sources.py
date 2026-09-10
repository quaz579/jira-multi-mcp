"""Config sources: where raw (unvalidated) configuration data comes from.

A source only parses and structures data; all semantic validation (auth
shapes, prefix uniqueness, env var resolution) happens in ``config.py`` so
every source produces errors in one consistent style.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TypedDict

from jira_multi_mcp.errors import ConfigError

_logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class RawConfig(TypedDict):
    defaults: dict[str, Any]
    upstream: dict[str, Any]
    sites: dict[str, dict[str, Any]]


def _empty_raw_config() -> RawConfig:
    return {"defaults": {}, "upstream": {}, "sites": {}}


class ConfigSource(Protocol):
    name: str

    def load(self) -> RawConfig: ...


class TomlFileConfigSource:
    """Reads ``[defaults]``, ``[upstream]``, and ``[[sites]]`` from a TOML file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = f"file:{path}"

    def load(self) -> RawConfig:
        if not self.path.is_file():
            raise ConfigError(f"config file not found: {self.path}")
        self._warn_if_insecure_permissions()
        with self.path.open("rb") as fh:
            try:
                data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"{self.name}: invalid TOML: {exc}") from exc

        sites: dict[str, dict[str, Any]] = {}
        for entry in data.get("sites", []):
            name = entry.get("name")
            if not name:
                raise ConfigError(f"{self.name}: a [[sites]] entry is missing the required 'name' field")
            if name in sites:
                raise ConfigError(f"{self.name}: duplicate [[sites]] entry for name '{name}'")
            sites[name] = dict(entry)

        return {
            "defaults": dict(data.get("defaults", {})),
            "upstream": dict(data.get("upstream", {})),
            "sites": sites,
        }

    def _warn_if_insecure_permissions(self) -> None:
        try:
            mode = self.path.stat().st_mode
        except OSError:
            return
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            _logger.warning(
                "config file %s is readable by group/other (mode %o); consider `chmod 600`",
                self.path,
                stat.S_IMODE(mode),
            )


_SUFFIX_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("_API_TOKEN_ENV", "api_token_env", "str"),
    ("_API_TOKEN", "api_token", "str"),
    ("_PERSONAL_TOKEN_ENV", "personal_token_env", "str"),
    ("_PERSONAL_TOKEN", "personal_token", "str"),
    ("_URL", "url", "str"),
    ("_KEY_PREFIXES", "key_prefixes", "list"),
    ("_USERNAME", "username", "str"),
    ("_READ_ONLY", "read_only", "bool"),
    ("_ENABLED_TOOLS", "enabled_tools", "list"),
)
# Sorted longest-suffix-first so a future field whose suffix is a substring of
# another's can never be matched by the wrong (shorter) entry.
_SUFFIXES_BY_LENGTH = tuple(sorted(_SUFFIX_FIELDS, key=lambda item: len(item[0]), reverse=True))

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


class EnvOverlaySource:
    """Reads ``JIRA_MULTI_SITE_<NAME>_<FIELD>`` variables as per-site overrides.

    ``<NAME>`` becomes the site name, lowercased. Because environment variable
    names can't contain a dash, a site name containing ``-`` (only ``[a-z0-9_-]``
    is allowed by the config's own naming rule) cannot be targeted this way —
    use ``_`` in the site name, or set the value in the TOML file instead.
    """

    PREFIX = "JIRA_MULTI_SITE_"

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self.name = "env"
        self._environ: Mapping[str, str] = environ if environ is not None else os.environ

    def load(self) -> RawConfig:
        sites: dict[str, dict[str, Any]] = {}
        for key, value in self._environ.items():
            if not key.startswith(self.PREFIX):
                continue
            rest = key[len(self.PREFIX) :]
            for suffix, field, kind in _SUFFIXES_BY_LENGTH:
                if not rest.endswith(suffix):
                    continue
                site_name = rest[: -len(suffix)].lower()
                if not site_name:
                    break
                sites.setdefault(site_name, {})[field] = _parse_env_value(value, kind, key)
                break
        return {"defaults": {}, "upstream": {}, "sites": sites}


def _parse_env_value(value: str, kind: str, var_name: str) -> Any:
    if kind == "list":
        return [item.strip() for item in value.split(",") if item.strip()]
    if kind == "bool":
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
        raise ConfigError(f"environment variable {var_name}: {value!r} is not a recognized boolean")
    return value


def merge_sources(*raw_configs: RawConfig) -> RawConfig:
    """Merges raw configs in order; later sources win, field by field.

    ``defaults`` and ``upstream`` are merged as flat dicts (a later source's
    keys overwrite matching keys). ``sites`` is merged per site name, and
    within a site, field by field — so an env overlay can override just one
    field of a site defined in the TOML file without restating the rest.
    """
    merged = _empty_raw_config()
    for raw in raw_configs:
        merged["defaults"].update(raw["defaults"])
        merged["upstream"].update(raw["upstream"])
        for site_name, fields in raw["sites"].items():
            merged["sites"].setdefault(site_name, {}).update(fields)
    return merged
