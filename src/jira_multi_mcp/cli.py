"""Command-line entry point: --check, --print-config, --warm, and (M2) serve."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from jira_multi_mcp import __version__
from jira_multi_mcp.config import load_config
from jira_multi_mcp.errors import JiraMultiError
from jira_multi_mcp.logging_setup import configure_logging
from jira_multi_mcp.model import AppConfig, SiteConfig
from jira_multi_mcp.sources import ConfigSource, EnvOverlaySource, TomlFileConfigSource

_CLOUD_MYSELF_PATH = "/rest/api/3/myself"
_SERVER_MYSELF_PATH = "/rest/api/2/myself"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jira-multi-mcp",
        description="One MCP server for many Jira Cloud sites.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (overrides JIRA_MULTI_CONFIG and the XDG default).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--check", action="store_true", help="Verify every configured site authenticates, then exit."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        dest="print_config",
        help="Print the effective configuration with secrets masked, then exit.",
    )
    parser.add_argument(
        "--warm", action="store_true", help="Run the upstream command once to prime its uvx cache, then exit."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="With --warm, force uvx to re-resolve the upstream package instead of using its cache.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="With --check, exit 0 if at least one site authenticates, even if others fail.",
    )
    return parser


def _sources_for(config_path: Path | None) -> list[ConfigSource] | None:
    if config_path is None:
        return None
    return [TomlFileConfigSource(config_path), EnvOverlaySource()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    chosen = [flag for flag in ("check", "print_config", "warm") if getattr(args, flag)]
    if len(chosen) > 1:
        parser.error("only one of --check, --print-config, --warm may be given")
    if args.refresh and not args.warm:
        parser.error("--refresh requires --warm")

    # Configured before load_config so an advisory message logged while parsing
    # (e.g. "consider api_token_env") actually reaches a handler; reconfigured
    # after with the real secrets so the redaction filter covers what follows.
    configure_logging(None, verbose=args.verbose)
    try:
        config = load_config(sources=_sources_for(args.config))
    except JiraMultiError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    configure_logging(config, verbose=args.verbose)

    if args.print_config:
        sys.stdout.write(_render_config(config))
        return 0
    if args.warm:
        return _cmd_warm(config, refresh=args.refresh)
    if args.check:
        return asyncio.run(_cmd_check(config, allow_partial=args.allow_partial))

    sys.stderr.write("serve is implemented in M2\n")
    return 2


def _render_config(config: AppConfig) -> str:
    lines = [
        "[defaults]",
        f"  toolset_preset = {config.defaults.toolset_preset!r}",
        f"  call_timeout_seconds = {config.defaults.call_timeout_seconds}",
        f"  connect_timeout_seconds = {config.defaults.connect_timeout_seconds}",
        f"  attachment_max_bytes = {config.defaults.attachment_max_bytes}",
        "",
        "[upstream]",
        f"  command = {list(config.upstream.command)!r}",
        f"  env_passthrough = {list(config.upstream.env_passthrough)!r}",
        f"  workspace_dir = {config.upstream.workspace_dir!r}",
    ]
    for site in config.sites:
        lines.append("")
        lines.append(f"[[sites]]  # {site.name}")
        lines.append(f"  url = {site.url!r}")
        lines.append(f"  key_prefixes = {list(site.key_prefixes)!r}")
        lines.append(f"  read_only = {site.read_only}")
        if site.api_token is not None:
            source = f"env:{site.api_token_env}" if site.api_token_env else "file"
            lines.append(f"  username = {site.username!r}")
            lines.append(f"  auth = cloud (api_token = *** [{source}])")
        else:
            source = f"env:{site.personal_token_env}" if site.personal_token_env else "file"
            lines.append(f"  auth = server_dc (personal_token = *** [{source}])")
        if site.enabled_tools is not None:
            lines.append(f"  enabled_tools = {sorted(site.enabled_tools)!r}")
    return "\n".join(lines) + "\n"


def _cmd_warm(config: AppConfig, *, refresh: bool) -> int:
    command = list(config.upstream.command)
    if refresh:
        command.insert(1, "--refresh")
    command.append("--help")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except OSError as exc:
        sys.stderr.write(f"error: failed to run upstream command {command!r}: {exc}\n")
        return 1
    if completed.returncode != 0:
        sys.stderr.write(f"warm failed (exit {completed.returncode}): {completed.stderr.strip()}\n")
        return 1
    sys.stdout.write(f"upstream cache warmed: {' '.join(command)}\n")
    return 0


@dataclass(frozen=True, slots=True)
class _CheckResult:
    site: SiteConfig
    ok: bool
    detail: str = ""
    display_name: str = ""
    account_id: str = ""


async def _check_site(site: SiteConfig, timeout: float) -> _CheckResult:
    if site.personal_token is not None:
        path = _SERVER_MYSELF_PATH
        headers = {"Authorization": f"Bearer {site.personal_token.get_secret_value()}"}
        auth = None
    else:
        assert site.api_token is not None
        path = _CLOUD_MYSELF_PATH
        headers = {}
        auth = httpx.BasicAuth(site.username or "", site.api_token.get_secret_value())

    url = f"{site.url}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, auth=auth)
    except httpx.HTTPError as exc:
        return _CheckResult(site=site, ok=False, detail=f"request failed: {exc.__class__.__name__}: {exc}")

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return _CheckResult(site=site, ok=False, detail="HTTP 200 but the response body was not JSON")
        return _CheckResult(
            site=site,
            ok=True,
            display_name=str(data.get("displayName", "")),
            account_id=str(data.get("accountId", data.get("key", ""))),
        )

    detail = f"HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("errorMessages"):
        detail += ": " + "; ".join(str(m) for m in body["errorMessages"])
    return _CheckResult(site=site, ok=False, detail=detail)


async def _cmd_check(config: AppConfig, *, allow_partial: bool) -> int:
    results = await asyncio.gather(
        *(_check_site(site, config.defaults.call_timeout_seconds) for site in config.sites)
    )
    sys.stdout.write(_render_check_table(results))
    all_ok = all(r.ok for r in results)
    any_ok = any(r.ok for r in results)
    if all_ok or (allow_partial and any_ok):
        return 0
    return 1


def _render_check_table(results: Sequence[_CheckResult]) -> str:
    header = f"{'SITE':<12} {'HOST':<28} {'AUTH':<8} {'STATUS':<32} {'PREFIXES'}"
    lines = [header, "-" * len(header)]
    for result in results:
        site = result.site
        host = site.url.removeprefix("https://")
        auth_mode = "bearer" if site.personal_token is not None else "basic"
        prefixes = ", ".join(site.key_prefixes)
        if result.ok:
            status = f"OK {result.display_name} ({result.account_id})"
        else:
            status = f"FAILED {result.detail}"
        lines.append(f"{site.name:<12} {host:<28} {auth_mode:<8} {status:<32} {prefixes}")
    return "\n".join(lines) + "\n"
