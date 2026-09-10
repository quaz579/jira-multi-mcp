"""Spawns and supervises one upstream ``mcp-atlassian`` child process per site.

Each child is started concurrently and failures are isolated: one site being
unreachable never blocks the others from becoming healthy, and never prevents
the server from starting (it just serves fewer tools — see ``discover_tools``).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import anyio
import mcp_types
from fastmcp import Client
from fastmcp.client.transports import ClientTransport, StdioTransport
from fastmcp.exceptions import ToolError
from mcp import MCPError

from jira_multi_mcp.model import Defaults, SiteConfig, UpstreamConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.tools_meta import WRAPPER_OWNED_TOOLS

_logger = logging.getLogger(__name__)

ChildState = Literal["starting", "healthy", "failed"]

# The smallest env a child subprocess needs on top of what a site/upstream
# config adds: PATH so the interpreter/uvx can even be found, HOME because uv
# and various libraries read it for cache/config dirs. No ambient secrets
# (e.g. an unrelated API token in the parent's own env) leak in by default.
BASE_ENV_PASSTHROUGH: tuple[str, ...] = ("PATH", "HOME")


def minimal_env(env_passthrough: Sequence[str]) -> dict[str, str]:
    """``BASE_ENV_PASSTHROUGH`` plus any explicitly configured passthrough
    variables that are actually set in the parent's environment. Shared by
    ``cli._warm_env`` (which primes the uvx cache) and ``build_child_env``
    (which additionally injects per-site Jira credentials) so both start from
    an identical, minimal base."""
    names = (*BASE_ENV_PASSTHROUGH, *env_passthrough)
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


def build_child_env(
    site: SiteConfig, upstream: UpstreamConfig, defaults: Defaults, *, verbose: bool = False
) -> dict[str, str]:
    """The full environment for one site's upstream ``mcp-atlassian`` child."""
    env = minimal_env(upstream.env_passthrough)
    env["JIRA_URL"] = site.url
    if site.personal_token is not None:
        env["JIRA_PERSONAL_TOKEN"] = site.personal_token.get_secret_value()
    else:
        assert site.api_token is not None, f"site '{site.name}' has no resolved credentials"
        env["JIRA_USERNAME"] = site.username or ""
        env["JIRA_API_TOKEN"] = site.api_token.get_secret_value()

    # TOOLSETS and ENABLED_TOOLS are ANDed by upstream (servers/main.py), and
    # several toolsets we curate tools from default to disabled — so TOOLSETS
    # must always be "all", with ENABLED_TOOLS doing the real narrowing.
    env["TOOLSETS"] = "all"

    if site.enabled_tools is not None:
        effective_allowlist: frozenset[str] | None = site.enabled_tools
    elif defaults.toolset_preset == "curated":
        from jira_multi_mcp.tools_meta import CURATED_TOOLS

        effective_allowlist = CURATED_TOOLS
    else:
        effective_allowlist = None  # preset "all", no site override: no restriction
    if effective_allowlist is not None:
        # WRAPPER_OWNED_TOOLS are served by the wrapper itself (jira_download_attachments,
        # M3); the child must never advertise or run its own version of them.
        env["ENABLED_TOOLS"] = ",".join(sorted(effective_allowlist - WRAPPER_OWNED_TOOLS))

    if site.read_only:
        env["READ_ONLY_MODE"] = "true"
    if site.projects_filter:
        env["JIRA_PROJECTS_FILTER"] = ",".join(site.projects_filter)
    if verbose:
        env["MCP_VERBOSE"] = "true"
    # So a child's own tool metadata carries FastMCP's `meta` (tags etc.),
    # which the mirror inspects when copying tool definitions.
    env["FASTMCP_INCLUDE_FASTMCP_META"] = "true"
    return env


TransportFactory = Callable[[SiteConfig, UpstreamConfig], ClientTransport]


def default_transport_factory(
    site: SiteConfig,
    upstream: UpstreamConfig,
    *,
    defaults: Defaults,
    log_dir: Path,
    verbose: bool = False,
) -> ClientTransport:
    return StdioTransport(
        command=upstream.command[0],
        args=list(upstream.command[1:]),
        env=build_child_env(site, upstream, defaults, verbose=verbose),
        cwd=upstream.workspace_dir,
        keep_alive=True,
        log_file=log_dir / f"{site.name}.log",
    )


@dataclass(slots=True)
class ChildHandle:
    site: SiteConfig
    client: Client[ClientTransport] | None
    state: ChildState
    error: str | None = None
    log_path: Path = field(default_factory=Path)
    connected_at: float | None = None
    server_version: str | None = None


class ChildManager:
    """Owns one upstream child (and its ``Client``) per configured site."""

    def __init__(
        self,
        registry: SiteRegistry,
        upstream: UpstreamConfig,
        defaults: Defaults,
        log_dir: Path,
        *,
        transport_factory: TransportFactory | None = None,
        verbose: bool = False,
    ) -> None:
        self._registry = registry
        self._upstream = upstream
        self._defaults = defaults
        self._log_dir = log_dir
        self._verbose = verbose
        self._transport_factory: TransportFactory = transport_factory or (
            lambda site, up: default_transport_factory(
                site, up, defaults=defaults, log_dir=log_dir, verbose=verbose
            )
        )
        self._handles: dict[str, ChildHandle] = {
            site.name: ChildHandle(
                site=site, client=None, state="starting", log_path=log_dir / f"{site.name}.log"
            )
            for site in registry.sites
        }
        # The site `discover_tools` picked as the sole schema source, if any
        # (unset while starting, and left `None` after a union-across-children
        # discovery, since no single site is "the" source in that case).
        self._discovery_source: str | None = None

    async def start_all(self, stack: AsyncExitStack, connect_timeout: float) -> None:
        """Connects every configured child concurrently.

        ``stack`` is an ``AsyncExitStack`` the caller owns; each client is
        entered into it so shutdown (``stack.aclose()``) tears every child
        down together. A site whose connect fails or times out is recorded as
        ``failed`` and never raises out of this method — the other sites must
        still get a chance to become healthy.
        """
        async with anyio.create_task_group() as tg:
            for site in self._registry.sites:
                tg.start_soon(self._start_one, site, stack, connect_timeout)

    @staticmethod
    async def _probe_liveness(client: Client[ClientTransport]) -> None:
        """A ``ping`` proves liveness with the least work, but not every MCP
        server implements it: the real upstream (older FastMCP-based
        mcp-atlassian) does, but a lowlevel MCP server built with the locally
        installed FastMCP does not wire an ``on_ping`` handler by default and
        answers "Method not found" -- verified against both during M2. Fall
        back to ``list_tools`` (always implemented) so a server that simply
        doesn't support ping isn't wrongly marked unhealthy.
        """
        try:
            await client.ping()
        except MCPError as exc:
            if "not found" not in str(exc).lower():
                raise
            await client.list_tools()

    async def _start_one(self, site: SiteConfig, stack: AsyncExitStack, connect_timeout: float) -> None:
        handle = self._handles[site.name]
        try:
            transport = self._transport_factory(site, self._upstream)
            client: Client[ClientTransport] = Client(transport)
            with anyio.fail_after(connect_timeout):
                await stack.enter_async_context(client)
                await self._probe_liveness(client)
        except Exception as exc:  # noqa: BLE001 - isolate one site's failure from the rest
            handle.client = None
            handle.state = "failed"
            handle.error = f"{exc.__class__.__name__}: {exc}"
            _logger.warning(
                "site '%s' failed to start: %s (see %s)", site.name, handle.error, handle.log_path
            )
            return

        handle.client = client
        handle.state = "healthy"
        handle.connected_at = time.time()
        info = client.server_info
        handle.server_version = info.version if info is not None else None
        _logger.info("site '%s' healthy: child serverInfo.version=%s", site.name, handle.server_version)
        self._warn_on_version_mismatch()

    def _warn_on_version_mismatch(self) -> None:
        versions = {
            h.server_version
            for h in self._handles.values()
            if h.state == "healthy" and h.server_version is not None
        }
        if len(versions) > 1:
            _logger.warning(
                "configured sites are running different mcp-atlassian versions: %s",
                ", ".join(sorted(versions)),
            )

    async def client_for(self, site_name: str) -> Client[ClientTransport]:
        handle = self._handles.get(site_name)
        if handle is None or handle.state != "healthy" or handle.client is None:
            reason = handle.error if handle is not None and handle.error else "not configured"
            raise ToolError(f"Site '{site_name}' is unavailable: {reason}. Run 'jira-multi-mcp --check'.")
        return handle.client

    async def discover_tools(self) -> list[mcp_types.Tool]:
        """Tools to mirror: prefer the first healthy, unrestricted site (the
        one most likely to expose the full curated set); otherwise union
        across every healthy child, deduped by name; otherwise none."""
        healthy = [h for h in self._handles.values() if h.state == "healthy" and h.client is not None]
        if not healthy:
            return []

        for handle in healthy:
            if handle.site.read_only or handle.site.enabled_tools is not None:
                continue
            assert handle.client is not None
            self._discovery_source = handle.site.name
            return await handle.client.list_tools()

        self._discovery_source = None
        seen: dict[str, mcp_types.Tool] = {}
        for handle in healthy:
            assert handle.client is not None
            for tool in await handle.client.list_tools():
                seen.setdefault(tool.name, tool)
        return list(seen.values())

    def health(self) -> list[dict[str, object]]:
        result = []
        for handle in self._handles.values():
            result.append(
                {
                    "name": handle.site.name,
                    "host": urlsplit(handle.site.url).netloc,
                    "key_prefixes": list(handle.site.key_prefixes),
                    "read_only": handle.site.read_only,
                    "state": handle.state,
                    "error": handle.error,
                    "log_path": str(handle.log_path),
                    "server_version": handle.server_version,
                    "discovery_source": handle.site.name == self._discovery_source,
                }
            )
        return result

    def log_path(self, site_name: str) -> Path | None:
        handle = self._handles.get(site_name)
        return handle.log_path if handle is not None else None

    async def aclose(self) -> None:
        # Clients are entered into the caller's AsyncExitStack (see start_all)
        # and are closed when that stack unwinds; nothing owned directly here
        # needs explicit teardown, but this gives server.py a single named
        # call site to hang shutdown logging off of.
        _logger.info("child manager shutting down")
