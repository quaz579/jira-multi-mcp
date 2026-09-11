"""Spawns and supervises one upstream ``mcp-atlassian`` child process per site.

Each child is started concurrently and failures are isolated: one site being
unreachable never blocks the others from becoming healthy, and never prevents
the server from starting (it just serves fewer tools — see ``discover_tools``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO
from urllib.parse import urlsplit

import anyio
import anyio.to_thread
import mcp_types
from fastmcp import Client
from fastmcp.client.transports import ClientTransport, StdioTransport
from fastmcp.exceptions import ToolError
from mcp import MCPError

from jira_multi_mcp.model import Defaults, SiteConfig, UpstreamConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.secrets import Secret, redact_text, scrub_urls
from jira_multi_mcp.tools_meta import CURATED_TOOLS, WRAPPER_OWNED_TOOLS

_logger = logging.getLogger(__name__)

ChildState = Literal["starting", "healthy", "failed"]

# The smallest env a child subprocess needs on top of what a site/upstream
# config adds: PATH so the interpreter/uvx can even be found, HOME because uv
# and various libraries read it for cache/config dirs. No ambient secrets
# (e.g. an unrelated API token in the parent's own env) leak in by default.
BASE_ENV_PASSTHROUGH: tuple[str, ...] = ("PATH", "HOME")

# Upstream's ENABLED_TOOLS parsing treats an EMPTY string as "no filter"
# (`get_enabled_tools()` in servers/main.py returns None for an empty env
# var, and None means "serve everything"). A site whose configured
# enabled_tools is made up entirely of WRAPPER_OWNED_TOOLS names (e.g. only
# "jira_download_attachments") would otherwise compute an empty
# ENABLED_TOOLS and the child would silently serve its FULL tool set --
# exactly backwards from what the site was restricted to. This sentinel can
# never match a real upstream tool name (upstream's should_include_tool does
# exact membership, never a prefix/glob match), forcing the child to serve
# zero tools instead of falling back to "no filter".
EMPTY_ENABLED_TOOLS_SENTINEL = "__jira_multi_mcp_none__"


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
        effective_allowlist = CURATED_TOOLS
    else:
        effective_allowlist = None  # preset "all", no site override: no restriction
    if effective_allowlist is not None:
        # WRAPPER_OWNED_TOOLS are served by the wrapper itself (jira_download_attachments,
        # M3); the child must never advertise or run its own version of them.
        names = sorted(effective_allowlist - WRAPPER_OWNED_TOOLS)
        if names:
            env["ENABLED_TOOLS"] = ",".join(names)
        else:
            env["ENABLED_TOOLS"] = EMPTY_ENABLED_TOOLS_SENTINEL
            _logger.warning(
                "site '%s': enabled_tools contains only wrapper-owned tool names; forcing "
                "ENABLED_TOOLS=%r so the child serves none, instead of upstream's "
                "empty-string 'no filter' fallback",
                site.name,
                EMPTY_ENABLED_TOOLS_SENTINEL,
            )

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

_LOG_FILE_MODE = 0o600


def _collect_secrets(sites: Sequence[SiteConfig]) -> list[Secret]:
    secrets: list[Secret] = []
    for site in sites:
        if site.api_token is not None:
            secrets.append(site.api_token)
        if site.personal_token is not None:
            secrets.append(site.personal_token)
    return secrets


def _open_child_log_file(site: SiteConfig, log_dir: Path) -> TextIO:
    """Opens this site's child log 0600, matching ``server.log`` -- a bare
    ``Path`` handed to ``StdioTransport`` gets opened at the process umask
    (0644 here), leaving the child's raw stderr more readable than our own
    logs. Must return a real OS-backed file (``fileno()``-capable): mcp's
    ``stdio_client(errlog=...)`` hands it straight to the subprocess as its
    stderr fd, so a pure-Python write()-only wrapper breaks connecting
    entirely (confirmed empirically) -- meaning this stream can NOT be
    redacted the way the logging-based ``server.log`` is; a future upstream
    version that ever echoed a credential here would leak it verbatim.

    Called at most ONCE per site for the whole process's life (see
    ``ChildManager._log_file_for``): fastmcp's ``StdioTransport`` never
    closes a caller-supplied ``log_file``, so opening a fresh one on every
    connect attempt -- including every recovery restart -- would leak one fd
    per attempt for as long as the process runs. Opened in append mode and
    reused across restarts of the same site instead.
    """
    path = log_dir / f"{site.name}.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _LOG_FILE_MODE)
    return os.fdopen(fd, mode="a", encoding="utf-8", errors="replace")


def default_transport_factory(
    site: SiteConfig,
    upstream: UpstreamConfig,
    *,
    defaults: Defaults,
    log_file: TextIO,
    verbose: bool = False,
) -> ClientTransport:
    return StdioTransport(
        command=upstream.command[0],
        args=list(upstream.command[1:]),
        env=build_child_env(site, upstream, defaults, verbose=verbose),
        cwd=upstream.workspace_dir,
        keep_alive=True,
        log_file=log_file,
    )


@dataclass(slots=True)
class ChildHandle:
    site: SiteConfig
    client: Client[ClientTransport] | None
    state: ChildState
    error: str | None = None
    log_path: Path = field(default_factory=Path)
    connected_at: float | None = None
    fastmcp_server_version: str | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    timeouts: int = 0
    # Recovery bookkeeping (M4): ``next_retry_at`` is the earliest time
    # ``client_for`` will attempt to restart this child once it's `failed`;
    # ``recovery_attempts`` is a lifetime counter (never reset on success --
    # it answers "how many restarts did this site ever need", not "how many
    # since it last went healthy"). ``recovery_lock`` serializes concurrent
    # callers hitting `client_for` for the same site the instant the cooldown
    # expires, so two tool calls racing in that window don't both restart the
    # child.
    next_retry_at: float | None = None
    recovery_attempts: int = 0
    recovery_lock: anyio.Lock = field(default_factory=anyio.Lock)


# A single slow call against an otherwise-healthy child shouldn't take the
# whole site down; only a run of these in a row (no successful call in
# between) is treated as evidence the child itself is stuck.
_MAX_CONSECUTIVE_TIMEOUTS = 3


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
        # One log file per site, opened lazily and reused across every
        # (re)connect attempt for that site -- see `_open_child_log_file`'s
        # docstring for why opening a fresh one per attempt would leak fds.
        self._log_files: dict[str, TextIO] = {}
        self._transport_factory: TransportFactory = transport_factory or (
            lambda site, up: default_transport_factory(
                site, up, defaults=defaults, log_file=self._log_file_for(site, log_dir), verbose=verbose
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
        self._secrets: list[Secret] = _collect_secrets(registry.sites)
        self._upstream_version: str | None = None
        self._close_logged = False

    def redact(self, text: str) -> str:
        """Masks every configured site's credential out of model- or
        log-visible text built outside the logging redaction path (e.g. a
        ``ToolError`` message forwarded straight to the calling model).

        Also strips any http(s) URL's query string (``scrub_urls``): a
        pre-signed CDN/media URL (e.g. Jira's attachment content redirect)
        can carry its own credential as a query parameter we never
        configured as a known secret, so ``redact_text`` alone wouldn't
        catch it."""
        return scrub_urls(redact_text(text, self._secrets))

    def _log_file_for(self, site: SiteConfig, log_dir: Path) -> TextIO:
        """Returns this site's child log file, opening it the first time
        (startup or the site's very first recovery) and handing back the
        SAME open handle on every later call -- a failed connect attempt or
        a restart never opens a second one. Closed only in ``aclose()``."""
        log_file = self._log_files.get(site.name)
        if log_file is None:
            log_file = _open_child_log_file(site, log_dir)
            self._log_files[site.name] = log_file
        return log_file

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
        timed_out = False
        try:
            with anyio.move_on_after(connect_timeout) as scope:
                transport = self._transport_factory(site, self._upstream)
                client: Client[ClientTransport] = Client(transport)
                # Assigned BEFORE the session is entered (not after): a
                # cancellation landing mid-connect is already handled by
                # fastmcp's own `Client._connect()` (its CancelledError
                # handler closes the transport regardless of whether we ever
                # held a reference), so this reorder isn't what prevents that
                # orphan -- server.py's cancel-before-close signal ordering
                # is (see its module docstring). This is defensive in a
                # different way: a handle that failed or aborted after the
                # transport/process was created stays reachable by
                # `ChildManager.aclose()` instead of only a handle that fully
                # connected. `Client.close()` on a never-entered client is a
                # safe no-op (verified empirically), so this costs nothing on
                # the ordinary failure paths below. Losing the reference here
                # (the M2r1 HIGH finding) is also exactly what let a child
                # whose probe fails after connecting leak for the rest of the
                # process's life.
                handle.client = client
                await stack.enter_async_context(client)
                await self._probe_liveness(client)
            # `move_on_after` (not `fail_after`) so a plain `TimeoutError` an
            # underlying library raises for its own OS-level reason (e.g. a
            # real ETIMEDOUT that happens to fire before our deadline) isn't
            # mistaken for OUR deadline expiring: only `cancelled_caught`
            # means this cancel scope's own timer fired.
            timed_out = scope.cancelled_caught
        except Exception as exc:  # noqa: BLE001 - isolate one site's failure from the rest
            handle.state = "failed"
            handle.error = self.redact(f"{exc.__class__.__name__}: {exc}")
            handle.next_retry_at = time.time() + self._defaults.recovery_cooldown_seconds
            _logger.warning(
                "site '%s' failed to start: %s (see %s)", site.name, handle.error, handle.log_path
            )
            return

        if timed_out:
            handle.state = "failed"
            handle.error = f"connect timed out after {connect_timeout}s"
            handle.next_retry_at = time.time() + self._defaults.recovery_cooldown_seconds
            _logger.warning(
                "site '%s' failed to start: %s (see %s)", site.name, handle.error, handle.log_path
            )
            return

        handle.state = "healthy"
        handle.connected_at = time.time()
        info = client.server_info
        handle.fastmcp_server_version = info.version if info is not None else None
        _logger.info(
            "site '%s' healthy: child fastmcp serverInfo.version=%s", site.name, handle.fastmcp_server_version
        )

    def mark_failed(self, site_name: str, reason: str) -> None:
        """Records a call-time connection failure discovered by the mirror
        (e.g. the child's pipe broke mid-session) so ``jira_sites`` reflects
        it instead of continuing to show a stale ``healthy``. Does not itself
        restart or reconnect the child -- it only stops ``client_for`` handing
        the dead client out again and records when/why. ``client_for`` is
        what actually retries, once ``recovery_cooldown_seconds`` has
        elapsed (see ``_maybe_recover``)."""
        handle = self._handles.get(site_name)
        if handle is None:
            return
        handle.state = "failed"
        redacted = self.redact(reason)
        handle.error = redacted
        handle.last_error = redacted
        handle.last_error_at = time.time()
        handle.next_retry_at = time.time() + self._defaults.recovery_cooldown_seconds
        _logger.warning("site '%s' marked failed after a call: %s", site_name, redacted)

    def mark_timeout(self, site_name: str, reason: str) -> None:
        """Records a per-call timeout (distinct from ``mark_failed``'s
        connection-level failure): the child's pipe is presumably still
        alive, it just didn't answer in time. Surfaced as ``timeouts`` in
        ``jira_sites``; flips to ``failed`` only after
        ``_MAX_CONSECUTIVE_TIMEOUTS`` in a row."""
        handle = self._handles.get(site_name)
        if handle is None:
            return
        redacted = self.redact(reason)
        handle.timeouts += 1
        handle.last_error = redacted
        handle.last_error_at = time.time()
        _logger.warning("site '%s' call timed out (%d consecutive): %s", site_name, handle.timeouts, redacted)
        if handle.timeouts >= _MAX_CONSECUTIVE_TIMEOUTS:
            handle.state = "failed"
            handle.error = redacted
            handle.next_retry_at = time.time() + self._defaults.recovery_cooldown_seconds

    def mark_success(self, site_name: str) -> None:
        """Resets the consecutive-timeout counter after a call that actually
        completed a round trip -- whether the tool itself errored or not,
        either way the child answered, so it isn't stuck."""
        handle = self._handles.get(site_name)
        if handle is not None:
            handle.timeouts = 0

    async def probe_upstream_version(self, *, timeout: float = 30.0) -> str | None:
        """Runs the shared ``upstream.command --version`` once at startup.

        Every site launches the same command, so this is one call, not one
        per site; it also reports mcp-atlassian's OWN version, unlike each
        child's FastMCP ``serverInfo.version`` (that's the bundled fastmcp
        library version, not mcp-atlassian's -- see ``ChildHandle.fastmcp_server_version``).
        Run off the event loop thread since this shells out synchronously;
        never raises -- a probe failure must not block startup. ``stdin`` is
        ``DEVNULL``: an upstream that doesn't recognize ``--version`` could
        otherwise start serving MCP on inherited fd 0 -- the live stdio pipe
        this process itself uses to talk to Claude Code -- and eat its
        ``initialize`` request. Caller passes a ``timeout`` bounded by the
        connect budget so a hanging probe can't itself blow past it.
        """
        command = [*self._upstream.command, "--version"]
        env = minimal_env(self._upstream.env_passthrough)
        try:
            completed = await anyio.to_thread.run_sync(
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    stdin=subprocess.DEVNULL,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a version probe must never block startup
            _logger.warning("upstream_version unavailable: %s: %s", exc.__class__.__name__, exc)
            return None
        if completed.returncode != 0:
            _logger.warning(
                "upstream_version unavailable: exit %d: %s", completed.returncode, completed.stderr.strip()
            )
            return None
        version = completed.stdout.strip()
        if not version:
            # Exit 0 with nothing on stdout is not a version -- leave
            # `upstream_version()` at its `None` default rather than
            # reporting an empty string as if it were one.
            _logger.warning("upstream_version unavailable: command exited 0 but printed nothing")
            return None
        self._upstream_version = version
        _logger.info("upstream command version: %s", version)
        return version

    def upstream_version(self) -> str | None:
        return self._upstream_version

    async def client_for(self, site_name: str) -> Client[ClientTransport]:
        """Resolves ``site_name`` to a healthy client, attempting recovery
        first if it's currently marked ``failed`` and its cooldown has
        elapsed.

        Callers must invoke this from *inside* their own
        ``anyio.fail_after(call_timeout_seconds)`` scope (see
        ``mirror.py``'s ``MultiSiteProxyTool.run``): recovery itself
        (closing the old client, connecting a new one, probing liveness) can
        take several seconds and must count against the caller's own call
        budget rather than running for free beforehand. May raise
        ``ToolError`` directly -- either "site is unavailable" below, or (via
        ``_maybe_recover``) "is recovering; retry shortly" when another
        caller already holds the recovery lock.
        """
        handle = self._handles.get(site_name)
        if handle is not None and handle.state == "failed":
            await self._maybe_recover(handle)
        if handle is None or handle.state != "healthy" or handle.client is None:
            reason = handle.error if handle is not None and handle.error else "not configured"
            raise ToolError(
                f"[site={site_name}] site is unavailable: {reason}. Run 'jira-multi-mcp --check'."
            )
        return handle.client

    async def _maybe_recover(self, handle: ChildHandle) -> None:
        """Re-probes a ``failed`` site once ``recovery_cooldown_seconds`` has
        elapsed since its last failure, restarting the child if so.

        The cheap check (``next_retry_at`` in the past) happens before ever
        touching ``handle.recovery_lock``, so the common case -- a site
        that's healthy, or still within its cooldown -- costs nothing beyond
        a timestamp comparison.

        Uses try-lock semantics (``acquire_nowait``), not ``async with
        handle.recovery_lock``: this now runs *inside* the caller's
        ``anyio.fail_after(call_timeout_seconds)`` (see ``mirror.py``'s
        ``MultiSiteProxyTool.run``), so a second caller queuing behind the
        lock would burn its own call budget waiting on someone else's
        restart instead of getting a fast, clear answer.
        """
        if handle.next_retry_at is not None and time.time() < handle.next_retry_at:
            return
        try:
            handle.recovery_lock.acquire_nowait()
        except anyio.WouldBlock:
            raise ToolError(f"Site '{handle.site.name}' is recovering; retry shortly") from None
        try:
            # Re-check now that the lock is held: another caller may have
            # already completed (or just started, resetting the cooldown on
            # failure) a recovery attempt before this one got here.
            if handle.state != "failed":
                return
            if handle.next_retry_at is not None and time.time() < handle.next_retry_at:
                return
            await self._attempt_recovery(handle)
        finally:
            handle.recovery_lock.release()

    async def _attempt_recovery(self, handle: ChildHandle) -> None:
        site = handle.site
        handle.recovery_attempts += 1
        _logger.info(
            "site '%s': recovery cooldown elapsed, attempting restart (attempt %d)",
            site.name,
            handle.recovery_attempts,
        )
        old_client = handle.client
        if old_client is not None:
            # Bounded: a child that's `failed` from 3 consecutive timeouts is
            # presumably stuck, and `Client.close()` has its own disconnect
            # wait -- don't let a corpse's teardown stall the restart. This
            # attempt now runs inside the caller's own call-timeout budget
            # (see `client_for`'s docstring), so the budget is kept small.
            with anyio.move_on_after(2.0), suppress(Exception):
                await old_client.close()  # type: ignore[no-untyped-call]
        handle.client = None

        # Bounded by whichever of the two timeouts is smaller: recovery now
        # runs inside the caller's `call_timeout_seconds` budget (see
        # `client_for`), so a `connect_timeout_seconds` of e.g. 90s must not
        # be allowed to blow straight through a 10s call timeout on its own.
        connect_timeout = min(self._defaults.connect_timeout_seconds, self._defaults.call_timeout_seconds)
        # Recovery now runs inside the caller's own
        # `anyio.fail_after(call_timeout_seconds)` (see `client_for`'s
        # docstring), so that timeout firing while `__aenter__`/
        # `_probe_liveness` is in flight raises anyio's cancelled-exception
        # type here -- a `BaseException` under the asyncio backend, so a
        # plain `except Exception` wouldn't catch it, and it would escape
        # leaving the handle pointed at a half-connected client with
        # `next_retry_at` still in the past (the very next call would then
        # immediately retry the same stuck connect). Caught explicitly
        # alongside `Exception` so the failure is always recorded, then
        # re-raised so the call's own timeout still actually fires.
        cancelled_exc = anyio.get_cancelled_exc_class()
        try:
            with anyio.move_on_after(connect_timeout) as scope:
                transport = self._transport_factory(site, self._upstream)
                client: Client[ClientTransport] = Client(transport)
                handle.client = client
                await client.__aenter__()  # type: ignore[no-untyped-call]
                await self._probe_liveness(client)
            timed_out = scope.cancelled_caught
        except (Exception, cancelled_exc) as exc:  # noqa: BLE001 - one site's failed recovery must not raise
            self._fail_recovery(handle, self.redact(f"{exc.__class__.__name__}: {exc}"))
            if isinstance(exc, cancelled_exc):
                raise
            return

        if timed_out:
            self._fail_recovery(handle, f"connect timed out after {connect_timeout}s")
            return

        handle.state = "healthy"
        handle.error = None
        handle.timeouts = 0
        handle.connected_at = time.time()
        handle.next_retry_at = None
        info = client.server_info
        handle.fastmcp_server_version = info.version if info is not None else None
        _logger.info(
            "site '%s' recovered: child fastmcp serverInfo.version=%s",
            site.name,
            handle.fastmcp_server_version,
        )

    def _fail_recovery(self, handle: ChildHandle, reason: str) -> None:
        handle.state = "failed"
        handle.error = reason
        handle.last_error = reason
        handle.last_error_at = time.time()
        handle.next_retry_at = time.time() + self._defaults.recovery_cooldown_seconds
        _logger.warning("site '%s' recovery attempt failed: %s", handle.site.name, reason)

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
                    "enabled_tools_restricted": handle.site.enabled_tools is not None,
                    "state": handle.state,
                    "error": handle.error,
                    "last_error": handle.last_error,
                    "last_error_at": handle.last_error_at,
                    "timeouts": handle.timeouts,
                    "recovery_attempts": handle.recovery_attempts,
                    "next_retry_at": handle.next_retry_at,
                    "log_path": str(handle.log_path),
                    "fastmcp_server_version": handle.fastmcp_server_version,
                    "discovery_source": handle.site.name == self._discovery_source,
                }
            )
        return result

    def log_path(self, site_name: str) -> Path | None:
        handle = self._handles.get(site_name)
        return handle.log_path if handle is not None else None

    async def aclose(self) -> None:
        """Explicitly closes every child that ever connected.

        With ``keep_alive=True`` (needed so a healthy child is reused across
        calls instead of respawned each time), ``StdioTransport.connect_session``'s
        ``finally`` deliberately skips ``disconnect()`` -- so merely unwinding
        the caller's ``AsyncExitStack`` (which only runs each ``Client``'s
        normal, ref-counted ``__aexit__``) never asks the transport to
        terminate the subprocess; upstream mcp-atlassian only exits on its own
        a few seconds after noticing its parent is gone. ``Client.close()``
        forces a real disconnect (killing the child process tree -- see
        ``mcp.client.stdio.stdio_client``'s teardown) and is idempotent, so
        this is safe to call even though the exit stack will still run each
        client's ordinary ``__aexit__`` afterward, and safe to call twice
        (e.g. once from a shutdown-signal handler, once from here) --
        ``_close_logged`` only keeps that second call from logging the same
        "shutting down" line again.
        """
        if not self._close_logged:
            self._close_logged = True
            _logger.info("child manager shutting down")
        async with anyio.create_task_group() as tg:
            for handle in self._handles.values():
                if handle.client is not None:
                    tg.start_soon(self._close_one, handle)
        self._close_log_files()

    def _close_log_files(self) -> None:
        """Closes every site's child log file opened by `_log_file_for`.

        Idempotent via draining the dict (a second `aclose()` call -- e.g.
        once from a shutdown-signal handler, once from `serve()`'s own
        `finally` -- finds it already empty and closes nothing again).
        """
        for name in list(self._log_files):
            log_file = self._log_files.pop(name)
            with suppress(Exception):
                log_file.close()

    async def _close_one(self, handle: ChildHandle) -> None:
        assert handle.client is not None
        try:
            await handle.client.close()  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 - one child's teardown failure must not block the rest
            _logger.warning(
                "error closing child '%s': %s",
                handle.site.name,
                self.redact(f"{exc.__class__.__name__}: {exc}"),
            )
