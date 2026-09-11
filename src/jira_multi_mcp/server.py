"""Builds and runs the single mirrored MCP server: starts every configured
child, discovers its tool set, registers the mirrored tools plus
``jira_sites``, and serves over stdio until the client disconnects or a
shutdown signal arrives."""

from __future__ import annotations

import asyncio
import logging
import os
import select
import signal
import threading
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass

import anyio
import anyio.to_thread
from fastmcp import FastMCP

from jira_multi_mcp import __version__
from jira_multi_mcp.attachments import AttachmentClientRegistry
from jira_multi_mcp.children import ChildManager
from jira_multi_mcp.errors import JiraMultiError
from jira_multi_mcp.logging_setup import attach_redaction, configure_logging, resolve_log_dir
from jira_multi_mcp.mirror import build_mirrored_tools
from jira_multi_mcp.model import AppConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.tools_meta import CURATED_TOOLS
from jira_multi_mcp.wrapper_tools import build_attachment_tools, build_jira_sites_tool

_logger = logging.getLogger(__name__)

_SERVER_NAME = "jira-multi-mcp"

# mcp's stdio transport reads stdin in a background thread anyio cannot cancel
# until the pipe actually closes (see mcp.server.stdio.stdio_server), so a
# shutdown signal alone can leave the process running indefinitely even after
# every child has been torn down. This is the hard upper bound on how long a
# supervisor's `kill <pid>` is ever allowed to wait; a module constant so a
# test can lower it. See `_cancel_on_shutdown_signal`.
SHUTDOWN_WATCHDOG_SECONDS = 5.0

# Always shorter than the watchdog itself: bounds the shutdown path's own
# `manager.aclose()` so a hung close is *reported* as incomplete (exit 1)
# instead of silently racing the watchdog to the same `os._exit` call.
_ACLOSE_BUDGET_SECONDS = SHUTDOWN_WATCHDOG_SECONDS - 1


@dataclass
class _ShutdownState:
    """Shared between the signal-handling task and ``_force_exit`` so the
    watchdog's hard exit can report whether children actually got torn down
    cleanly, instead of always claiming success."""

    aclose_completed: bool = False


async def serve(config: AppConfig, *, verbose: bool = False) -> int:
    configure_logging(config, verbose=verbose)
    # Defense in depth: fastmcp's own import-time logging setup attaches a
    # non-propagating RichHandler directly to the "fastmcp" logger. By this
    # point fastmcp is fully imported (configure_logging above already covers
    # this logger via _REDACTED_LOGGER_NAMES), so this re-attach is a no-op
    # in the current call order -- kept so the guarantee holds even if that
    # order ever changes. See logging_setup's module docstring.
    attach_redaction("fastmcp")

    log_dir = resolve_log_dir()
    registry = SiteRegistry(config.sites)
    manager = ChildManager(registry, config.upstream, config.defaults, log_dir, verbose=verbose)
    attachment_clients = AttachmentClientRegistry(config.sites, config.defaults, redact=manager.redact)

    mcp: FastMCP = FastMCP(_SERVER_NAME, version=__version__)
    state = _ShutdownState()
    startup_error: JiraMultiError | None = None

    async with AsyncExitStack() as stack:
        try:
            async with anyio.create_task_group() as tg:
                # Armed BEFORE any startup work runs -- not after `start_all`
                # (the M2r2 MEDIUM finding): `tg.start()` only returns once
                # `open_signal_receiver` has actually been entered inside
                # `_cancel_on_shutdown_signal`, which is what overrides the OS
                # default disposition. Arming it after `start_all` instead
                # (via `tg.start_soon`, which only schedules) left a window --
                # up to `connect_timeout_seconds`, default 90s -- where a
                # SIGTERM hits SIG_DFL: the process dies immediately, no
                # `finally` ever runs, and every already-spawned child is
                # orphaned.
                await tg.start(_cancel_on_shutdown_signal, tg.cancel_scope, manager, state)

                # `run_stdio_async` below is the first thing that ever reads
                # fd 0 -- so a client that abandons the connection (closes
                # its end of stdin without sending a signal) while we're
                # still deep in `start_all`/`discover_tools` goes unnoticed
                # for as long as that startup work takes (up to
                # `connect_timeout_seconds + 30s`), holding every spawned
                # child alive the whole time. Watched only for this startup
                # window -- `stop_event` is set the moment startup finishes
                # (success or failure) so `run_stdio_async` becomes the sole
                # reader of fd 0 from then on, exactly as before this change.
                stop_watching_stdin = threading.Event()
                await tg.start(_watch_stdin_for_eof, tg.cancel_scope, manager, state, stop_watching_stdin)

                try:
                    try:
                        await manager.start_all(
                            stack, connect_timeout=config.defaults.connect_timeout_seconds
                        )
                        await manager.probe_upstream_version(
                            timeout=min(30.0, config.defaults.connect_timeout_seconds)
                        )
                        tools = await manager.discover_tools()

                        allowlist = CURATED_TOOLS if config.defaults.toolset_preset == "curated" else None
                        mirrored = build_mirrored_tools(
                            tools, manager, registry, allowlist, config.defaults.call_timeout_seconds
                        )
                    except JiraMultiError as exc:
                        # Caught here rather than left to escape the task group:
                        # anyio 4 wraps ANY exception leaving a task group --
                        # even one raised directly in the group's own body, not
                        # just a child task -- in an ExceptionGroup, which would
                        # break `cli.main`'s `except JiraMultiError` -> exit-2
                        # handling. Stashed and re-raised once the group has
                        # actually finished unwinding.
                        startup_error = exc
                        tg.cancel_scope.cancel()
                    except Exception:
                        # A shutdown signal (or the stdin-EOF watcher below)
                        # landing mid-startup cancels this scope BEFORE
                        # closing children (see `_cancel_on_shutdown_signal`),
                        # so a child can be force-disconnected out from under
                        # a still-in-flight call here (e.g. `discover_tools`'s
                        # `list_tools()` raising a plain `RuntimeError("Client
                        # is not connected")`, not a `Cancelled` -- confirmed
                        # empirically). That's an artifact of the abort, not a
                        # bug: swallow it once shutdown is already underway.
                        # Anything else (a genuine bug during normal startup)
                        # still propagates.
                        if tg.cancel_scope.cancel_called:
                            _logger.debug("startup work aborted by shutdown", exc_info=True)
                        else:
                            raise
                    else:
                        mcp.add_tool(build_jira_sites_tool(manager))
                        for attachment_tool in build_attachment_tools(registry, attachment_clients):
                            mcp.add_tool(attachment_tool)
                        for tool in mirrored:
                            mcp.add_tool(tool)

                        _log_startup_summary(manager, mirrored)

                        # Startup is over; stop the stdin-EOF watcher before
                        # `run_stdio_async` takes over as fd 0's sole reader
                        # (see the watcher's own module-level comment above).
                        stop_watching_stdin.set()
                        await mcp.run_stdio_async(show_banner=False)
                        tg.cancel_scope.cancel()
                finally:
                    # Guarantees the watcher's background thread notices
                    # within one poll interval and the task in this group
                    # finishes, on EVERY exit from the block above --
                    # including the two exception branches, which never reach
                    # the `stop_watching_stdin.set()` call above. Setting it
                    # any later (e.g. in the `finally:` below) would deadlock:
                    # that block only runs once this task group has already
                    # exited, and the group can't exit while this task is
                    # still blocked waiting to be told to stop.
                    stop_watching_stdin.set()
        finally:
            # Own httpx clients, no subprocess -- closed first (fast, no
            # watchdog risk) so the shared shutdown budget below is spent
            # entirely on the part that can actually hang: tearing down
            # child processes.
            await attachment_clients.aclose()
            # Attempted unconditionally (aclose() is idempotent) even if the
            # signal handler already ran it -- but bounded and its own
            # completion tracked, so a hang here is visible in the exit code
            # rather than masked.
            with anyio.move_on_after(_ACLOSE_BUDGET_SECONDS) as scope:
                await manager.aclose()
            if not scope.cancelled_caught:
                state.aclose_completed = True

    if startup_error is not None:
        raise startup_error

    return 0 if state.aclose_completed else 1


async def _shutdown_children(
    cancel_scope: anyio.CancelScope, manager: ChildManager, state: _ShutdownState
) -> None:
    """Shared teardown for every trigger that means "the client is gone":
    a SIGTERM/SIGINT (`_cancel_on_shutdown_signal`) and stdin hitting EOF
    during startup (`_watch_stdin_for_eof`).

    Arm the hard exit BEFORE tearing children down: aclose() itself can
    block (each child gets fastmcp's own disconnect timeout), and mcp's
    stdio transport blocks this coroutine's cancellation on a background
    thread reading stdin that never notices the cancel scope below -- so
    `serve()` can fail to return even after every child is dead. This
    guarantees the process still exits.
    """
    asyncio.get_running_loop().call_later(SHUTDOWN_WATCHDOG_SECONDS, _force_exit, state)
    # Cancel FIRST, close second: a child whose connect is still in flight
    # (this trigger landing during the startup window) is a task blocked
    # inside fastmcp's `Client._connect()`, which holds that Client's own
    # session lock for as long as it's awaiting the handshake.
    # `manager.aclose()` closes the same Client and needs that identical
    # lock -- it cannot acquire it until the connecting coroutine itself
    # unwinds, and nothing unwinds a stuck handshake except cancellation
    # (verified empirically: aclose()-before-cancel deadlocked here, timing
    # out every time against a hanging fake upstream). Cancelling this scope
    # is what unwinds it -- fastmcp's own CancelledError handler in
    # `Client._connect()` releases the lock and closes the transport. Our
    # own cleanup below is shielded so cancelling the scope we're inside
    # doesn't also cut off THIS coroutine.
    cancel_scope.cancel()
    with anyio.CancelScope(shield=True):
        # Close children directly here rather than trusting the
        # `finally: await manager.aclose()` in `serve()` to ever run -- it
        # only runs once `mcp.run_stdio_async()` (or the startup work)
        # returns, which mcp's stdio transport blocking this coroutine's
        # cancellation on a background thread reading stdin can prevent
        # indefinitely. Bounded so a hung close is reported (exit 1 below)
        # rather than left for the watchdog to silently paper over as
        # success.
        with anyio.move_on_after(_ACLOSE_BUDGET_SECONDS) as scope:
            await manager.aclose()
        state.aclose_completed = not scope.cancelled_caught


async def _cancel_on_shutdown_signal(
    cancel_scope: anyio.CancelScope,
    manager: ChildManager,
    state: _ShutdownState,
    *,
    task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        # `open_signal_receiver` entering its body is what actually arms the
        # OS-level handling (overriding SIG_DFL) -- only now is it safe for
        # `serve()` to start the connect/discover/serve work concurrently.
        task_status.started()
        async for _ in signals:
            _logger.info("received shutdown signal; unwinding children")
            await _shutdown_children(cancel_scope, manager, state)
            return


# How often the stdin-EOF watcher wakes to re-check `stop_event` once fd 0
# has real (non-EOF) data queued or nothing at all -- short enough that
# `stop_event.set()` at the end of startup is noticed promptly, long enough
# not to spin.
_STDIN_WATCH_POLL_MS = 250


def _stdin_hit_eof_blocking(stop_event: threading.Event) -> bool:
    """Runs in a background thread for the whole startup window: True only
    if fd 0's write end closed (the client gave up); False if told to stop.

    Uses ``select.poll()``, not ``select.select()``: a real MCP client
    writes its ``initialize`` request immediately on spawn, so fd 0 is
    typically POLLIN-readable within milliseconds regardless of whether the
    client later abandons the connection -- `select()` can't tell "data
    queued" from "write end closed", but `poll()`'s POLLHUP is set purely by
    the write end closing, independent of whatever bytes are still sitting
    unread in the pipe (verified empirically on macOS with an unread
    ``initialize``-sized payload still buffered). Never calls ``os.read``:
    those queued bytes belong to ``run_stdio_async``'s eventual real reader,
    not to this watcher.
    """
    poller = select.poll()
    poller.register(0, select.POLLIN)
    while not stop_event.is_set():
        for _fd, revents in poller.poll(_STDIN_WATCH_POLL_MS):
            if revents & (select.POLLHUP | select.POLLERR):
                return True
            if revents & select.POLLNVAL:
                # fd 0 isn't pollable at all (e.g. redirected from something
                # unusual) -- not this watcher's problem to diagnose.
                return False
            # POLLIN with no HUP/ERR: the client's own bytes are queued,
            # untouched, for the real reader to pick up once startup ends.
    return False


async def _watch_stdin_for_eof(
    cancel_scope: anyio.CancelScope,
    manager: ChildManager,
    state: _ShutdownState,
    stop_event: threading.Event,
    *,
    task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> None:
    task_status.started()
    hit_eof = await anyio.to_thread.run_sync(_stdin_hit_eof_blocking, stop_event, abandon_on_cancel=True)
    if not hit_eof:
        return
    _logger.info("stdin closed during startup (client disconnected); unwinding children")
    await _shutdown_children(cancel_scope, manager, state)


def _force_exit(state: _ShutdownState) -> None:
    if state.aclose_completed:
        _logger.info(
            "shutdown watchdog forcing exit after %ss (children already closed cleanly)",
            SHUTDOWN_WATCHDOG_SECONDS,
        )
        os._exit(0)
    _logger.warning(
        "shutdown watchdog forcing exit after %ss (children did NOT finish closing)",
        SHUTDOWN_WATCHDOG_SECONDS,
    )
    os._exit(1)


def _log_startup_summary(manager: ChildManager, mirrored: Sequence[object]) -> None:
    health = manager.health()
    healthy_handles = [h for h in health if h["state"] == "healthy"]
    healthy = [str(h["name"]) for h in healthy_handles]
    failed = [str(h["name"]) for h in health if h["state"] == "failed"]
    source = next((str(h["name"]) for h in health if h["discovery_source"]), "union-of-healthy-children")
    log = _logger.error if not healthy else _logger.info
    log(
        "startup summary: %d/%d sites healthy (%s), %d failed (%s), %d tools mirrored, discovery source: %s",
        len(healthy),
        len(health),
        ", ".join(healthy) or "none",
        len(failed),
        ", ".join(failed) or "none",
        len(mirrored),
        source,
    )
    if healthy_handles and all(h["read_only"] or h["enabled_tools_restricted"] for h in healthy_handles):
        _logger.warning(
            "every healthy site (%s) is read_only or has enabled_tools configured; no write "
            "tool is mirrored -- see 'jira_sites' for detail",
            ", ".join(healthy),
        )
