"""Builds and runs the single mirrored MCP server: starts every configured
child, discovers its tool set, registers the mirrored tools plus
``jira_sites``, and serves over stdio until the client disconnects or a
shutdown signal arrives."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Sequence
from contextlib import AsyncExitStack

import anyio
from fastmcp import FastMCP

from jira_multi_mcp import __version__
from jira_multi_mcp.children import ChildManager
from jira_multi_mcp.logging_setup import attach_redaction, configure_logging, resolve_log_dir
from jira_multi_mcp.mirror import build_mirrored_tools
from jira_multi_mcp.model import AppConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.tools_meta import CURATED_TOOLS
from jira_multi_mcp.wrapper_tools import build_jira_sites_tool

_logger = logging.getLogger(__name__)

_SERVER_NAME = "jira-multi-mcp"

# mcp's stdio transport reads stdin in a background thread anyio cannot cancel
# until the pipe actually closes (see mcp.server.stdio.stdio_server), so a
# shutdown signal alone can leave the process running indefinitely even after
# every child has been torn down. This is the hard upper bound on how long a
# supervisor's `kill <pid>` is ever allowed to wait; a module constant so a
# test can lower it. See `_cancel_on_shutdown_signal`.
SHUTDOWN_WATCHDOG_SECONDS = 8.0


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

    mcp: FastMCP = FastMCP(_SERVER_NAME, version=__version__)

    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=config.defaults.connect_timeout_seconds)
        await manager.probe_upstream_version()
        tools = await manager.discover_tools()

        allowlist = CURATED_TOOLS if config.defaults.toolset_preset == "curated" else None
        mirrored = build_mirrored_tools(
            tools, manager, registry, allowlist, config.defaults.call_timeout_seconds
        )

        mcp.add_tool(build_jira_sites_tool(manager))
        for tool in mirrored:
            mcp.add_tool(tool)

        _log_startup_summary(manager, mirrored)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_cancel_on_shutdown_signal, tg.cancel_scope, manager)
                await mcp.run_stdio_async(show_banner=False)
                tg.cancel_scope.cancel()
        finally:
            await manager.aclose()

    return 0


async def _cancel_on_shutdown_signal(cancel_scope: anyio.CancelScope, manager: ChildManager) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for _ in signals:
            _logger.info("received shutdown signal; unwinding children")
            # Arm the hard exit BEFORE tearing children down: aclose() itself
            # can block (each child gets fastmcp's own disconnect timeout),
            # and mcp's stdio transport blocks this coroutine's cancellation
            # on a background thread reading stdin that never notices the
            # cancel scope below -- so `serve()` can fail to return even after
            # every child is dead. This guarantees the process still exits.
            asyncio.get_running_loop().call_later(SHUTDOWN_WATCHDOG_SECONDS, _force_exit)
            # Close children directly here rather than trusting the `finally:
            # await manager.aclose()` in `serve()` to ever run -- it only runs
            # once `mcp.run_stdio_async()` returns, which this same stdin
            # thread problem can prevent indefinitely.
            await manager.aclose()
            cancel_scope.cancel()
            return


def _force_exit() -> None:
    _logger.warning("shutdown watchdog forcing exit after %ss", SHUTDOWN_WATCHDOG_SECONDS)
    os._exit(0)


def _log_startup_summary(manager: ChildManager, mirrored: Sequence[object]) -> None:
    health = manager.health()
    healthy = [str(h["name"]) for h in health if h["state"] == "healthy"]
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
