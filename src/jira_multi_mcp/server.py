"""Builds and runs the single mirrored MCP server: starts every configured
child, discovers its tool set, registers the mirrored tools plus
``jira_sites``, and serves over stdio until the client disconnects or a
shutdown signal arrives."""

from __future__ import annotations

import logging
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


async def serve(config: AppConfig, *, verbose: bool = False) -> int:
    configure_logging(config, verbose=verbose)
    # Load-bearing, not redundant: fastmcp's own import-time logging setup
    # attaches a non-propagating RichHandler directly to the "fastmcp"
    # logger, whose children bypass root's handlers entirely. Re-attaching
    # here (fastmcp is fully imported by this point) guarantees our filter
    # covers whatever handler that setup installed. See logging_setup's
    # module docstring for the full mechanism.
    attach_redaction("fastmcp")

    log_dir = resolve_log_dir()
    registry = SiteRegistry(config.sites)
    manager = ChildManager(registry, config.upstream, config.defaults, log_dir, verbose=verbose)

    mcp: FastMCP = FastMCP(_SERVER_NAME, version=__version__)

    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=config.defaults.connect_timeout_seconds)
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
                tg.start_soon(_cancel_on_shutdown_signal, tg.cancel_scope)
                await mcp.run_stdio_async(show_banner=False)
                tg.cancel_scope.cancel()
        finally:
            await manager.aclose()

    return 0


async def _cancel_on_shutdown_signal(cancel_scope: anyio.CancelScope) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for _ in signals:
            _logger.info("received shutdown signal; unwinding children")
            cancel_scope.cancel()
            return


def _log_startup_summary(manager: ChildManager, mirrored: Sequence[object]) -> None:
    health = manager.health()
    healthy = [str(h["name"]) for h in health if h["state"] == "healthy"]
    failed = [str(h["name"]) for h in health if h["state"] == "failed"]
    source = next((str(h["name"]) for h in health if h["discovery_source"]), "union-of-healthy-children")
    _logger.info(
        "startup summary: %d/%d sites healthy (%s), %d failed (%s), %d tools mirrored, discovery source: %s",
        len(healthy),
        len(health),
        ", ".join(healthy) or "none",
        len(failed),
        ", ".join(failed) or "none",
        len(mirrored),
        source,
    )
