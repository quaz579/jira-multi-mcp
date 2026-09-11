"""Real-network integration fixtures: a live server process, driven by a real
fastmcp Client, against whatever ``jira-multi-mcp`` config is actually
configured on this machine. No mocks, no fakes -- see ``test_live_sites.py``'s
module docstring for what that does and doesn't prove.

The whole suite is a no-op unless ``JIRA_MULTI_INTEGRATION=1`` -- see the
``pytestmark`` in each test module, not this file, so ``-m integration``
still *collects* the tests (and reports them skipped with the reason) rather
than silently vanishing them the way a conftest-level ``collect_ignore``
would.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import ClientTransport, StdioTransport

from jira_multi_mcp.config import load_config, resolve_config_path
from jira_multi_mcp.errors import JiraMultiError
from jira_multi_mcp.model import AppConfig

# Three real `uvx mcp-atlassian@latest` children resolving/starting cold can
# each take several seconds; generous so a slow first run doesn't flake.
CLIENT_TIMEOUT_SECONDS = 120.0


def _integration_env() -> dict[str, str]:
    """The current process's own environment, plus an explicit
    ``JIRA_MULTI_CONFIG`` so the spawned server resolves the exact same
    config file this fixture just loaded -- never left to the subprocess's
    own (potentially different) default-path resolution."""
    env = dict(os.environ)
    env.setdefault("JIRA_MULTI_CONFIG", str(resolve_config_path()))
    return env


@pytest.fixture(scope="session")
def real_config() -> AppConfig:
    try:
        return load_config()
    except JiraMultiError as exc:
        pytest.fail(
            f"JIRA_MULTI_INTEGRATION=1 but no usable config was found ({exc}); "
            "run `jira-multi-mcp --check` first to confirm one exists and is valid."
        )


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def live_client() -> AsyncIterator[Client[ClientTransport]]:
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "jira_multi_mcp"],
        env=_integration_env(),
    )
    async with Client(
        transport,
        timeout=CLIENT_TIMEOUT_SECONDS,
        init_timeout=CLIENT_TIMEOUT_SECONDS,
        # fastmcp Client's default mode="auto" probes the newer `server/discover`
        # protocol era before falling back; against this server (built with the
        # same fastmcp version but never exercised over a real stdio subprocess
        # this way before) that probe hangs indefinitely -- confirmed by a raw
        # JSON-RPC probe (plain "initialize" handshake, no discover) getting an
        # instant, correct response on the identical server process. "legacy"
        # skips the probe and uses the classic initialize handshake instead,
        # which every configured client (Claude Code, Claude Desktop) already
        # uses in practice.
        mode="legacy",
    ) as client:
        yield client
