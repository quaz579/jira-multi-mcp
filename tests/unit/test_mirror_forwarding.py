"""Drives the mirrored parent server (built from two in-process fake
children) through a real Client: proves `site` stripping, cross-site
routing, error/timeout shaping, and that a wrapper-owned tool is shadowed."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from pathlib import Path

import mcp_types
import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import ClientTransport, FastMCPTransport

from jira_multi_mcp.children import ChildManager
from jira_multi_mcp.mirror import build_mirrored_tools
from jira_multi_mcp.model import Defaults, SiteConfig, UpstreamConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.secrets import Secret
from jira_multi_mcp.tools_meta import CURATED_TOOLS, WRAPPER_OWNED_TOOLS
from jira_multi_mcp.wrapper_tools import build_jira_sites_tool
from tests.fakes.fake_child import make_fake_child

_ALLOWLIST = CURATED_TOOLS | {"jira_boom", "jira_slow", "jira_write_blocked"}
_TIMEOUT = 1.0

_Rig = tuple[Client[ClientTransport], ChildManager, SiteRegistry]


def _text(result: mcp_types.CallToolResult) -> str:
    """First content block's text, asserting it's actually a TextContent (every
    error path in mirror.py only ever emits TextContent)."""
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


def _site(name: str, *prefixes: str, read_only: bool = False) -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.atlassian.net",
        key_prefixes=prefixes,
        username="bgrossman@jumpmind.com",
        api_token=Secret("token"),
        read_only=read_only,
    )


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncGenerator[_Rig, None]:
    registry = SiteRegistry([_site("acme", "ACME"), _site("beta", "BETA", read_only=True)])
    manager = ChildManager(
        registry,
        UpstreamConfig(),
        Defaults(),
        tmp_path,
        transport_factory=lambda site, up: FastMCPTransport(make_fake_child(site.name)),
    )

    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        tools = await manager.discover_tools()
        mirrored = build_mirrored_tools(tools, manager, registry, _ALLOWLIST, timeout=_TIMEOUT)

        parent = FastMCP("test-parent")
        parent.add_tool(build_jira_sites_tool(manager))
        for tool in mirrored:
            parent.add_tool(tool)

        async with Client(FastMCPTransport(parent)) as client:
            yield client, manager, registry


async def test_every_mirrored_tool_has_site_in_its_schema(rig: _Rig) -> None:
    client, _, _ = rig
    tools = await client.list_tools()
    mirrored = [t for t in tools if t.name != "jira_sites"]
    assert mirrored
    for tool in mirrored:
        assert "site" in tool.input_schema["properties"], tool.name


async def test_download_attachments_is_shadowed_from_the_parent(rig: _Rig) -> None:
    client, _, _ = rig
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert "jira_download_attachments" not in names
    assert WRAPPER_OWNED_TOOLS.isdisjoint(names)


async def test_site_is_stripped_before_forwarding_to_the_child(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_get_issue", {"issue_key": "ACME-1", "site": "acme"})
    assert result.is_error is False
    assert result.structured_content == {"site": "acme", "issue_key": "ACME-1"}


async def test_issue_key_prefix_routes_to_the_matching_child(rig: _Rig) -> None:
    client, _, _ = rig
    acme = await client.call_tool_mcp("jira_get_issue", {"issue_key": "ACME-1"})
    beta = await client.call_tool_mcp("jira_get_issue", {"issue_key": "BETA-1"})
    assert acme.structured_content == {"site": "acme", "issue_key": "ACME-1"}
    assert beta.structured_content == {"site": "beta", "issue_key": "BETA-1"}


async def test_explicit_site_wins_over_inference(rig: _Rig) -> None:
    client, _, _ = rig
    # ACME-1 would normally infer 'acme'; force 'beta' instead.
    result = await client.call_tool_mcp("jira_get_issue", {"issue_key": "ACME-1", "site": "beta"})
    assert result.structured_content == {"site": "beta", "issue_key": "ACME-1"}


async def test_ambiguous_call_is_a_tool_error_naming_every_prefix(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_search", {"jql": "project = FOO"})
    assert result.is_error is True
    text = _text(result)
    assert "acme: ACME" in text
    assert "beta: BETA" in text
    assert "jql" in text


async def test_unknown_prefix_names_it_in_the_error(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_get_issue", {"issue_key": "ZZZ-1"})
    assert result.is_error is True
    assert "ZZZ" in _text(result)


async def test_cross_site_arguments_raise_an_error(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp(
        "jira_create_issue_link",
        {
            "link_type": "Blocks",
            "inward_issue_key": "ACME-1",
            "outward_issue_key": "BETA-1",
        },
    )
    assert result.is_error is True
    assert "acme" in _text(result)
    assert "beta" in _text(result)


async def test_child_is_error_gets_a_site_prefix(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_boom", {"site": "acme"})
    assert result.is_error is True
    assert _text(result) == "[site=acme] boom"


async def test_read_only_hint_is_appended_for_a_blocked_write(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_write_blocked", {"issue_key": "BETA-1", "site": "beta"})
    assert result.is_error is True
    assert "[site=beta]" in _text(result)
    assert "read_only = true" in _text(result)


async def test_read_only_hint_is_absent_for_a_non_read_only_site(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_write_blocked", {"issue_key": "ACME-1", "site": "acme"})
    assert result.is_error is True
    assert "read_only" not in _text(result)


async def test_timeout_message_names_site_tool_and_log_path(rig: _Rig) -> None:
    client, manager, _ = rig
    result = await client.call_tool_mcp("jira_slow", {"site": "acme"})
    assert result.is_error is True
    text = _text(result)
    assert "[site=acme]" in text
    assert "jira_slow" in text
    assert "timed out after" in text
    assert str(manager.log_path("acme")) in text


async def test_success_meta_carries_site_and_reason(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_get_issue", {"issue_key": "ACME-1"})
    assert result.meta is not None
    assert result.meta["jira-multi-mcp/site"] == "acme"
    assert "ACME-1" in result.meta["jira-multi-mcp/site_reason"]


async def test_jira_sites_shape_and_no_credentials(rig: _Rig) -> None:
    client, _, _ = rig
    result = await client.call_tool_mcp("jira_sites", {})
    assert result.is_error is False
    sites = result.structured_content["result"]
    assert {s["name"] for s in sites} == {"acme", "beta"}
    for site in sites:
        assert site["state"] == "healthy"
        assert "token" not in str(site).lower()
        assert "api_token" not in site
        assert "password" not in str(site).lower()
    acme = next(s for s in sites if s["name"] == "acme")
    beta = next(s for s in sites if s["name"] == "beta")
    assert acme["discovery_source"] is True
    assert beta["discovery_source"] is False
    assert acme["host"] == "acme.atlassian.net"
    assert acme["key_prefixes"] == ["ACME"]


async def test_discover_tools_prefers_unrestricted_site_over_read_only(
    tmp_path: Path,
) -> None:
    registry = SiteRegistry([_site("acme", "ACME", read_only=True), _site("beta", "BETA")])
    manager = ChildManager(
        registry,
        UpstreamConfig(),
        Defaults(),
        tmp_path,
        transport_factory=lambda site, up: FastMCPTransport(make_fake_child(site.name)),
    )
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        await manager.discover_tools()
        health = {h["name"]: h for h in manager.health()}
        assert health["beta"]["discovery_source"] is True
        assert health["acme"]["discovery_source"] is False
