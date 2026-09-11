"""Tool-level tests for `jira_delete_comment` through an in-process FastMCP
parent, proving site resolution, site-policy enforcement, and comment_id
validation all happen the same way the attachment tools' do."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
import respx
from fastmcp import Client, FastMCP
from fastmcp.client.transports import FastMCPTransport

from jira_multi_mcp.attachments import AttachmentClientRegistry
from jira_multi_mcp.model import Defaults, SiteConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.secrets import Secret
from jira_multi_mcp.wrapper_tools import build_comment_tools


def _cloud_site(
    name: str,
    *prefixes: str,
    read_only: bool = False,
    enabled_tools: frozenset[str] | None = None,
    projects_filter: tuple[str, ...] | None = None,
) -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.atlassian.net",
        key_prefixes=prefixes,
        username="bgrossman@jumpmind.com",
        api_token=Secret(f"{name}-token"),
        read_only=read_only,
        enabled_tools=enabled_tools,
        projects_filter=projects_filter,
    )


def _dc_site(name: str, *prefixes: str) -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.example.com",
        key_prefixes=prefixes,
        personal_token=Secret(f"{name}-dc-token"),
    )


def _server_for(*sites: SiteConfig) -> tuple[SiteRegistry, FastMCP]:
    registry = SiteRegistry(sites)
    attachment_clients = AttachmentClientRegistry(registry.sites, Defaults(), redact=lambda t: t)
    parent = FastMCP("test-parent")
    for tool in build_comment_tools(registry, attachment_clients):
        parent.add_tool(tool)
    return registry, parent


@pytest.fixture
async def client() -> AsyncGenerator[Client[FastMCPTransport], None]:
    _registry, parent = _server_for(_cloud_site("acme", "ACME"), _cloud_site("beta", "BETA"))
    async with Client(FastMCPTransport(parent)) as c:
        yield c


@respx.mock
async def test_delete_comment_infers_site_from_issue_key(client: Client[FastMCPTransport]) -> None:
    respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050").mock(
        return_value=httpx.Response(204)
    )

    result = await client.call_tool_mcp("jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"})

    assert result.is_error is False
    assert result.structured_content == {
        "site": "acme",
        "issue_key": "ACME-1",
        "comment_id": "10050",
        "deleted": True,
    }


@respx.mock
async def test_delete_comment_explicit_site_wins(client: Client[FastMCPTransport]) -> None:
    respx.delete("https://beta.atlassian.net/rest/api/3/issue/ACME-1/comment/10050").mock(
        return_value=httpx.Response(204)
    )

    result = await client.call_tool_mcp(
        "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050", "site": "beta"}
    )

    assert result.is_error is False
    assert result.structured_content["site"] == "beta"


@respx.mock
async def test_delete_comment_403_surfaces_jira_message(client: Client[FastMCPTransport]) -> None:
    respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050").mock(
        return_value=httpx.Response(
            403, json={"errorMessages": ["You do not have permission to delete this comment."]}
        )
    )

    result = await client.call_tool_mcp("jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"})

    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "[site=acme] jira_delete_comment:" in text
    assert "You do not have permission to delete this comment." in text


@respx.mock
async def test_delete_comment_404_surfaces_jira_message(client: Client[FastMCPTransport]) -> None:
    respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/99999").mock(
        return_value=httpx.Response(404, json={"errorMessages": ["Comment does not exist"]})
    )

    result = await client.call_tool_mcp("jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "99999"})

    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "[site=acme] jira_delete_comment: Jira returned 404" in text
    assert "Comment does not exist" in text


async def test_delete_comment_transport_error_is_shaped(client: Client[FastMCPTransport]) -> None:
    with respx.mock:

        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection failed")

        respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050").mock(
            side_effect=boom
        )

        result = await client.call_tool_mcp(
            "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"}
        )

    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "[site=acme] jira_delete_comment:" in text


async def test_delete_comment_refused_on_read_only_site_with_no_http_request_made() -> None:
    _registry, parent = _server_for(_cloud_site("acme", "ACME", read_only=True))

    with respx.mock:
        route = respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "read_only" in text
        assert not route.called


async def test_delete_comment_refused_when_enabled_tools_excludes_it() -> None:
    _registry, parent = _server_for(
        _cloud_site("acme", "ACME", enabled_tools=frozenset({"jira_list_attachments"}))
    )

    with respx.mock:
        route = respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "enabled_tools" in text
        assert not route.called


async def test_delete_comment_refused_by_projects_filter_mismatch() -> None:
    _registry, parent = _server_for(_cloud_site("acme", "ACME", "ZZZ", projects_filter=("ZZZ",)))

    with respx.mock:
        route = respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": "10050"}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "projects_filter" in text
        assert not route.called


@pytest.mark.parametrize("bad_comment_id", ["12/34", "abc", "", "12?x=1", "12 34", "-1", "1.5", "113395\n"])
async def test_delete_comment_rejects_invalid_comment_id_before_any_http_call(
    client: Client[FastMCPTransport], bad_comment_id: str
) -> None:
    """``"113395\\n"`` is the trailing-newline case: ``re.match`` (unlike
    ``re.fullmatch``) lets ``$`` match just before a trailing newline, so a
    comment id ending in ``\\n`` used to slip past validation and reach
    httpx, which then raised an unshaped ``httpx.InvalidURL``."""
    with respx.mock:
        route = respx.delete(url__regex=r"https://acme\.atlassian\.net/rest/api/3/issue/ACME-1/comment/.*")

        result = await client.call_tool_mcp(
            "jira_delete_comment", {"issue_key": "ACME-1", "comment_id": bad_comment_id}
        )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "comment_id" in text
        assert not route.called


@pytest.mark.parametrize(
    "bad_issue_key",
    [
        "ACME-1#x",
        "ACME-1?x=1",
        "ACME-1/comment/1/../../../ACME-2",
        "not-a-key",
        "",
    ],
)
async def test_delete_comment_rejects_invalid_issue_key_before_any_http_call(bad_issue_key: str) -> None:
    """A SINGLE configured site (unlike the two-site `client` fixture above)
    is exactly the reachable path the un-fixed bug hid behind: `resolve_site`
    short-circuits to "only configured site" without ever regex-checking
    `issue_key`, and there's no `projects_filter` configured to trip
    `enforce_site_policy`'s own key check either -- so `_validate_issue_key`
    is the ONLY thing standing between a malicious `issue_key` and the REST
    path built from it."""
    _registry, parent = _server_for(_cloud_site("acme", "ACME"))

    with respx.mock:
        route = respx.delete(url__regex=r"https://acme\.atlassian\.net/rest/api/3/issue/.*")
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_delete_comment", {"issue_key": bad_issue_key, "comment_id": "10050"}
            )

        assert result.is_error is True
        text = result.content[0].text
        assert "issue_key" in text
        assert not route.called


@pytest.mark.parametrize("raw_issue_key", [" acme-1 ", "ACME-1\n", "acme-1"])
@respx.mock
async def test_delete_comment_normalizes_issue_key_case_and_whitespace(raw_issue_key: str) -> None:
    """A single configured site again -- proves normalization (not merely
    rejection) happens: a lowercase and/or whitespace-padded key must still
    resolve and succeed, matching the REST call Jira actually receives for
    the normalized `ACME-1`. `.strip()` deliberately absorbs a trailing
    newline the same way it absorbs leading/trailing spaces."""
    _registry, parent = _server_for(_cloud_site("acme", "ACME"))
    respx.delete("https://acme.atlassian.net/rest/api/3/issue/ACME-1/comment/10050").mock(
        return_value=httpx.Response(204)
    )

    async with Client(FastMCPTransport(parent)) as c:
        result = await c.call_tool_mcp(
            "jira_delete_comment", {"issue_key": raw_issue_key, "comment_id": "10050"}
        )

    assert result.is_error is False
    assert result.structured_content == {
        "site": "acme",
        "issue_key": "ACME-1",
        "comment_id": "10050",
        "deleted": True,
    }


async def test_dc_site_refuses_delete_comment() -> None:
    _registry, parent = _server_for(_dc_site("onprem", "ONPREM"))

    with respx.mock:
        route = respx.delete("https://onprem.example.com/rest/api/3/issue/ONPREM-1/comment/10050")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_delete_comment", {"issue_key": "ONPREM-1", "comment_id": "10050"}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "Jira Cloud sites only in this version" in text
        assert not route.called


async def test_delete_comment_annotations() -> None:
    _registry, parent = _server_for(_cloud_site("acme", "ACME"))
    async with Client(FastMCPTransport(parent)) as c:
        tools = await c.list_tools()
    tool = next(t for t in tools if t.name == "jira_delete_comment")
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is False
    assert tool.annotations.destructive_hint is True
    assert tool.annotations.idempotent_hint is False
