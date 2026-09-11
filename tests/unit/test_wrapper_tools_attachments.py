"""Tool-level tests for the three attachment tools through an in-process
FastMCP parent, proving `site` resolution (explicit and inferred from
`issue_key`) reaches the correct site's `JiraAttachmentClient`."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp import Client, FastMCP
from fastmcp.client.transports import FastMCPTransport

from jira_multi_mcp.attachments import AttachmentClientRegistry
from jira_multi_mcp.model import Defaults, SiteConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.secrets import Secret
from jira_multi_mcp.wrapper_tools import build_attachment_tools


def _site(
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


def _server_for(*sites: SiteConfig) -> tuple[SiteRegistry, FastMCP]:
    registry = SiteRegistry(sites)
    attachment_clients = AttachmentClientRegistry(registry.sites, Defaults(), redact=lambda t: t)
    parent = FastMCP("test-parent")
    for tool in build_attachment_tools(registry, attachment_clients):
        parent.add_tool(tool)
    return registry, parent


@pytest.fixture
async def client() -> AsyncGenerator[Client[FastMCPTransport], None]:
    _registry, parent = _server_for(_site("acme", "ACME"), _site("beta", "BETA"))
    async with Client(FastMCPTransport(parent)) as c:
        yield c


@respx.mock
async def test_jira_list_attachments_infers_site_from_issue_key(client: Client[FastMCPTransport]) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "fields": {
                    "attachment": [
                        {
                            "id": "1",
                            "filename": "a.txt",
                            "size": 1,
                            "mimeType": "text/plain",
                            "created": "2026-01-01T00:00:00.000+0000",
                            "author": {"displayName": "Ben"},
                            "content": "https://acme.atlassian.net/rest/api/3/attachment/content/1",
                        }
                    ]
                }
            },
        )
    )

    result = await client.call_tool_mcp("jira_list_attachments", {"issue_key": "ACME-1"})

    assert result.is_error is False
    assert result.structured_content["site"] == "acme"
    assert result.structured_content["attachments"][0]["filename"] == "a.txt"


@respx.mock
async def test_jira_list_attachments_explicit_site_wins(client: Client[FastMCPTransport]) -> None:
    respx.get("https://beta.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(200, json={"fields": {"attachment": []}})
    )

    result = await client.call_tool_mcp("jira_list_attachments", {"issue_key": "ACME-1", "site": "beta"})

    assert result.is_error is False
    assert result.structured_content["site"] == "beta"


async def test_jira_list_attachments_ambiguous_without_site_is_a_tool_error(
    client: Client[FastMCPTransport],
) -> None:
    result = await client.call_tool_mcp("jira_list_attachments", {"issue_key": "ZZZ-1"})
    assert result.is_error is True


async def test_jira_upload_attachments_rejects_a_missing_path(
    client: Client[FastMCPTransport], tmp_path: Path
) -> None:
    result = await client.call_tool_mcp(
        "jira_upload_attachments",
        {"issue_key": "ACME-1", "paths": [str(tmp_path / "does-not-exist.txt")]},
    )
    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "not an existing regular file" in text


@respx.mock
async def test_jira_download_attachments_writes_to_disk_and_returns_paths(
    client: Client[FastMCPTransport], tmp_path: Path
) -> None:
    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-2", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "fields": {
                    "attachment": [
                        {
                            "id": "5",
                            "filename": "screenshot.png",
                            "size": 3,
                            "mimeType": "image/png",
                            "created": "2026-01-01T00:00:00.000+0000",
                            "author": {"displayName": "Ben"},
                            "content": "https://acme.atlassian.net/rest/api/3/attachment/content/5",
                        }
                    ]
                }
            },
        )
    )
    respx.get("https://acme.atlassian.net/rest/api/3/attachment/content/5").mock(
        return_value=httpx.Response(200, content=b"\x89PN")
    )

    result = await client.call_tool_mcp(
        "jira_download_attachments", {"issue_key": "ACME-2", "target_dir": str(tmp_path)}
    )

    assert result.is_error is False
    payload = result.structured_content
    assert payload["downloaded"][0]["filename"] == "screenshot.png"
    written_path = Path(payload["downloaded"][0]["path"])
    assert written_path.read_bytes() == b"\x89PN"


async def test_upload_refused_on_read_only_site_with_no_http_request_made(tmp_path: Path) -> None:
    _registry, parent = _server_for(_site("acme", "ACME", read_only=True))
    upload_path = tmp_path / "file.txt"
    upload_path.write_text("hello")

    with respx.mock:
        route = respx.post("https://acme.atlassian.net/rest/api/3/issue/ACME-1/attachments")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_upload_attachments", {"issue_key": "ACME-1", "paths": [str(upload_path)]}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "read_only" in text
        assert not route.called


async def test_list_and_download_still_work_on_a_read_only_site(tmp_path: Path) -> None:
    _registry, parent = _server_for(_site("acme", "ACME", read_only=True))

    with respx.mock:
        respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
            return_value=httpx.Response(200, json={"fields": {"attachment": []}})
        )
        async with Client(FastMCPTransport(parent)) as c:
            list_result = await c.call_tool_mcp("jira_list_attachments", {"issue_key": "ACME-1"})
            download_result = await c.call_tool_mcp(
                "jira_download_attachments", {"issue_key": "ACME-1", "target_dir": str(tmp_path)}
            )

        assert list_result.is_error is False
        assert download_result.is_error is False


async def test_upload_refused_when_enabled_tools_excludes_it(tmp_path: Path) -> None:
    _registry, parent = _server_for(_site("acme", "ACME", enabled_tools=frozenset({"jira_list_attachments"})))
    upload_path = tmp_path / "file.txt"
    upload_path.write_text("hello")

    with respx.mock:
        route = respx.post("https://acme.atlassian.net/rest/api/3/issue/ACME-1/attachments")
        c: Client[FastMCPTransport]
        async with Client(FastMCPTransport(parent)) as c:
            result = await c.call_tool_mcp(
                "jira_upload_attachments", {"issue_key": "ACME-1", "paths": [str(upload_path)]}
            )

        assert result.is_error is True
        text = result.content[0].text  # type: ignore[union-attr]
        assert "enabled_tools" in text
        assert not route.called


async def test_projects_filter_mismatch_is_a_tool_error() -> None:
    _registry, parent = _server_for(_site("acme", "ACME", "ZZZ", projects_filter=("ZZZ",)))

    c: Client[FastMCPTransport]
    async with Client(FastMCPTransport(parent)) as c:
        result = await c.call_tool_mcp("jira_list_attachments", {"issue_key": "ACME-1"})

    assert result.is_error is True
    text = result.content[0].text  # type: ignore[union-attr]
    assert "projects_filter" in text
