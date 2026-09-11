"""HTTP-level behavior of ``JiraAttachmentClient`` against a respx-mocked Jira
Cloud REST v3, including the cross-host redirect for attachment content and
that a child's error body never lets a credential reach the model."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp.exceptions import ToolError

from jira_multi_mcp.attachments import JiraAttachmentClient
from jira_multi_mcp.model import SiteConfig
from jira_multi_mcp.secrets import Secret, redact_text

_SECRET_TOKEN = "zz-super-secret-api-token-zz"


def _cloud_site(name: str = "acme", *, token: str = _SECRET_TOKEN) -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.atlassian.net",
        key_prefixes=("ACME",),
        username="bgrossman@jumpmind.com",
        api_token=Secret(token),
    )


def _dc_site(name: str = "onprem") -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.example.com",
        key_prefixes=("ONPREM",),
        personal_token=Secret("dc-token"),
    )


def _client(
    site: SiteConfig, http: httpx.AsyncClient, *, max_bytes: int = 10_000_000
) -> JiraAttachmentClient:
    return JiraAttachmentClient(
        site, http, max_bytes=max_bytes, redact=lambda text: redact_text(text, [Secret(_SECRET_TOKEN)])
    )


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(auth=httpx.BasicAuth("bgrossman@jumpmind.com", _SECRET_TOKEN)) as client:
        yield client


@respx.mock
async def test_list_attachments_parses_fields(http_client: httpx.AsyncClient) -> None:
    site = _cloud_site()
    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "fields": {
                    "attachment": [
                        {
                            "id": "10001",
                            "filename": "notes.txt",
                            "size": 42,
                            "mimeType": "text/plain",
                            "created": "2026-09-10T12:00:00.000+0000",
                            "author": {"displayName": "Ben Grossman"},
                            "content": "https://acme.atlassian.net/rest/api/3/attachment/content/10001",
                        }
                    ]
                }
            },
        )
    )

    attachments = await _client(site, http_client).list_attachments("ACME-1")

    assert len(attachments) == 1
    a = attachments[0]
    assert a.id == "10001"
    assert a.filename == "notes.txt"
    assert a.size == 42
    assert a.mime_type == "text/plain"
    assert a.created == "2026-09-10T12:00:00.000+0000"
    assert a.author == "Ben Grossman"
    assert a.content_url == "https://acme.atlassian.net/rest/api/3/attachment/content/10001"


@respx.mock
async def test_download_follows_cross_host_redirect_and_writes_bytes(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    site = _cloud_site()
    content_url = "https://acme.atlassian.net/rest/api/3/attachment/content/10001"
    cdn_url = "https://media-cdn.example-atlassian-media.net/some-signed-path"

    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(
            200,
            json={
                "fields": {
                    "attachment": [
                        {
                            "id": "10001",
                            "filename": "notes.txt",
                            "size": 5,
                            "mimeType": "text/plain",
                            "created": "2026-09-10T12:00:00.000+0000",
                            "author": {"displayName": "Ben Grossman"},
                            "content": content_url,
                        }
                    ]
                }
            },
        )
    )
    respx.get(content_url).mock(return_value=httpx.Response(302, headers={"location": cdn_url}))
    respx.get(cdn_url).mock(return_value=httpx.Response(200, content=b"hello"))

    downloaded, skipped_or_failed = await _client(site, http_client).download("ACME-1", tmp_path)

    assert skipped_or_failed == []
    assert len(downloaded) == 1
    result = downloaded[0]
    assert result.filename == "notes.txt"
    assert Path(result.path).read_bytes() == b"hello"
    assert result.size == 5

    cdn_request = next(call.request for call in respx.calls if call.request.url == cdn_url)
    assert "authorization" not in {h.lower() for h in cdn_request.headers.keys()}


@respx.mock
async def test_upload_sends_multipart_with_no_check_header(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    site = _cloud_site()
    upload_path = tmp_path / "jira-multi-mcp-test.txt"
    upload_path.write_text("hello world")

    route = respx.post("https://acme.atlassian.net/rest/api/3/issue/ACME-1/attachments").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "20002",
                    "filename": "jira-multi-mcp-test.txt",
                    "size": 11,
                    "mimeType": "text/plain",
                    "created": "2026-09-10T12:00:00.000+0000",
                    "author": {"displayName": "Ben Grossman"},
                    "content": "https://acme.atlassian.net/rest/api/3/attachment/content/20002",
                }
            ],
        )
    )

    uploaded = await _client(site, http_client).upload("ACME-1", [upload_path])

    assert route.called
    sent_request = route.calls.last.request
    assert sent_request.headers["X-Atlassian-Token"] == "no-check"
    assert b'name="file"' in sent_request.content
    assert b"jira-multi-mcp-test.txt" in sent_request.content
    assert len(uploaded) == 1
    assert uploaded[0].id == "20002"
    assert uploaded[0].filename == "jira-multi-mcp-test.txt"


@respx.mock
async def test_403_error_carries_jira_messages_and_not_the_token(http_client: httpx.AsyncClient) -> None:
    site = _cloud_site()
    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-1", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(
            403, json={"errorMessages": ["You do not have permission to view this issue."]}
        )
    )

    with pytest.raises(ToolError) as exc_info:
        await _client(site, http_client).list_attachments("ACME-1")

    message = str(exc_info.value)
    assert "403" in message
    assert "You do not have permission to view this issue." in message
    assert _SECRET_TOKEN not in message


@respx.mock
async def test_404_issue_raises_tool_error_with_jira_message(http_client: httpx.AsyncClient) -> None:
    site = _cloud_site()
    respx.get("https://acme.atlassian.net/rest/api/3/issue/ACME-99999", params={"fields": "attachment"}).mock(
        return_value=httpx.Response(404, json={"errorMessages": ["Issue does not exist"]})
    )

    with pytest.raises(ToolError) as exc_info:
        await _client(site, http_client).list_attachments("ACME-99999")

    assert "404" in str(exc_info.value)
    assert "Issue does not exist" in str(exc_info.value)


async def test_dc_site_refuses_all_three_operations(http_client: httpx.AsyncClient, tmp_path: Path) -> None:
    site = _dc_site()
    client = _client(site, http_client)

    with pytest.raises(ToolError, match="Jira Cloud sites only in this version"):
        await client.list_attachments("ONPREM-1")
    with pytest.raises(ToolError, match="Jira Cloud sites only in this version"):
        await client.download("ONPREM-1", tmp_path)
    with pytest.raises(ToolError, match="Jira Cloud sites only in this version"):
        await client.upload("ONPREM-1", [])
