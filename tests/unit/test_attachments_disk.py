"""Disk-safety behavior of ``JiraAttachmentClient.download``: no clobbering,
basename-only filenames, size caps enforced both from ``Content-Length`` and
the actual streamed byte count, and directory/overwrite handling."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from jira_multi_mcp.attachments import JiraAttachmentClient
from jira_multi_mcp.model import SiteConfig
from jira_multi_mcp.secrets import Secret

_ISSUE_URL = "https://acme.atlassian.net/rest/api/3/issue/ACME-1"


def _site() -> SiteConfig:
    return SiteConfig(
        name="acme",
        url="https://acme.atlassian.net",
        key_prefixes=("ACME",),
        username="bgrossman@jumpmind.com",
        api_token=Secret("token"),
    )


def _attachment(att_id: str, filename: str) -> dict[str, object]:
    return {
        "id": att_id,
        "filename": filename,
        "size": 0,
        "mimeType": "text/plain",
        "created": "2026-09-10T12:00:00.000+0000",
        "author": {"displayName": "Ben Grossman"},
        "content": f"https://acme.atlassian.net/rest/api/3/attachment/content/{att_id}",
    }


def _mock_list(attachments: list[dict[str, object]]) -> None:
    respx.get(_ISSUE_URL, params={"fields": "attachment"}).mock(
        return_value=httpx.Response(200, json={"fields": {"attachment": attachments}})
    )


def _mock_content(att_id: str, body: bytes, *, chunked: bool = False) -> None:
    url = f"https://acme.atlassian.net/rest/api/3/attachment/content/{att_id}"
    if chunked:

        async def gen() -> AsyncIterator[bytes]:
            step = 1024
            for i in range(0, len(body), step):
                yield body[i : i + step]

        respx.get(url).mock(return_value=httpx.Response(200, content=gen()))
    else:
        respx.get(url).mock(return_value=httpx.Response(200, content=body))


@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(auth=httpx.BasicAuth("bgrossman@jumpmind.com", "token")) as client:
        yield client


def _client(http_client: httpx.AsyncClient, *, max_bytes: int = 10_000_000) -> JiraAttachmentClient:
    return JiraAttachmentClient(_site(), http_client, max_bytes=max_bytes, redact=lambda t: t)


@respx.mock
async def test_existing_file_falls_back_then_skips_when_both_taken(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_bytes(b"original")
    (tmp_path / "notes-1.txt").write_bytes(b"fallback-taken")
    _mock_list([_attachment("1", "notes.txt")])
    _mock_content("1", b"new-content")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path)

    assert downloaded == []
    assert len(entries) == 1
    assert entries[0]["status"] == "skipped"
    assert (tmp_path / "notes.txt").read_bytes() == b"original"
    assert (tmp_path / "notes-1.txt").read_bytes() == b"fallback-taken"


@respx.mock
async def test_existing_file_falls_back_to_id_suffixed_name(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_bytes(b"original")
    _mock_list([_attachment("1", "notes.txt")])
    _mock_content("1", b"new-content")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path)

    assert entries == []
    assert len(downloaded) == 1
    assert downloaded[0].path == str(tmp_path / "notes-1.txt")
    assert (tmp_path / "notes.txt").read_bytes() == b"original"
    assert (tmp_path / "notes-1.txt").read_bytes() == b"new-content"


@respx.mock
async def test_path_traversal_filename_becomes_basename(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "../../etc/passwd")])
    _mock_content("1", b"payload")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path)

    assert entries == []
    assert len(downloaded) == 1
    result_path = Path(downloaded[0].path)
    assert result_path.parent == tmp_path
    assert result_path.name == "passwd"


@respx.mock
async def test_two_same_named_attachments_both_land(http_client: httpx.AsyncClient, tmp_path: Path) -> None:
    _mock_list([_attachment("1", "notes.txt"), _attachment("2", "notes.txt")])
    _mock_content("1", b"first")
    _mock_content("2", b"second")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path)

    assert entries == []
    assert len(downloaded) == 2
    paths = {Path(d.path).name: Path(d.path).read_bytes() for d in downloaded}
    assert paths["notes.txt"] == b"first"
    assert paths["notes-2.txt"] == b"second"


@respx.mock
async def test_content_length_over_max_bytes_aborts_before_writing(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "big.bin")])
    respx.get("https://acme.atlassian.net/rest/api/3/attachment/content/1").mock(
        return_value=httpx.Response(200, content=b"x" * 20, headers={"content-length": "999999"})
    )

    downloaded, entries = await _client(http_client, max_bytes=100).download("ACME-1", tmp_path)

    assert downloaded == []
    assert entries[0]["status"] == "failed"
    assert "max_bytes" in entries[0]["reason"]
    assert not (tmp_path / "big.bin").exists()


@respx.mock
async def test_streamed_body_exceeding_max_bytes_with_no_content_length_leaves_no_partial_file(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "big.bin")])
    _mock_content("1", b"x" * 5000, chunked=True)

    downloaded, entries = await _client(http_client, max_bytes=100).download("ACME-1", tmp_path)

    assert downloaded == []
    assert entries[0]["status"] == "failed"
    assert "max_bytes" in entries[0]["reason"]
    assert "while streaming" in entries[0]["reason"]
    assert not (tmp_path / "big.bin").exists()
    assert list(tmp_path.iterdir()) == []  # no leftover .part scratch file either


@respx.mock
async def test_missing_target_dir_is_created(http_client: httpx.AsyncClient, tmp_path: Path) -> None:
    _mock_list([_attachment("1", "notes.txt")])
    _mock_content("1", b"hello")
    target = tmp_path / "nested" / "does" / "not" / "exist"

    downloaded, entries = await _client(http_client).download("ACME-1", target)

    assert entries == []
    assert len(downloaded) == 1
    assert Path(downloaded[0].path).read_bytes() == b"hello"


@respx.mock
async def test_overwrite_true_replaces_existing_file(http_client: httpx.AsyncClient, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_bytes(b"stale")
    _mock_list([_attachment("1", "notes.txt")])
    _mock_content("1", b"fresh")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path, overwrite=True)

    assert entries == []
    assert len(downloaded) == 1
    assert downloaded[0].path == str(tmp_path / "notes.txt")
    assert (tmp_path / "notes.txt").read_bytes() == b"fresh"


@respx.mock
async def test_filenames_selection_downloads_only_the_named_attachment(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "a.txt"), _attachment("2", "b.txt")])
    _mock_content("1", b"aaa")
    _mock_content("2", b"bbb")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path, filenames=["b.txt"])

    assert entries == []
    assert [d.filename for d in downloaded] == ["b.txt"]


@respx.mock
async def test_attachment_ids_selection_downloads_only_the_matching_id(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "a.txt"), _attachment("2", "b.txt")])
    _mock_content("1", b"aaa")
    _mock_content("2", b"bbb")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path, attachment_ids=["1"])

    assert entries == []
    assert [d.filename for d in downloaded] == ["a.txt"]


@respx.mock
@pytest.mark.parametrize("bad_name", ["", ".", "..", "\x00\x01\x02"])
async def test_unsafe_filename_is_a_failed_entry_not_a_crash(
    http_client: httpx.AsyncClient, tmp_path: Path, bad_name: str
) -> None:
    _mock_list([_attachment("1", bad_name)])

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path)

    assert downloaded == []
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert entries[0]["reason"] == "unsafe or empty filename"


@respx.mock
async def test_intra_batch_same_name_with_overwrite_lands_both_files(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "notes.txt"), _attachment("2", "notes.txt")])
    _mock_content("1", b"first")
    _mock_content("2", b"second")

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path, overwrite=True)

    assert entries == []
    assert len(downloaded) == 2
    paths = {Path(d.path) for d in downloaded}
    assert len(paths) == 2  # two distinct files, not one overwriting the other
    contents = {path.read_bytes() for path in paths}
    assert contents == {b"first", b"second"}


@respx.mock
async def test_mid_stream_failure_with_overwrite_leaves_original_file_intact(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_bytes(b"original-untouched")
    _mock_list([_attachment("1", "notes.txt")])
    _mock_content("1", b"x" * 5000, chunked=True)

    downloaded, entries = await _client(http_client, max_bytes=100).download(
        "ACME-1", tmp_path, overwrite=True
    )

    assert downloaded == []
    assert entries[0]["status"] == "failed"
    assert (tmp_path / "notes.txt").read_bytes() == b"original-untouched"
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]  # no leftover .part


@respx.mock
async def test_unmatched_attachment_id_is_a_failed_entry_and_creates_no_directory(
    http_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    _mock_list([_attachment("1", "a.txt")])
    target = tmp_path / "fresh-nonexistent-dir"

    downloaded, entries = await _client(http_client).download("ACME-1", target, attachment_ids=["999"])

    assert downloaded == []
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert "999" in entries[0]["reason"]
    assert "ACME-1" in entries[0]["reason"]
    assert not target.exists()


@respx.mock
async def test_unmatched_filename_is_a_failed_entry(http_client: httpx.AsyncClient, tmp_path: Path) -> None:
    _mock_list([_attachment("1", "a.txt")])

    downloaded, entries = await _client(http_client).download("ACME-1", tmp_path, filenames=["nope.txt"])

    assert downloaded == []
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert "nope.txt" in entries[0]["reason"]
