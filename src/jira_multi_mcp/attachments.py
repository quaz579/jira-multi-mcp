"""Direct Jira Cloud REST v3 attachment client: list, download-to-disk, upload.

Implemented directly against the REST API (not through an upstream child)
because upstream ``mcp-atlassian`` only exposes a base64-in-band download
tool -- see ``tools_meta.WRAPPER_OWNED_TOOLS``. Cloud only in this version:
Server/Data Center attachment endpoints differ and are out of scope (the
plan's v1 decision), so every method refuses on a ``personal_token`` site.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from fastmcp.exceptions import ToolError

from jira_multi_mcp.model import Defaults, SiteConfig

_logger = logging.getLogger(__name__)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class _AttachmentTooLarge(Exception):
    """Internal signal that a streamed download exceeded ``max_bytes``;
    always caught inside ``_download_one`` and turned into a "failed" entry,
    never propagated to a caller."""


@dataclass(frozen=True)
class AttachmentMeta:
    id: str
    filename: str
    size: int
    mime_type: str
    created: str
    author: str
    content_url: str


@dataclass(frozen=True)
class DownloadedFile:
    """``filename`` is the name actually written under ``target_dir`` -- it
    may differ from ``original_filename`` when a same-named collision (either
    on disk already, or against another attachment selected in the same
    call) forced the id-suffixed fallback name."""

    attachment_id: str
    filename: str
    original_filename: str
    path: str
    size: int
    mime_type: str


class _DownloadEntry:
    """One selected attachment that was NOT written to disk (name/reason kept
    separate from ``DownloadedFile`` so a caller can bucket "skipped" --
    already exists, both fallback names taken -- from "failed" -- refused by
    the server, too large, an unsafe filename)."""

    def __init__(self, filename: str, reason: str, status: Literal["skipped", "failed"]) -> None:
        self.filename = filename
        self.reason = reason
        self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"filename": self.filename, "reason": self.reason, "status": self.status}


def _safe_filename(filename: str) -> str:
    """Basename only (discards any directory component a hostile or odd
    server-reported filename might carry), with control characters stripped.
    Empty, ``.``, or ``..`` after that cleanup is never a usable filename."""
    name = Path(filename.strip()).name
    name = _CONTROL_CHARS_RE.sub("", name)
    if name in ("", ".", ".."):
        return ""
    return name


class JiraAttachmentClient:
    """One site's attachment operations, over a caller-supplied ``httpx.AsyncClient``."""

    def __init__(
        self,
        site: SiteConfig,
        http: httpx.AsyncClient,
        *,
        max_bytes: int,
        redact: Callable[[str], str],
    ) -> None:
        self._site = site
        self._http = http
        self._max_bytes = max_bytes
        self._redact = redact

    @property
    def api_base(self) -> str:
        return f"{self._site.url}/rest/api/3"

    def _require_cloud(self, tool_name: str) -> None:
        if self._site.personal_token is not None:
            raise ToolError(
                f"[site={self._site.name}] {tool_name}: attachment tools support Jira Cloud sites "
                f"only in this version (site '{self._site.name}' is configured as Server/Data Center)"
            )

    async def list_attachments(self, issue_key: str) -> list[AttachmentMeta]:
        self._require_cloud("jira_list_attachments")
        path = f"/issue/{issue_key}"
        response = await self._http.get(f"{self.api_base}{path}", params={"fields": "attachment"})
        await self._raise_for_status(response, "GET", path)
        data = response.json()
        raw_attachments = (data.get("fields") or {}).get("attachment") or []
        return [self._parse_attachment(raw) for raw in raw_attachments]

    async def download(
        self,
        issue_key: str,
        target_dir: Path,
        *,
        filenames: Sequence[str] | None = None,
        attachment_ids: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> tuple[list[DownloadedFile], list[dict[str, str]]]:
        self._require_cloud("jira_download_attachments")
        attachments = await self.list_attachments(issue_key)
        selected, unmatched = self._select(attachments, filenames, attachment_ids, issue_key)

        downloaded: list[DownloadedFile] = []
        entries: list[_DownloadEntry] = list(unmatched)
        if not selected:
            # Nothing to write -- in particular, a selector that matched
            # nothing must not create an otherwise-empty target_dir.
            return downloaded, [entry.as_dict() for entry in entries]

        resolved_dir = target_dir.expanduser().resolve()
        try:
            resolved_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(
                self._redact(
                    f"[site={self._site.name}] jira_download_attachments: could not create "
                    f"target_dir '{resolved_dir}': {exc.__class__.__name__}: {exc}"
                )
            ) from exc

        used_names: set[str] = set()
        for attachment in selected:
            result = await self._download_one(
                attachment, resolved_dir, overwrite=overwrite, used_names=used_names
            )
            if isinstance(result, DownloadedFile):
                downloaded.append(result)
            else:
                entries.append(result)
        return downloaded, [entry.as_dict() for entry in entries]

    async def upload(self, issue_key: str, paths: Sequence[Path]) -> list[AttachmentMeta]:
        self._require_cloud("jira_upload_attachments")
        with ExitStack() as stack:
            try:
                handles = [stack.enter_context(open(path, "rb")) for path in paths]
            except OSError as exc:
                raise ToolError(
                    f"[site={self._site.name}] jira_upload_attachments: could not open "
                    f"'{exc.filename}': {exc.__class__.__name__}: {exc}"
                ) from exc
            files = [
                ("file", (path.name, fh, mimetypes.guess_type(path.name)[0] or "application/octet-stream"))
                for path, fh in zip(paths, handles, strict=True)
            ]
            path_str = f"/issue/{issue_key}/attachments"
            response = await self._http.post(
                f"{self.api_base}{path_str}",
                files=files,
                headers={"X-Atlassian-Token": "no-check"},
            )
        await self._raise_for_status(response, "POST", path_str)
        return [self._parse_attachment(raw) for raw in response.json()]

    def _select(
        self,
        attachments: Sequence[AttachmentMeta],
        filenames: Sequence[str] | None,
        attachment_ids: Sequence[str] | None,
        issue_key: str,
    ) -> tuple[list[AttachmentMeta], list[_DownloadEntry]]:
        if filenames is None and attachment_ids is None:
            return list(attachments), []
        filename_set = set(filenames) if filenames is not None else set()
        id_set = set(attachment_ids) if attachment_ids is not None else set()
        selected = [a for a in attachments if a.filename in filename_set or a.id in id_set]

        matched_filenames = {a.filename for a in selected}
        matched_ids = {a.id for a in selected}
        unmatched = [
            _DownloadEntry(name, f"no attachment with filename '{name}' on {issue_key}", "failed")
            for name in sorted(filename_set - matched_filenames)
        ] + [
            _DownloadEntry(att_id, f"no attachment with id '{att_id}' on {issue_key}", "failed")
            for att_id in sorted(id_set - matched_ids)
        ]
        return selected, unmatched

    def _pick_dest_name(
        self, safe: str, attachment_id: str, target_dir: Path, used_names: set[str], *, overwrite: bool
    ) -> str | None:
        """Chooses the basename this attachment writes to, or ``None`` if
        both it and its id-suffixed fallback are already taken.

        ``used_names`` (names already claimed by an earlier attachment in
        THIS SAME ``download()`` call) always forces the fallback name,
        regardless of ``overwrite`` -- ``overwrite`` is about a file that
        pre-dates this call, not about two attachments selected together
        that happen to share a name, which must never collide with each
        other either way.
        """
        if safe not in used_names and (overwrite or not (target_dir / safe).exists()):
            return safe
        stem, suffix = Path(safe).stem, Path(safe).suffix
        candidate = f"{stem}-{attachment_id}{suffix}"
        if candidate in used_names:
            return None
        if not overwrite and (target_dir / candidate).exists():
            return None
        return candidate

    async def _download_one(
        self, attachment: AttachmentMeta, target_dir: Path, *, overwrite: bool, used_names: set[str]
    ) -> DownloadedFile | _DownloadEntry:
        safe = _safe_filename(attachment.filename)
        if not safe:
            return _DownloadEntry(attachment.filename, "unsafe or empty filename", "failed")
        if not attachment.content_url:
            return _DownloadEntry(attachment.filename, "attachment has no content URL", "failed")

        dest_name = self._pick_dest_name(safe, attachment.id, target_dir, used_names, overwrite=overwrite)
        if dest_name is None:
            fallback_name = f"{Path(safe).stem}-{attachment.id}{Path(safe).suffix}"
            return _DownloadEntry(
                attachment.filename, f"both '{safe}' and '{fallback_name}' already exist", "skipped"
            )
        dest = target_dir / dest_name
        # Streamed into a per-attachment scratch name first, never straight
        # into `dest`: `overwrite=True` must not truncate an existing file
        # before the download is known to have fully succeeded, and the
        # attachment id keeps this unique across concurrent calls targeting
        # the same dest.
        part = target_dir / f"{dest.name}.{attachment.id}.part"

        try:
            # Cloud's attachment `content` URL 302s to a pre-signed media-CDN
            # URL on a different host; httpx drops the Authorization header
            # on that cross-host redirect. That's correct, not a bug -- the
            # redirect target is pre-signed and needs no auth of ours. Never
            # "fix" this into replaying our Basic auth cross-host, which
            # would leak it to whatever host Jira sends us to.
            async with self._http.stream("GET", attachment.content_url, follow_redirects=True) as response:
                try:
                    await self._raise_for_status(response, "GET", attachment.content_url)
                except ToolError as exc:
                    return _DownloadEntry(attachment.filename, str(exc), "failed")

                declared_size = self._parse_content_length(response.headers.get("content-length"))
                if declared_size is not None and declared_size > self._max_bytes:
                    return _DownloadEntry(
                        attachment.filename,
                        f"reported size {declared_size} bytes exceeds max_bytes {self._max_bytes}",
                        "failed",
                    )

                try:
                    handle = open(part, "wb")
                except OSError as exc:
                    return _DownloadEntry(
                        attachment.filename,
                        f"could not create temp file: {exc.__class__.__name__}: {exc}",
                        "failed",
                    )

                written = 0
                try:
                    async for chunk in response.aiter_bytes(65536):
                        written += len(chunk)
                        if written > self._max_bytes:
                            raise _AttachmentTooLarge
                        handle.write(chunk)
                except _AttachmentTooLarge:
                    handle.close()
                    part.unlink(missing_ok=True)
                    return _DownloadEntry(
                        attachment.filename,
                        f"exceeded max_bytes {self._max_bytes} while streaming",
                        "failed",
                    )
                except BaseException:
                    handle.close()
                    part.unlink(missing_ok=True)
                    raise
                else:
                    handle.close()
        except httpx.HTTPError as exc:
            # A transport-level failure (connect/read timeout, dropped
            # connection, ...), not a Jira-returned error status -- shape it
            # the same way rather than letting it escape as a raw httpx
            # exception. `dest` was never touched (only `part` may exist,
            # cleaned up by the streaming loop above or, if the failure was
            # before opening `part`, never created).
            part.unlink(missing_ok=True)
            return _DownloadEntry(attachment.filename, f"{exc.__class__.__name__}: {exc}", "failed")

        try:
            if overwrite:
                os.replace(part, dest)
            else:
                try:
                    os.link(part, dest)
                except FileExistsError:
                    return _DownloadEntry(attachment.filename, f"'{dest.name}' already exists", "skipped")
        finally:
            # `os.replace` already consumed `part`'s name; `os.link` leaves it
            # behind as a redundant second name to the same content -- either
            # way, cleaned up unconditionally (a no-op if already gone).
            part.unlink(missing_ok=True)

        used_names.add(dest.name)
        return DownloadedFile(
            attachment_id=attachment.id,
            filename=dest.name,
            original_filename=attachment.filename,
            path=str(dest),
            size=written,
            mime_type=attachment.mime_type,
        )

    @staticmethod
    def _parse_content_length(raw: str | None) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            # A malformed Content-Length is treated as absent rather than
            # crashing the download -- the streamed byte count still caps it.
            return None

    def _parse_attachment(self, raw: dict[str, Any]) -> AttachmentMeta:
        author = raw.get("author") or {}
        return AttachmentMeta(
            id=str(raw["id"]),
            filename=raw["filename"],
            size=int(raw.get("size", 0)),
            mime_type=raw.get("mimeType", "application/octet-stream"),
            created=raw.get("created", ""),
            author=author.get("displayName", ""),
            content_url=raw.get("content", ""),
        )

    async def _raise_for_status(self, response: httpx.Response, method: str, path: str) -> None:
        if response.is_success:
            return
        if not response.is_stream_consumed:
            await response.aread()
        detail = self._error_detail(response)
        message = self._redact(
            f"[site={self._site.name}] Jira returned {response.status_code} for {method} {path}: {detail}"
        )
        raise ToolError(message)

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.reason_phrase or str(response.status_code)
        if isinstance(body, dict):
            messages = [str(m) for m in (body.get("errorMessages") or [])]
            errors = body.get("errors")
            if isinstance(errors, dict):
                messages.extend(f"{key}: {value}" for key, value in errors.items())
            if messages:
                return "; ".join(messages)
        return response.reason_phrase or str(response.status_code)


class AttachmentClientRegistry:
    """One lazily-created ``httpx.AsyncClient`` (and ``JiraAttachmentClient``)
    per configured site, mirroring ``ChildManager``'s per-site lifecycle so
    ``server.py`` can close everything together on shutdown."""

    def __init__(
        self, sites: Sequence[SiteConfig], defaults: Defaults, *, redact: Callable[[str], str]
    ) -> None:
        self._sites = {site.name: site for site in sites}
        self._defaults = defaults
        self._redact = redact
        self._http_clients: dict[str, httpx.AsyncClient] = {}
        self._clients: dict[str, JiraAttachmentClient] = {}

    def get(self, site_name: str) -> JiraAttachmentClient:
        client = self._clients.get(site_name)
        if client is not None:
            return client
        site = self._sites[site_name]
        http = self._build_http_client(site)
        self._http_clients[site_name] = http
        client = JiraAttachmentClient(
            site, http, max_bytes=self._defaults.attachment_max_bytes, redact=self._redact
        )
        self._clients[site_name] = client
        return client

    def _build_http_client(self, site: SiteConfig) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            self._defaults.call_timeout_seconds, connect=self._defaults.connect_timeout_seconds
        )
        auth = (
            httpx.BasicAuth(site.username or "", site.api_token.get_secret_value())
            if site.api_token is not None
            else None
        )
        return httpx.AsyncClient(timeout=timeout, auth=auth)

    async def aclose(self) -> None:
        for site_name, http in self._http_clients.items():
            try:
                await http.aclose()
            except Exception as exc:  # noqa: BLE001 - one site's teardown must not block the rest
                _logger.warning(
                    "error closing attachment client for '%s': %s", site_name, self._redact(str(exc))
                )
