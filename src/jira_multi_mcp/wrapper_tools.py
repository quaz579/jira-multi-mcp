"""Tools this wrapper implements itself rather than mirroring from a child:
``jira_sites`` plus the three attachment tools (M3), which talk to Jira Cloud
REST v3 directly via ``attachments.AttachmentClientRegistry`` rather than
going through an upstream child -- see that module's docstring for why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mcp_types
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import Tool

from jira_multi_mcp.attachments import AttachmentClientRegistry
from jira_multi_mcp.children import ChildManager
from jira_multi_mcp.errors import SiteResolutionError
from jira_multi_mcp.registry import SiteRegistry, SiteResolution, resolve_site
from jira_multi_mcp.site_policy import enforce_site_policy


def _shape_entry(entry: dict[str, str]) -> dict[str, str]:
    """Passes a `_DownloadEntry.as_dict()` result through to the tool result,
    keeping only the keys it actually has: `filename` for an entry describing
    a real attachment, `selector` (e.g. `"id:999"`) for an unmatched selector
    that was never a real filename to begin with."""
    shaped = {"reason": entry["reason"]}
    if "filename" in entry:
        shaped["filename"] = entry["filename"]
    if "selector" in entry:
        shaped["selector"] = entry["selector"]
    return shaped


def _resolve_or_raise(
    registry: SiteRegistry, issue_key: str, site: str | None, tool_name: str
) -> SiteResolution:
    try:
        return resolve_site(registry, {"issue_key": issue_key}, explicit=site, tool_name=tool_name)
    except SiteResolutionError as exc:
        raise ToolError(str(exc)) from exc


def build_jira_sites_tool(manager: ChildManager) -> Tool:
    """Per-site health, never credentials: name, host, prefixes, read_only,
    state, error, last_error/last_error_at, log path, the FastMCP library
    version each child reports, and whether that site was the tool-discovery
    source. Also carries the ONE shared ``upstream_version`` (mcp-atlassian's
    own version, probed once at startup -- every site launches the same
    command) and, when no configured site is currently healthy, a top-level
    ``note`` explaining that no child could be reached."""

    async def jira_sites() -> dict[str, Any]:
        sites = manager.health()
        payload: dict[str, Any] = {
            "sites": sites,
            "upstream_version": manager.upstream_version(),
        }
        if not any(site["state"] == "healthy" for site in sites):
            payload["note"] = (
                "no configured site is currently healthy; tool discovery could not run "
                "against any child. See each site's 'error'/'log_path' above."
            )
        return payload

    return Tool.from_function(jira_sites, name="jira_sites")


def build_attachment_tools(
    registry: SiteRegistry, attachment_clients: AttachmentClientRegistry
) -> list[Tool]:
    """The three disk-based attachment tools. Each resolves its site the same
    way a mirrored tool does (explicit ``site`` argument, else inferred from
    ``issue_key``'s prefix) via the shared ``resolve_site``."""

    async def jira_list_attachments(issue_key: str, site: str | None = None) -> dict[str, Any]:
        resolution = _resolve_or_raise(registry, issue_key, site, "jira_list_attachments")
        enforce_site_policy(resolution.site, "jira_list_attachments", is_write=False, issue_key=issue_key)
        client = attachment_clients.get(resolution.site.name)
        attachments = await client.list_attachments(issue_key)
        return {
            "site": resolution.site.name,
            "issue_key": issue_key,
            "attachments": [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "size": a.size,
                    "mime_type": a.mime_type,
                    "created": a.created,
                    "author": a.author,
                }
                for a in attachments
            ],
        }

    async def jira_download_attachments(
        issue_key: str,
        target_dir: str,
        site: str | None = None,
        filenames: list[str] | None = None,
        attachment_ids: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        resolution = _resolve_or_raise(registry, issue_key, site, "jira_download_attachments")
        enforce_site_policy(resolution.site, "jira_download_attachments", is_write=False, issue_key=issue_key)
        client = attachment_clients.get(resolution.site.name)
        downloaded, entries = await client.download(
            issue_key,
            Path(target_dir),
            filenames=filenames,
            attachment_ids=attachment_ids,
            overwrite=overwrite,
        )
        return {
            "site": resolution.site.name,
            "issue_key": issue_key,
            "target_dir": str(Path(target_dir).expanduser().resolve()),
            "downloaded": [
                {
                    "attachment_id": d.attachment_id,
                    "filename": d.filename,
                    "original_filename": d.original_filename,
                    "path": d.path,
                    "size": d.size,
                    "mime_type": d.mime_type,
                }
                for d in downloaded
            ],
            "skipped": [_shape_entry(e) for e in entries if e["status"] == "skipped"],
            "failed": [_shape_entry(e) for e in entries if e["status"] == "failed"],
        }

    async def jira_upload_attachments(
        issue_key: str, paths: list[str], site: str | None = None
    ) -> dict[str, Any]:
        resolution = _resolve_or_raise(registry, issue_key, site, "jira_upload_attachments")
        enforce_site_policy(resolution.site, "jira_upload_attachments", is_write=True, issue_key=issue_key)
        resolved_paths = []
        for raw_path in paths:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_file():
                raise ToolError(
                    f"[site={resolution.site.name}] jira_upload_attachments: "
                    f"not an existing regular file: {raw_path}"
                )
            resolved_paths.append(candidate)
        client = attachment_clients.get(resolution.site.name)
        uploaded = await client.upload(issue_key, resolved_paths)
        return {
            "site": resolution.site.name,
            "issue_key": issue_key,
            "uploaded": [
                {"id": a.id, "filename": a.filename, "size": a.size, "mime_type": a.mime_type}
                for a in uploaded
            ],
        }

    return [
        Tool.from_function(
            jira_list_attachments,
            name="jira_list_attachments",
            description=(
                "Lists an issue's attachments: id, filename, size, mime_type, created, author. "
                "Read-only; does not fetch content -- use jira_download_attachments for that."
            ),
            annotations=mcp_types.ToolAnnotations(read_only_hint=True),
        ),
        Tool.from_function(
            jira_download_attachments,
            name="jira_download_attachments",
            description=(
                "Downloads one or more of an issue's attachments to disk under target_dir. "
                "Writes files to disk and returns their paths; then use your file-reading tool "
                "on the path. Prefer this over any base64 tool. Omit filenames/attachment_ids "
                "to download every attachment (an empty list for either is refused -- omit the "
                "argument instead). Jira Cloud sites only. Use an absolute target_dir "
                "-- a relative one resolves against the server process's own working directory, "
                "not yours (the resolved path is echoed back in the result either way)."
            ),
            # Not read-only despite reading from Jira: it writes to a
            # caller-supplied local path (target_dir), which is exactly the
            # kind of side effect readOnlyHint promises a tool doesn't have.
            # Not destructive (never removes/truncates something the caller
            # didn't ask it to write over) and not idempotent (overwrite=False,
            # the default, can produce a DIFFERENT result on a second call --
            # the id-suffixed fallback name -- once the first call's file
            # exists on disk).
            annotations=mcp_types.ToolAnnotations(
                read_only_hint=False, destructive_hint=False, idempotent_hint=False
            ),
        ),
        Tool.from_function(
            jira_upload_attachments,
            name="jira_upload_attachments",
            description=(
                "Uploads one or more local files as new attachments on an issue. Each path must "
                "already exist as a regular file; use absolute paths -- a relative one resolves "
                "against the server process's own working directory, not yours. Write operation: "
                "refused with a clear error on a site configured read_only = true."
            ),
            annotations=mcp_types.ToolAnnotations(read_only_hint=False),
        ),
    ]
