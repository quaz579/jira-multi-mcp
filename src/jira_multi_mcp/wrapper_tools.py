"""Tools this wrapper implements itself rather than mirroring from a child.

Only ``jira_sites`` exists in M2. Attachment tools (``jira_list_attachments``,
``jira_download_attachments``, ``jira_upload_attachments``) are M3 — the
``WRAPPER_OWNED_TOOLS`` allowlist in ``tools_meta`` already reserves
``jira_download_attachments`` so the mirror shadows the upstream (base64)
version starting now, ahead of the real disk-writing implementation landing.
"""

from __future__ import annotations

from typing import Any

from fastmcp.tools.base import Tool

from jira_multi_mcp.children import ChildManager


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
