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
    state, error, log path, upstream version, and whether that site was the
    tool-discovery source."""

    async def jira_sites() -> list[dict[str, Any]]:
        return manager.health()

    return Tool.from_function(jira_sites, name="jira_sites")
