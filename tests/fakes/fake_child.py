"""An in-process FastMCP server standing in for one upstream mcp-atlassian
child: echoes back what it received (plus its own site name) so a test can
prove `site` was stripped before forwarding, and see which child answered."""

from __future__ import annotations

from typing import Any

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp_types import ToolAnnotations

_READ_ONLY = ToolAnnotations(read_only_hint=True)
_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=True)


def make_fake_child(site_name: str) -> FastMCP:
    mcp: FastMCP = FastMCP(f"fake-{site_name}")

    @mcp.tool(annotations=_READ_ONLY)
    def jira_get_issue(issue_key: str) -> dict[str, Any]:
        return {"site": site_name, "issue_key": issue_key}

    @mcp.tool(annotations=_READ_ONLY)
    def jira_search(jql: str) -> dict[str, Any]:
        return {"site": site_name, "jql": jql}

    @mcp.tool(annotations=_WRITE)
    def jira_create_issue_link(
        inward_issue_key: str, outward_issue_key: str, link_type: str
    ) -> dict[str, Any]:
        return {
            "site": site_name,
            "inward_issue_key": inward_issue_key,
            "outward_issue_key": outward_issue_key,
        }

    @mcp.tool
    def jira_boom() -> str:
        raise ToolError("boom")

    @mcp.tool
    async def jira_slow() -> str:
        await anyio.sleep(3600)
        return "never"

    # Upstream's base64-in-band tool. Present here purely so a test can prove
    # the parent's tool list never includes it (WRAPPER_OWNED_TOOLS shadows
    # it in favor of the wrapper's own disk-writing version, M3).
    @mcp.tool(annotations=_READ_ONLY)
    def jira_download_attachments(issue_key: str) -> dict[str, Any]:
        return {"site": site_name, "issue_key": issue_key, "base64": "not-really"}

    @mcp.tool(tags={"write"}, annotations=_WRITE)
    def jira_delete_issue(issue_key: str) -> dict[str, Any]:
        return {"site": site_name, "issue_key": issue_key, "deleted": True}

    # Always fails, tagged as a write -- stands in for upstream actually
    # enforcing READ_ONLY_MODE (which this fake doesn't simulate), so a test
    # can exercise the mirror's "configured read_only = true" hint text.
    @mcp.tool(tags={"write"}, annotations=_WRITE)
    def jira_write_blocked(issue_key: str) -> str:
        raise ToolError("write not permitted")

    return mcp
