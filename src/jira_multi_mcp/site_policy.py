"""Shared per-site policy enforcement for every wrapper-owned tool.

A mirrored (child) tool gets ``read_only``/``enabled_tools`` enforcement for
free from the upstream child process itself -- ``READ_ONLY_MODE`` and
``ENABLED_TOOLS`` are env vars baked into that child at launch (see
``children.build_child_env``). A wrapper-owned tool (the three attachment
tools; anything added later) talks to Jira directly and never passes through
a child, so without this it would silently bypass every per-site policy.
One guard, used by every wrapper tool, rather than three (or more) separate
checks that could drift out of sync.
"""

from __future__ import annotations

from fastmcp.exceptions import ToolError

from jira_multi_mcp.model import SiteConfig
from jira_multi_mcp.tools_meta import ISSUE_KEY_RE


def enforce_site_policy(site: SiteConfig, tool_name: str, *, is_write: bool, issue_key: str | None) -> None:
    """Raises ``ToolError`` if ``site``'s configuration forbids this call.

    Checked in order -- whether the site serves this tool at all, then
    whether this specific call is a write against a read-only site, then
    whether the target issue's project is in scope -- so a caller always
    sees the most fundamental reason first rather than a generic refusal.
    """
    if site.enabled_tools is not None and tool_name not in site.enabled_tools:
        raise ToolError(f"[site={site.name}] {tool_name}: tool is not in this site's enabled_tools")

    if is_write and site.read_only:
        raise ToolError(f"[site={site.name}] {tool_name}: site is configured read_only = true")

    if site.projects_filter is not None and issue_key is not None:
        match = ISSUE_KEY_RE.match(issue_key.strip().upper())
        if match:
            project = match.group(1)
            if project not in site.projects_filter:
                raise ToolError(
                    f"[site={site.name}] {tool_name}: project '{project}' (from issue "
                    f"'{issue_key}') is not in this site's projects_filter "
                    f"({', '.join(site.projects_filter)})"
                )
