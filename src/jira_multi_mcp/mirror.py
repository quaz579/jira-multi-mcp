"""Builds the ONE mirrored tool set: each upstream tool gets an optional
``site`` parameter and forwards its call to the right child."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import anyio
import mcp_types
from fastmcp.exceptions import ToolError
from fastmcp.tools.base import Tool, ToolResult
from mcp import MCPError
from pydantic import PrivateAttr

from jira_multi_mcp.children import ChildManager
from jira_multi_mcp.errors import SchemaConflictError, SiteResolutionError
from jira_multi_mcp.registry import SiteRegistry, resolve_site
from jira_multi_mcp.tools_meta import WRAPPER_OWNED_TOOLS

_logger = logging.getLogger(__name__)

SITE_PARAM = "site"

_CONNECTION_ERROR_TYPES = (anyio.ClosedResourceError, anyio.BrokenResourceError)


def _is_connection_failure(exc: BaseException) -> bool:
    """Whether ``exc`` looks like the child's pipe/session breaking mid-call
    (as opposed to a plain application error), the trigger for marking a site
    ``failed`` in ``jira_sites`` -- restart is out of scope (M4), but a dead
    site should stop being reported ``healthy``."""
    if isinstance(exc, _CONNECTION_ERROR_TYPES):
        return True
    if isinstance(exc, MCPError):
        text = str(exc).lower()
        return "closed" in text or "connection" in text
    return False


def augment_input_schema(schema: Mapping[str, Any], registry: SiteRegistry) -> dict[str, Any]:
    """Adds an optional ``site`` property to an upstream tool's JSON schema.

    Never touches ``required``: a tool call omitting ``site`` is always valid
    (the resolver infers it from an issue/project key, or raises a clear
    error naming every prefix). Raises if upstream already defines a ``site``
    property — silently overwriting it would surprise nobody more than us.
    """
    augmented = copy.deepcopy(dict(schema))
    augmented.setdefault("type", "object")
    properties = augmented.setdefault("properties", {})
    if SITE_PARAM in properties:
        raise SchemaConflictError(
            f"upstream tool schema already defines a '{SITE_PARAM}' property; refusing to mirror it"
        )
    properties[SITE_PARAM] = {
        "type": "string",
        "enum": [site.name for site in registry.sites],
        "description": (
            "Jira site to target. Omit to infer it from the issue key prefix. "
            f"Prefixes — {registry.prefix_table()}. "
            "Required when no issue key is in the arguments (for example jira_search)."
        ),
    }
    return augmented


def augment_description(description: str | None, registry: SiteRegistry) -> str:
    suffix = f"Configured Jira sites — {registry.prefix_table()}."
    if not description:
        return suffix
    return f"{description}\n\n{suffix}"


class MultiSiteProxyTool(Tool):
    """A mirrored tool: resolves ``site`` client-side, then forwards the call
    to that site's child via ``call_tool_mcp`` (never through the child's own
    routing — the child has no idea other sites exist)."""

    KEY_PREFIX: ClassVar[str] = "tool"

    _manager: ChildManager = PrivateAttr()
    _registry: SiteRegistry = PrivateAttr()

    @classmethod
    def from_mcp_tool(
        cls,
        manager: ChildManager,
        registry: SiteRegistry,
        mcp_tool: mcp_types.Tool,
        timeout: float,
    ) -> MultiSiteProxyTool:
        tags: set[str] = set()
        if mcp_tool.meta:
            fastmcp_meta = mcp_tool.meta.get("fastmcp")
            if isinstance(fastmcp_meta, dict):
                raw_tags = fastmcp_meta.get("tags")
                if isinstance(raw_tags, list):
                    tags = {str(tag) for tag in raw_tags}

        instance = cls(
            name=mcp_tool.name,
            title=mcp_tool.title,
            description=augment_description(mcp_tool.description, registry),
            parameters=augment_input_schema(mcp_tool.input_schema, registry),
            output_schema=mcp_tool.output_schema,
            annotations=mcp_tool.annotations,
            tags=tags,
            timeout=timeout,
        )
        instance._manager = manager
        instance._registry = registry
        return instance

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        args = dict(arguments)
        explicit_site = args.pop(SITE_PARAM, None)
        try:
            resolution = resolve_site(self._registry, args, explicit=explicit_site, tool_name=self.name)
        except SiteResolutionError as exc:
            raise ToolError(str(exc)) from exc

        site = resolution.site
        client = await self._manager.client_for(site.name)

        assert self.timeout is not None, f"mirrored tool '{self.name}' was built without a timeout"
        log_path = self._manager.log_path(site.name)
        try:
            with anyio.fail_after(self.timeout):
                result = await client.call_tool_mcp(self.name, args)
        except TimeoutError as exc:
            raise ToolError(
                f"[site={site.name}] {self.name} timed out after {self.timeout}s. Child log: {log_path}"
            ) from exc
        except Exception as exc:
            # The child died or the connection otherwise broke mid-call (not a
            # timeout, not a tool-level error the child reported normally) --
            # give it the same [site=] shape and a pointer to its log instead
            # of letting a raw MCPError/connection exception escape unshaped.
            reason = f"{exc.__class__.__name__}: {exc}"
            if _is_connection_failure(exc):
                self._manager.mark_failed(site.name, reason)
            redacted = self._manager.redact(reason)
            raise ToolError(
                f"[site={site.name}] {self.name} failed: {redacted}. Child log: {log_path}"
            ) from exc

        if result.is_error:
            text = "\n".join(
                block.text for block in result.content if isinstance(block, mcp_types.TextContent)
            )
            hint = ""
            if site.read_only and self.annotations is not None and self.annotations.read_only_hint is False:
                hint = f" (site '{site.name}' is configured read_only = true)"
            raise ToolError(f"[site={site.name}] {text}{hint}")

        return ToolResult(
            content=result.content,
            structured_content=result.structured_content,
            meta={
                **(result.meta or {}),
                "jira-multi-mcp/site": site.name,
                "jira-multi-mcp/site_reason": resolution.reason,
            },
        )


def build_mirrored_tools(
    mcp_tools: Sequence[mcp_types.Tool],
    manager: ChildManager,
    registry: SiteRegistry,
    allowlist: frozenset[str] | None,
    timeout: float,
) -> list[MultiSiteProxyTool]:
    """One ``MultiSiteProxyTool`` per discovered tool that isn't wrapper-owned
    and (when ``allowlist`` is given) is in it. ``allowlist=None`` mirrors
    every discovered tool (the "all" toolset preset)."""
    mirrored = []
    discovered_names = {tool.name for tool in mcp_tools}
    for mcp_tool in mcp_tools:
        if mcp_tool.name in WRAPPER_OWNED_TOOLS:
            continue
        if allowlist is not None and mcp_tool.name not in allowlist:
            continue
        mirrored.append(MultiSiteProxyTool.from_mcp_tool(manager, registry, mcp_tool, timeout))

    if allowlist is not None:
        for name in sorted((allowlist - WRAPPER_OWNED_TOOLS) - discovered_names):
            _logger.warning("tool '%s' is allowlisted but no healthy child advertised it", name)

    return mirrored
