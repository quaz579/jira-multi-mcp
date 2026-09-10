"""Metadata about upstream mcp-atlassian's Jira tools used for site routing.

Tool names verified against sooperset/mcp-atlassian at tag ``v0.23.1``
(``src/mcp_atlassian/servers/jira.py``, mounted under the ``jira`` namespace
in ``servers/main.py`` — every tool's wire name is ``jira_<function_name>``).
"""

from __future__ import annotations

import re

# Arguments that may carry one or more issue keys (str, comma-separated str,
# or list). Never includes "jql": free-text JQL is never parsed for a site.
ISSUE_KEY_ARGS: tuple[str, ...] = (
    "issue_key",
    "issue_keys",
    "epic_key",
    "parent",
    "inward_issue_key",
    "outward_issue_key",
    "issue_ids_or_keys",
)

# Arguments that carry a bare project key (no issue number suffix).
PROJECT_KEY_ARGS: tuple[str, ...] = (
    "project_key",
    "target_project_key",
)

# jira_search's own project-scoping argument (comma-separated). It may mix
# real project keys with numeric project ids; only tokens that look like a
# project key are used for routing (see PROJECT_KEY_RE) — a numeric id is
# silently skipped rather than treated as an unknown prefix, since it never
# carries one.
PROJECTS_FILTER_ARGS: tuple[str, ...] = ("projects_filter",)

# Arguments that must never be inspected for site-routing purposes, even
# though they can contain text that looks like an issue key.
NEVER_PARSED: frozenset[str] = frozenset({"jql"})

ISSUE_KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]+)-\d+(?:-\d+)*$")

# Shape of a bare project key (also used to validate a configured key_prefix).
PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")

_ROUTABLE_ARGS = frozenset(ISSUE_KEY_ARGS) | frozenset(PROJECT_KEY_ARGS) | frozenset(PROJECTS_FILTER_ARGS)
assert not (NEVER_PARSED & _ROUTABLE_ARGS), "NEVER_PARSED must never overlap a routable argument name"

# The curated subset of upstream's Jira tools this server mirrors by default
# (toolset_preset = "curated"). Excludes agile board/sprint tools (out of
# scope for v1) and anything not read/write core issue, search, comment,
# transition, user, link, worklog, or basic project functionality.
CURATED_TOOLS: frozenset[str] = frozenset(
    {
        # issues
        "jira_get_issue",
        "jira_create_issue",
        "jira_batch_create_issues",
        "jira_batch_get_changelogs",
        "jira_update_issue",
        "jira_assign_issue",
        "jira_delete_issue",
        "jira_move_issue",
        # search / fields
        "jira_search",
        "jira_search_fields",
        "jira_get_field_options",
        "jira_get_create_fields",
        "jira_get_project_fields",
        # comments
        "jira_add_comment",
        "jira_edit_comment",
        # transitions
        "jira_get_transitions",
        "jira_transition_issue",
        # attachments (image inline content; disk download is wrapper-owned)
        "jira_get_issue_images",
        "jira_download_attachments",
        # users
        "jira_get_user_profile",
        "jira_search_assignable_users",
        "jira_get_issue_watchers",
        "jira_add_watcher",
        "jira_remove_watcher",
        # links
        "jira_get_link_types",
        "jira_create_issue_link",
        "jira_create_remote_issue_link",
        "jira_remove_issue_link",
        "jira_link_to_epic",
        # worklog
        "jira_get_worklog",
        "jira_add_worklog",
        # projects (basics)
        "jira_get_project_issues",
        "jira_get_project_issue_types",
        "jira_get_project_versions",
        "jira_get_project_components",
        "jira_get_all_projects",
        "jira_search_projects",
    }
)

# Tools this wrapper implements itself (in wrapper_tools.py, M3) instead of
# exposing the upstream child's version. The upstream "jira_download_attachments"
# returns base64 in-band; ours writes to disk. Mirror logic (M2) must exclude
# these names from ENABLED_TOOLS on children and never forward calls to them.
WRAPPER_OWNED_TOOLS: frozenset[str] = frozenset({"jira_download_attachments"})
