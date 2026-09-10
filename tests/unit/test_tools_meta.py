"""Sanity checks on the tool-name constants routing and config validation rely on."""

from __future__ import annotations

from jira_multi_mcp.tools_meta import (
    CURATED_TOOLS,
    ISSUE_KEY_ARGS,
    ISSUE_KEY_RE,
    NEVER_PARSED,
    PROJECT_KEY_ARGS,
    WRAPPER_OWNED_TOOLS,
)


def test_curated_tools_are_all_jira_prefixed() -> None:
    assert all(name.startswith("jira_") for name in CURATED_TOOLS)


def test_wrapper_owned_tools_are_a_subset_of_curated() -> None:
    assert WRAPPER_OWNED_TOOLS <= CURATED_TOOLS


def test_jql_is_never_a_routing_argument() -> None:
    assert "jql" not in ISSUE_KEY_ARGS
    assert "jql" not in PROJECT_KEY_ARGS
    assert NEVER_PARSED == frozenset({"jql"})


def test_issue_key_regex_examples() -> None:
    assert ISSUE_KEY_RE.match("ACME-123")
    assert ISSUE_KEY_RE.match("ACME-123-4")
    assert not ISSUE_KEY_RE.match("acme-123")
    assert not ISSUE_KEY_RE.match("123-ACME")
