"""Site resolution: every routing edge case from the design doc's algorithm."""

from __future__ import annotations

import pytest

from jira_multi_mcp.errors import AmbiguousSiteError, CrossSiteError, UnknownPrefixError, UnknownSiteError
from jira_multi_mcp.model import SiteConfig
from jira_multi_mcp.registry import SiteRegistry, resolve_site
from jira_multi_mcp.secrets import Secret


def _site(name: str, *prefixes: str) -> SiteConfig:
    return SiteConfig(
        name=name,
        url=f"https://{name}.atlassian.net",
        key_prefixes=prefixes,
        username="bgrossman@jumpmind.com",
        api_token=Secret("token"),
    )


@pytest.fixture
def multi_site_registry() -> SiteRegistry:
    return SiteRegistry(
        [
            _site("acme", "ACME", "ACMEOPS"),
            _site("beta", "BETA"),
            _site("jm", "JM"),
            _site("jmc", "JMC"),
        ]
    )


@pytest.fixture
def single_site_registry() -> SiteRegistry:
    return SiteRegistry([_site("acme", "ACME")])


def test_lowercase_key_still_resolves(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {"issue_key": "acme-123"}, tool_name="jira_get_issue")
    assert result.site.name == "acme"
    assert "ACME-123" in result.reason


def test_comma_separated_string(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(
        multi_site_registry, {"issue_keys": "ACME-1,ACME-2"}, tool_name="jira_batch_get_changelogs"
    )
    assert result.site.name == "acme"


def test_list_of_keys(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {"issue_ids_or_keys": ["ACME-1", "ACME-2"]}, tool_name="x")
    assert result.site.name == "acme"


def test_multi_segment_issue_key(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {"issue_key": "ACME-123-4"}, tool_name="x")
    assert result.site.name == "acme"


def test_prefix_exact_match_not_startswith(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {"issue_key": "JM-1"}, tool_name="x")
    assert result.site.name == "jm"
    result = resolve_site(multi_site_registry, {"issue_key": "JMC-1"}, tool_name="x")
    assert result.site.name == "jmc"


def test_empty_and_none_values_are_skipped(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(AmbiguousSiteError):
        resolve_site(multi_site_registry, {"issue_key": None, "epic_key": "", "parent": "  "}, tool_name="x")


def test_jql_is_never_parsed(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(AmbiguousSiteError):
        resolve_site(multi_site_registry, {"jql": "project = ACME"}, tool_name="jira_search")


def test_url_argument_is_never_parsed(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(AmbiguousSiteError):
        resolve_site(multi_site_registry, {"url": "https://acme.atlassian.net/browse/ACME-1"}, tool_name="x")


def test_project_key_and_contradicting_issue_key_is_cross_site(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(CrossSiteError) as exc_info:
        resolve_site(multi_site_registry, {"project_key": "BETA", "issue_key": "ACME-1"}, tool_name="x")
    message = str(exc_info.value)
    assert "acme" in message
    assert "beta" in message


def test_explicit_wins_over_contradicting_key(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {"issue_key": "ACME-1"}, explicit="beta", tool_name="x")
    assert result.site.name == "beta"
    assert result.reason == "explicit"


def test_explicit_is_case_insensitive(multi_site_registry: SiteRegistry) -> None:
    result = resolve_site(multi_site_registry, {}, explicit="ACME", tool_name="x")
    assert result.site.name == "acme"


def test_unknown_explicit_site_lists_configured_names(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(UnknownSiteError) as exc_info:
        resolve_site(multi_site_registry, {}, explicit="nope", tool_name="x")
    message = str(exc_info.value)
    assert "acme" in message and "beta" in message


def test_single_site_shortcut_ignores_arguments(single_site_registry: SiteRegistry) -> None:
    result = resolve_site(single_site_registry, {"issue_key": "does-not-matter"}, tool_name="x")
    assert result.site.name == "acme"
    assert result.reason == "only configured site"


def test_unknown_prefix_names_it(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(UnknownPrefixError, match="ZZZ"):
        resolve_site(multi_site_registry, {"issue_key": "ZZZ-1"}, tool_name="x")


def test_cross_site_names_both_keys_and_sites(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(CrossSiteError) as exc_info:
        resolve_site(
            multi_site_registry, {"issue_key": "ACME-1", "epic_key": "BETA-2"}, tool_name="jira_get_issue"
        )
    message = str(exc_info.value)
    assert "ACME-1" in message
    assert "BETA-2" in message
    assert "acme" in message
    assert "beta" in message


def test_zero_keys_is_ambiguous_with_prefix_table(multi_site_registry: SiteRegistry) -> None:
    with pytest.raises(AmbiguousSiteError) as exc_info:
        resolve_site(multi_site_registry, {}, tool_name="jira_search")
    message = str(exc_info.value)
    assert "acme: ACME, ACMEOPS" in message
    assert "beta: BETA" in message


def test_prefix_table_format(multi_site_registry: SiteRegistry) -> None:
    assert multi_site_registry.prefix_table() == "acme: ACME, ACMEOPS | beta: BETA | jm: JM | jmc: JMC"
