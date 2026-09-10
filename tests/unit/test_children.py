"""ChildManager: env construction, fail-soft startup, discovery source
selection, and health reporting -- all against in-process fake children."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from fastmcp.client.transports import ClientTransport, FastMCPTransport
from fastmcp.exceptions import ToolError

from jira_multi_mcp.children import BASE_ENV_PASSTHROUGH, ChildManager, build_child_env, minimal_env
from jira_multi_mcp.model import Defaults, SiteConfig, UpstreamConfig
from jira_multi_mcp.registry import SiteRegistry
from jira_multi_mcp.secrets import Secret
from jira_multi_mcp.tools_meta import CURATED_TOOLS, WRAPPER_OWNED_TOOLS
from tests.fakes.fake_child import make_fake_child


def _cloud_site(name: str, *prefixes: str, **overrides: object) -> SiteConfig:
    fields: dict[str, object] = {
        "name": name,
        "url": f"https://{name}.atlassian.net",
        "key_prefixes": prefixes,
        "username": "bgrossman@jumpmind.com",
        "api_token": Secret("token-value"),
    }
    fields.update(overrides)
    return SiteConfig(**fields)  # type: ignore[arg-type]


# --- build_child_env ---


def test_toolsets_is_always_all(monkeypatch: pytest.MonkeyPatch) -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert env["TOOLSETS"] == "all"


def test_enabled_tools_is_curated_minus_wrapper_owned_by_default() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="curated"))
    names = set(env["ENABLED_TOOLS"].split(","))
    assert names == CURATED_TOOLS - WRAPPER_OWNED_TOOLS
    assert "jira_download_attachments" not in names


def test_enabled_tools_omitted_for_all_preset_with_no_site_override() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="all"))
    assert "ENABLED_TOOLS" not in env


def test_site_level_enabled_tools_override_wins_even_under_all_preset() -> None:
    site = _cloud_site(
        "acme", "ACME", enabled_tools=frozenset({"jira_get_issue", "jira_download_attachments"})
    )
    env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="all"))
    assert env["ENABLED_TOOLS"] == "jira_get_issue"


def test_read_only_mode_set_only_when_site_is_read_only() -> None:
    read_only_env = build_child_env(_cloud_site("acme", "ACME", read_only=True), UpstreamConfig(), Defaults())
    normal_env = build_child_env(_cloud_site("beta", "BETA"), UpstreamConfig(), Defaults())
    assert read_only_env["READ_ONLY_MODE"] == "true"
    assert "READ_ONLY_MODE" not in normal_env


def test_projects_filter_passed_through_when_configured() -> None:
    site = _cloud_site("acme", "ACME", projects_filter=("FOO", "BAR"))
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert env["JIRA_PROJECTS_FILTER"] == "FOO,BAR"


def test_cloud_auth_env_vars() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert env["JIRA_URL"] == "https://acme.atlassian.net"
    assert env["JIRA_USERNAME"] == "bgrossman@jumpmind.com"
    assert env["JIRA_API_TOKEN"] == "token-value"
    assert "JIRA_PERSONAL_TOKEN" not in env


def test_server_dc_auth_env_vars() -> None:
    site = SiteConfig(
        name="onprem",
        url="https://jira.example.com",
        key_prefixes=("ONPREM",),
        personal_token=Secret("pat-value"),
    )
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert env["JIRA_PERSONAL_TOKEN"] == "pat-value"
    assert "JIRA_API_TOKEN" not in env
    assert "JIRA_USERNAME" not in env


def test_minimal_env_passes_through_only_configured_and_base_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/certs.pem")
    monkeypatch.setenv("SOME_UNRELATED_SECRET", "should-not-appear")
    env = minimal_env(("SSL_CERT_FILE",))
    assert env["SSL_CERT_FILE"] == "/etc/ssl/certs.pem"
    assert "SOME_UNRELATED_SECRET" not in env
    assert set(BASE_ENV_PASSTHROUGH) <= {"PATH", "HOME"}


def test_build_child_env_never_leaks_unrelated_ambient_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_OTHER_APPS_SECRET", "leaked")
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert "SOME_OTHER_APPS_SECRET" not in env


# --- ChildManager against fake children ---


def _make_manager(registry: SiteRegistry, tmp_path: Path, transport_factory: object) -> ChildManager:
    return ChildManager(
        registry,
        UpstreamConfig(),
        Defaults(),
        tmp_path,
        transport_factory=transport_factory,  # type: ignore[arg-type]
    )


async def test_discovery_skips_a_read_only_first_site(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME", read_only=True), _cloud_site("beta", "BETA")])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        await manager.discover_tools()
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["discovery_source"] is False
        assert health["beta"]["discovery_source"] is True


async def test_one_failing_site_leaves_the_other_healthy(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME"), _cloud_site("beta", "BETA")])

    def factory(site: SiteConfig, up: UpstreamConfig) -> ClientTransport:
        if site.name == "acme":
            raise RuntimeError("simulated connect failure")
        return FastMCPTransport(make_fake_child(site.name))

    manager = _make_manager(registry, tmp_path, factory)
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["state"] == "failed"
        assert "simulated connect failure" in str(health["acme"]["error"])
        assert health["beta"]["state"] == "healthy"


async def test_call_to_failed_site_names_it_in_the_tool_error(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])

    def factory(site: SiteConfig, up: UpstreamConfig) -> ClientTransport:
        raise RuntimeError("simulated connect failure")

    manager = _make_manager(registry, tmp_path, factory)
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        with pytest.raises(ToolError, match="acme"):
            await manager.client_for("acme")


async def test_all_sites_failed_means_no_tools_discovered(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME"), _cloud_site("beta", "BETA")])

    def factory(site: SiteConfig, up: UpstreamConfig) -> ClientTransport:
        raise RuntimeError("simulated connect failure")

    manager = _make_manager(registry, tmp_path, factory)
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        tools = await manager.discover_tools()
        assert tools == []
        assert all(h["state"] == "failed" for h in manager.health())


async def test_aclose_does_not_raise(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        await manager.aclose()
