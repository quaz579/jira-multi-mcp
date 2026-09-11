"""ChildManager: env construction, fail-soft startup, discovery source
selection, and health reporting -- all against in-process fake children."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path

import anyio
import pytest
from fastmcp.client.transports import ClientTransport, FastMCPTransport, StdioTransport
from fastmcp.exceptions import ToolError

from jira_multi_mcp.children import (
    BASE_ENV_PASSTHROUGH,
    EMPTY_ENABLED_TOOLS_SENTINEL,
    ChildManager,
    build_child_env,
    minimal_env,
)
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


def test_toolsets_is_always_all() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults())
    assert env["TOOLSETS"] == "all"


def test_enabled_tools_is_curated_minus_wrapper_owned_by_default() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="curated"))
    names = set(env["ENABLED_TOOLS"].split(","))
    assert names == CURATED_TOOLS - WRAPPER_OWNED_TOOLS
    assert "jira_download_attachments" not in names


def test_enabled_tools_excludes_all_wrapper_owned_attachment_names() -> None:
    site = _cloud_site("acme", "ACME")
    env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="curated"))
    names = set(env["ENABLED_TOOLS"].split(","))
    assert names.isdisjoint(WRAPPER_OWNED_TOOLS)
    assert "jira_list_attachments" not in names
    assert "jira_upload_attachments" not in names
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


def test_enabled_tools_of_only_wrapper_owned_names_forces_the_none_sentinel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A site whose enabled_tools is entirely WRAPPER_OWNED_TOOLS names (e.g.
    someone only wants the attachment tools on this site) would otherwise
    compute an empty ENABLED_TOOLS -- which upstream treats as "no filter",
    serving the child's FULL 63-tool set. The sentinel must be used instead."""
    site = _cloud_site(
        "acme", "ACME", enabled_tools=frozenset({"jira_download_attachments", "jira_list_attachments"})
    )
    with caplog.at_level("WARNING"):
        env = build_child_env(site, UpstreamConfig(), Defaults(toolset_preset="all"))
    assert env["ENABLED_TOOLS"] == EMPTY_ENABLED_TOOLS_SENTINEL
    assert any("enabled_tools" in record.message for record in caplog.records)


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


async def test_aclose_kills_a_real_child_process(tmp_path: Path) -> None:
    """Regression test for the M2r1 HIGH finding: with keep_alive=True,
    StdioTransport.connect_session's `finally` skips disconnect(), so merely
    unwinding an AsyncExitStack around the Client never asks the transport to
    kill the subprocess. ChildManager.aclose() must close every child's
    Client explicitly instead of relying on that unwind alone -- proved here
    against a REAL OS subprocess, not an in-process fake."""
    pid_file = tmp_path / "pid.txt"
    script = (
        "import os\n"
        f"with open({str(pid_file)!r}, 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('probe')\n"
        "mcp.run(transport='stdio', show_banner=False)\n"
    )
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(
        registry,
        tmp_path,
        lambda site, up: StdioTransport(command=sys.executable, args=["-c", script], keep_alive=True),
    )

    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=15)
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["state"] == "healthy"

        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # still alive; raises ProcessLookupError otherwise

        await manager.aclose()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await anyio.sleep(0.05)
        else:
            pytest.fail(f"child pid {pid} was still alive 5s after ChildManager.aclose()")


async def test_cancelling_a_stuck_handshake_still_kills_the_spawned_child(tmp_path: Path) -> None:
    """Characterization test, not a regression test: this passes whether or
    not `handle.client` is assigned before or after `enter_async_context`,
    because fastmcp's own `Client._connect()` has a `CancelledError` handler
    that closes the transport on a cancelled connect regardless (verified
    empirically). It's still worth pinning explicitly -- `server.py`'s
    cancel-before-close shutdown ordering (see its module docstring) depends
    on this exact behavior to unblock a child stuck mid-handshake, and this
    proves it against a REAL subprocess that never speaks MCP, so the
    handshake hangs until cancelled."""
    pid_file = tmp_path / "pid.txt"
    script = (
        "import os, time\n"
        f"with open({str(pid_file)!r}, 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(
        registry,
        tmp_path,
        lambda site, up: StdioTransport(command=sys.executable, args=["-c", script], keep_alive=True),
    )

    async with AsyncExitStack() as stack:
        async with anyio.create_task_group() as tg:
            tg.start_soon(manager.start_all, stack, 30.0)
            deadline = time.monotonic() + 5.0
            while not pid_file.exists() and time.monotonic() < deadline:
                await anyio.sleep(0.02)
            assert pid_file.exists(), "child never started"
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # still alive

            # Simulates a shutdown signal landing mid-connect.
            tg.cancel_scope.cancel()

        await manager.aclose()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await anyio.sleep(0.05)
        else:
            pytest.fail(f"child pid {pid} was still alive 5s after cancel+aclose (orphaned)")


async def test_probe_failure_after_connect_keeps_the_client_for_later_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same HIGH finding: a child whose liveness probe
    fails AFTER its session was entered must still be closeable later --
    losing the reference here is exactly what let such a child leak for the
    rest of the process's life."""
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))

    async def _boom(client: object) -> None:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(manager, "_probe_liveness", _boom)

    closed: list[str] = []
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        handle = manager._handles["acme"]  # noqa: SLF001 - whitebox on our own fake
        assert handle.state == "failed"
        assert handle.client is not None

        original_close = handle.client.close

        async def _spy_close() -> None:
            closed.append("acme")
            await original_close()  # type: ignore[no-untyped-call]

        monkeypatch.setattr(handle.client, "close", _spy_close)
        await manager.aclose()

    assert closed == ["acme"]


async def test_connect_failure_error_is_redacted(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME", api_token=Secret("zz-unique-secret-zz"))])

    def factory(site: SiteConfig, up: UpstreamConfig) -> ClientTransport:
        raise RuntimeError("auth failed with token zz-unique-secret-zz")

    manager = _make_manager(registry, tmp_path, factory)
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        health = {h["name"]: h for h in manager.health()}
        assert "zz-unique-secret-zz" not in str(health["acme"]["error"])
        assert "***" in str(health["acme"]["error"])


class _HangingTransport(ClientTransport):
    """Never yields a session -- stands in for a real connect that hangs past
    the deadline, so ``_start_one``'s cancel scope is the thing that actually
    fires (``scope.cancelled_caught``), not merely a ``TimeoutError`` raised
    for some unrelated reason."""

    @contextlib.asynccontextmanager
    async def connect_session(  # type: ignore[override]
        self, *, transport_options: object = None, **session_kwargs: object
    ) -> AsyncIterator[object]:
        await anyio.sleep_forever()
        yield None  # pragma: no cover - unreachable, connect_session never yields


async def test_a_real_hang_past_the_deadline_reports_our_own_timeout_message(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(registry, tmp_path, lambda site, up: _HangingTransport())
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=0.2)
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["error"] == "connect timed out after 0.2s"


async def test_a_bare_timeouterror_raised_by_the_factory_is_not_relabeled_as_our_deadline(
    tmp_path: Path,
) -> None:
    """A plain ``TimeoutError`` an underlying library raises for its own
    reason (e.g. a real OS-level ETIMEDOUT) is not the same thing as OUR
    connect deadline expiring -- it must go through the generic
    "failed to start" path, not be mislabeled "connect timed out after Xs"."""
    registry = SiteRegistry([_cloud_site("acme", "ACME")])

    def factory(site: SiteConfig, up: UpstreamConfig) -> ClientTransport:
        raise TimeoutError

    manager = _make_manager(registry, tmp_path, factory)
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        health = {h["name"]: h for h in manager.health()}
        # Not "connect timed out after 5s" (our own deadline message) -- this
        # exception was never near the deadline; it just isn't the thing that
        # message means.
        assert health["acme"]["error"] == "TimeoutError: "


async def test_mark_failed_redacts_the_reason_and_records_a_timestamp(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME", api_token=Secret("zz-unique-secret-zz"))])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)
        manager.mark_failed("acme", "connection closed near token zz-unique-secret-zz")
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["state"] == "failed"
        assert "zz-unique-secret-zz" not in str(health["acme"]["last_error"])
        assert "***" in str(health["acme"]["last_error"])
        assert health["acme"]["last_error_at"] is not None


async def test_mark_timeout_increments_and_flips_to_failed_after_three(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)

        manager.mark_timeout("acme", "jira_get_issue timed out after 1s")
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["timeouts"] == 1
        assert health["acme"]["state"] == "healthy"

        manager.mark_timeout("acme", "jira_get_issue timed out after 1s")
        manager.mark_timeout("acme", "jira_get_issue timed out after 1s")
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["timeouts"] == 3
        assert health["acme"]["state"] == "failed"


async def test_mark_success_resets_the_timeout_counter(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = _make_manager(registry, tmp_path, lambda site, up: FastMCPTransport(make_fake_child(site.name)))
    async with AsyncExitStack() as stack:
        await manager.start_all(stack, connect_timeout=5)

        manager.mark_timeout("acme", "timed out")
        manager.mark_timeout("acme", "timed out")
        manager.mark_success("acme")
        health = {h["name"]: h for h in manager.health()}
        assert health["acme"]["timeouts"] == 0
        assert health["acme"]["state"] == "healthy"


async def test_probe_upstream_version_captures_stdout(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    upstream = UpstreamConfig(command=(sys.executable, "-c", "import sys; print(sys.argv[-1])"))
    manager = ChildManager(registry, upstream, Defaults(), tmp_path)

    version = await manager.probe_upstream_version()

    assert version == "--version"
    assert manager.upstream_version() == "--version"


async def test_probe_upstream_version_stdin_is_devnull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream that doesn't recognize `--version` could start serving MCP
    on inherited stdin -- the live pipe this process itself uses to talk to
    Claude Code -- and eat its `initialize` request. A real-subprocess
    behavioral check is unreliable here (pytest's own default capturing
    already redirects fd 0 for any child, regardless of this code's own
    ``stdin=`` kwarg), so this asserts the kwarg directly; also proved end to
    end in the adversarial-loop real run."""
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    manager = ChildManager(registry, UpstreamConfig(), Defaults(), tmp_path)
    captured_stdin: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_stdin.append(kwargs.get("stdin"))
        return subprocess.CompletedProcess(command, 0, stdout="1.0.0", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    version = await manager.probe_upstream_version(timeout=5)

    assert version == "1.0.0"
    assert captured_stdin == [subprocess.DEVNULL]


async def test_probe_upstream_version_handles_a_missing_command(tmp_path: Path) -> None:
    registry = SiteRegistry([_cloud_site("acme", "ACME")])
    upstream = UpstreamConfig(command=(str(tmp_path / "does-not-exist"),))
    manager = ChildManager(registry, upstream, Defaults(), tmp_path)

    version = await manager.probe_upstream_version()

    assert version is None
    assert manager.upstream_version() is None
