"""Real-subprocess proof of the M2r2 MEDIUM finding: the shutdown signal
receiver must be armed BEFORE `start_all`, not after -- otherwise a SIGTERM
landing during the connect window hits the OS default disposition (the
process dies with no `finally` ever running) and every already-spawned child
is orphaned. Also covers the watchdog's exit-code fidelity (M2r2 item 5)."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jira_multi_mcp.server import SHUTDOWN_WATCHDOG_SECONDS

# The watchdog's own bound plus slack for interpreter/process-spawn overhead
# on a loaded CI runner -- NOT a re-assertion of the exact round-2 timings
# (parent 5.08s / children 0.32s), which the real run measures precisely.
_MAX_SHUTDOWN_SECONDS = SHUTDOWN_WATCHDOG_SECONDS + 5.0


def _write_hanging_upstream_script(tmp_path: Path) -> Path:
    """A fake upstream that spawns (proving `start_all` got as far as
    launching the real OS process) but never speaks MCP -- the connect
    handshake hangs until the site is torn down or the process is killed."""
    script = tmp_path / "hanging_upstream.py"
    script.write_text(
        "import os, sys, time\n"
        "if '--version' not in sys.argv:\n"
        "    with open(sys.argv[1], 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    return script


def _write_config(tmp_path: Path, *, command: list[str]) -> Path:
    config_path = tmp_path / "config.toml"
    command_toml = ", ".join(repr(part) for part in command)
    config_path.write_text(
        f"""
        [defaults]
        username = "bgrossman@jumpmind.com"
        api_token = "test-token"
        connect_timeout_seconds = 30
        call_timeout_seconds = 5

        [upstream]
        command = [{command_toml}]

        [[sites]]
        name = "acme"
        url = "https://acme.atlassian.net"
        key_prefixes = ["ACME"]
        """
    )
    return config_path


def _wait_for_file(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            pytest.fail(f"{path} never appeared within {timeout}s")
        time.sleep(0.02)


def _pid_alive(pid: int) -> bool:
    try:
        __import__("os").kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_sigterm_during_startup_does_not_orphan_the_spawned_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    upstream_script = _write_hanging_upstream_script(tmp_path)
    config_path = _write_config(tmp_path, command=[sys.executable, str(upstream_script), str(pid_file)])

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from jira_multi_mcp.cli import main; sys.exit(main())",
            "--config",
            str(config_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(pid_file, timeout=15.0)
        child_pid = int(pid_file.read_text().strip())
        assert _pid_alive(child_pid), "fake upstream child never actually started"

        t_signal = time.monotonic()
        proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=_MAX_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(
                f"parent did not exit within {_MAX_SHUTDOWN_SECONDS}s of SIGTERM "
                "landing during the connect window (armed-before-start_all regression)"
            )
        parent_elapsed = time.monotonic() - t_signal

        deadline = time.monotonic() + 5.0
        while _pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert not _pid_alive(child_pid), (
            f"fake upstream child pid {child_pid} was orphaned: still alive after the "
            "parent exited following a SIGTERM landing mid-connect"
        )
        assert parent_elapsed <= _MAX_SHUTDOWN_SECONDS
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_stdin_eof_during_startup_does_not_orphan_the_spawned_child(tmp_path: Path) -> None:
    """The client-abandons-mid-connect scenario, but via stdin closing
    (no bytes ever written) rather than a signal -- proves the stdin-EOF
    watcher, not just the SIGTERM path, unblocks a stuck connect and kills
    the spawned child."""
    pid_file = tmp_path / "child.pid"
    upstream_script = _write_hanging_upstream_script(tmp_path)
    config_path = _write_config(tmp_path, command=[sys.executable, str(upstream_script), str(pid_file)])

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from jira_multi_mcp.cli import main; sys.exit(main())",
            "--config",
            str(config_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(pid_file, timeout=15.0)
        child_pid = int(pid_file.read_text().strip())
        assert _pid_alive(child_pid), "fake upstream child never actually started"

        time.sleep(0.5)
        t_close = time.monotonic()
        assert proc.stdin is not None
        proc.stdin.close()

        try:
            proc.wait(timeout=_MAX_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(
                f"parent did not exit within {_MAX_SHUTDOWN_SECONDS}s of stdin closing "
                "during the connect window (stdin-EOF watcher regression)"
            )
        parent_elapsed = time.monotonic() - t_close

        deadline = time.monotonic() + 5.0
        while _pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert not _pid_alive(child_pid), (
            f"fake upstream child pid {child_pid} was orphaned: still alive after the "
            "parent exited following stdin closing mid-connect"
        )
        assert parent_elapsed <= _MAX_SHUTDOWN_SECONDS
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_stdin_eof_still_detected_after_the_client_already_wrote_bytes(tmp_path: Path) -> None:
    """The realistic case: a real MCP client writes its ``initialize``
    request immediately on spawn, then later abandons the connection by
    closing its end of stdin without ever reading a reply. Those bytes sit
    unread in the pipe the whole time -- proving detection still fires
    (via POLLHUP, not a data-vs-EOF guess from `select()`) and that nothing
    about consuming/peeking those bytes breaks the shutdown path."""
    pid_file = tmp_path / "child.pid"
    upstream_script = _write_hanging_upstream_script(tmp_path)
    config_path = _write_config(tmp_path, command=[sys.executable, str(upstream_script), str(pid_file)])

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from jira_multi_mcp.cli import main; sys.exit(main())",
            "--config",
            str(config_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(pid_file, timeout=15.0)
        child_pid = int(pid_file.read_text().strip())
        assert _pid_alive(child_pid), "fake upstream child never actually started"

        assert proc.stdin is not None
        proc.stdin.write('{"jsonrpc": "2.0", "method": "initialize"}\n')
        proc.stdin.flush()
        time.sleep(0.5)
        t_close = time.monotonic()
        proc.stdin.close()

        try:
            proc.wait(timeout=_MAX_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(
                f"parent did not exit within {_MAX_SHUTDOWN_SECONDS}s of stdin closing "
                "with an unread 'initialize' request still queued"
            )
        parent_elapsed = time.monotonic() - t_close

        deadline = time.monotonic() + 5.0
        while _pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert not _pid_alive(child_pid)
        assert parent_elapsed <= _MAX_SHUTDOWN_SECONDS
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _write_real_fastmcp_upstream_script(tmp_path: Path) -> Path:
    """A fake upstream that speaks real MCP over stdio (via the installed
    fastmcp), so `start_all` actually completes and the site becomes
    healthy -- the "normal" shutdown path, as opposed to the startup-window
    one above."""
    script = tmp_path / "real_upstream.py"
    script.write_text(
        "import os, sys\n"
        "if '--version' not in sys.argv:\n"
        "    with open(sys.argv[1], 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('fake-upstream')\n"
        "mcp.run(transport='stdio', show_banner=False)\n"
    )
    return script


def test_sigterm_after_healthy_shuts_down_quickly(tmp_path: Path) -> None:
    """The cancel-before-close reordering (needed to unblock a child stuck
    mid-handshake, see the test above) must not slow down or break the
    ordinary case: a site that's already healthy shuts down well under the
    watchdog bound, not by riding it out."""
    pid_file = tmp_path / "child.pid"
    upstream_script = _write_real_fastmcp_upstream_script(tmp_path)
    config_path = _write_config(tmp_path, command=[sys.executable, str(upstream_script), str(pid_file)])

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from jira_multi_mcp.cli import main; sys.exit(main())",
            "--config",
            str(config_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_file(pid_file, timeout=15.0)
        child_pid = int(pid_file.read_text().strip())

        # Give the site a moment to actually finish the MCP handshake and
        # become healthy (not just spawned) before signaling.
        time.sleep(0.5)

        t_signal = time.monotonic()
        proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=_MAX_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(f"parent did not exit within {_MAX_SHUTDOWN_SECONDS}s of a normal SIGTERM")
        parent_elapsed = time.monotonic() - t_signal

        assert proc.returncode == 0
        # Well under the watchdog: this is the healthy-child fast path, not
        # the one that has to wait out a stuck handshake.
        assert parent_elapsed < SHUTDOWN_WATCHDOG_SECONDS

        deadline = time.monotonic() + 5.0
        while _pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(child_pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_schema_conflict_still_exits_2_not_an_exceptiongroup(tmp_path: Path) -> None:
    """`build_mirrored_tools` now runs inside the same task group as the
    shutdown-signal handler (needed to arm the receiver before `start_all`,
    see `serve`'s module docstring) -- anyio 4 wraps ANY exception escaping a
    task group, even one raised directly in the group's own body, in an
    ExceptionGroup. Proves the stash-and-reraise in `serve()` actually keeps
    `cli.main`'s `except JiraMultiError` -> exit-2 handling intact rather
    than letting a raw ExceptionGroup escape."""
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "conflicting_upstream.py"
    script.write_text(
        "import os, sys\n"
        "with open(sys.argv[1], 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('fake-upstream')\n"
        "@mcp.tool\n"
        "def jira_get_issue(site: str) -> dict:\n"
        "    return {'site': site}\n"
        "mcp.run(transport='stdio', show_banner=False)\n"
    )
    config_path = _write_config(tmp_path, command=[sys.executable, str(script), str(pid_file)])

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from jira_multi_mcp.cli import main; sys.exit(main())",
            "--config",
            str(config_path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "ExceptionGroup" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_force_exit_returns_0_when_aclose_completed() -> None:
    script = (
        "from jira_multi_mcp.server import _force_exit, _ShutdownState\n"
        "_force_exit(_ShutdownState(aclose_completed=True))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], timeout=10)
    assert completed.returncode == 0


def test_force_exit_returns_1_when_aclose_did_not_complete() -> None:
    script = (
        "from jira_multi_mcp.server import _force_exit, _ShutdownState\n"
        "_force_exit(_ShutdownState(aclose_completed=False))\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], timeout=10)
    assert completed.returncode == 1
