# Contributing

## Commit messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): message`, where `type` is one of `feat`, `fix`, `chore`, `docs`,
`ci`, `test`, or `refactor`. The scope is optional.

## Pull requests

Open pull requests against `main`. The PR title must itself be a valid
Conventional Commit message; CI checks this.

## Local development

```bash
uv sync --all-groups
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -m "not integration"
uv build
uvx --from . jira-multi-mcp --help
```

These are the same commands CI runs (`.github/workflows/ci.yaml`), on Python
3.10 and 3.13 — matching them locally before pushing saves a round trip.

## Running the integration suite

`tests/integration/test_live_sites.py` drives the real server against your
real, configured Jira sites (no mocks) — it's skipped by default. It needs:

- A working config (`JIRA_MULTI_CONFIG`, or the default
  `${XDG_CONFIG_HOME:-~/.config}/jira-multi-mcp/config.toml`), verified with
  `jira-multi-mcp --check` first.
- `JIRA_MULTI_INTEGRATION=1` to opt in.
- `JIRA_MULTI_TEST_SANDBOX_ISSUE=<KEY>` (a real, writable issue you're happy
  to receive and lose test attachments) to additionally run the
  upload → list → download → delete round trip. Without it, the suite still
  runs its read-only checks (`jira_sites`, one `jira_get_issue` /
  `jira_list_attachments` per configured site) but hard-fails rather than
  silently skipping if it would otherwise need to write.

```bash
JIRA_MULTI_INTEGRATION=1 JIRA_MULTI_TEST_SANDBOX_ISSUE=JMI-395 uv run pytest -m integration
```

Never `cat`/`grep`/`sed` your own `config.toml` in a transcript an agent or a
CI log might capture — it holds a real credential. Use `--print-config`
(secrets always masked) to inspect it instead.

## Adversarial review expectation

A PR here is expected to go through a review-and-fix loop (a rigorous code
review plus, for anything runtime-dependent, real execution — not just
mocked unit tests) before it's considered done. See
`~/Code/research/.claude/skills/adversarial-fix-loop` if you're driving that
loop yourself; otherwise just expect review comments that ask for real
`--check`/integration evidence, not just "the unit tests pass."
