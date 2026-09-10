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
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -m "not integration"
```
