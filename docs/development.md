# Development Guide

## Dependency Matrix

| Install | Purpose |
| --- | --- |
| `pip install -e .` | Stable runtime dependencies only |
| `pip install -e ".[dev]"` | Local development, repo checks, packaging, and pixel/render smoke coverage |
| `pip install -e ".[atari]"` | Atari environment support |
| `pip install -e ".[offline]"` | Offline dataset support |
| `pip install -e ".[experimental]"` | Experimental namespace toggles |

## Quality Commands

```bash
make lint
make typecheck
make test-fast
make test-integration
make test-smoke
make build
make verify
```

The repo now treats quality checks as layered workflows instead of a single
undifferentiated `pytest` run.

## Marker Strategy

- `unit` is the default marker for repository tests unless a file is classified
  as integration or smoke.
- `integration` covers cross-module workflows such as CLI, checkpoint, and
  end-to-end orchestration.
- `smoke` covers representative algorithm and reference-script runs.
- `slow` is additive and excludes long-running files from the default fast CI
  path.

Marker assignment is centralized in `tests/support/markers.py` and applied at
collection time from `tests/conftest.py`.

## Current Guardrail Scope

- Ruff currently gates the repo tooling and marker harness files added in this
  milestone.
- Mypy currently gates the CLI entry surface, zoo core exports, registry type
  contracts, and the repo tooling tests.
- CI runs fast unit tests on the Python version matrix, then runs dedicated
  integration, smoke, and packaging jobs on top.

This staged scope is intentional: it adds durable guardrails without forcing a
single large cleanup across the entire algorithm surface in one change.
