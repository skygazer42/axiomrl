# Contributing

Thanks for contributing to AxiomRL.

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

The `dev` extra includes the repo quality tools plus the optional pixel/render
dependencies used by a subset of smoke tests.

## Daily Workflow

```bash
make lint
make typecheck
make test-fast
make test-integration
make test-smoke
make build
```

Use `make verify` before opening a pull request when your change touches
multiple subsystems.

## Test Layers

- `unit`: fast isolated tests and repository contract checks
- `integration`: cross-module workflow tests such as CLI, checkpoint, and
  end-to-end orchestration flows
- `smoke`: representative training/runtime checks for algorithms and reference
  scripts
- `slow`: tests intentionally excluded from the default fast path

## Change Expectations

- Add or update tests for every behavior change.
- Keep stable APIs documented in `README.md` and `docs/compatibility.md`.
- Update `CHANGELOG.md` when the public developer experience changes.
- Prefer isolated feature branches or git worktrees instead of working on
  `main`.

## Feature-Specific Notes

- New CLI surface area should include success-path and failure-path tests.
- New algorithms or presets should include at least one representative smoke
  test plus packaging/resource coverage when assets are shipped.
- Changes to package metadata or release flows should keep
  `tests/test_release_contracts.py` green.
