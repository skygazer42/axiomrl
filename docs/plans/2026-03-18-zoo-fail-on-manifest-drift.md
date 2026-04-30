# Zoo Fail-On-Manifest-Drift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a zoo CLI flag that preserves normal report and leaderboard output but returns a non-zero exit code when the filtered benchmark slice contains manifest drift.

**Architecture:** Reuse the existing top-level `manifest_alignment_summary` produced by `build_report_payload()`. Thread a `--fail-on-manifest-drift` boolean through the report and leaderboard print paths, emit output as usual, then return exit code `1` when `drifted_runs > 0`; otherwise return `0`.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing CLI tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report test asserting `--fail-on-manifest-drift` returns `1` when report payloads contain drift and still prints the report.
- Add a leaderboard test asserting the same behavior for leaderboard output.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_fail_on_manifest_drift tests/test_cli.py::test_zoo_subcommand_leaderboard_can_fail_on_manifest_drift`

Expected: FAIL because the parser and CLI currently always return `0`.

### Task 2: Implement non-zero drift exit codes

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add `--fail-on-manifest-drift` to the parser.
- Make report and leaderboard print helpers return exit codes based on the rendered payload’s `manifest_alignment_summary`.
- Keep table and commands behavior unchanged.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_fail_on_manifest_drift tests/test_cli.py::test_zoo_subcommand_leaderboard_can_fail_on_manifest_drift`

Expected: PASS.

### Task 3: Document the CLI flag

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Note that `--fail-on-manifest-drift` makes report and leaderboard commands usable as CI guards while still emitting machine-readable artifacts.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
