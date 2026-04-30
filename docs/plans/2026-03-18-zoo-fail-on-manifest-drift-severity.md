# Zoo Fail-On-Manifest-Drift Severity Threshold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let zoo report and leaderboard commands treat `warning` and `error` manifest drift severities differently so CI can ignore protocol-only drift while still failing on unknown presets.

**Architecture:** Extend the existing `--fail-on-manifest-drift` behavior with a severity-threshold option. Reuse the top-level `manifest_alignment_summary["severity"]` field, map severities to a stable rank, and fail only when the summary severity is at or above the requested threshold.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing threshold tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report test asserting `--fail-on-manifest-drift-severity error` does **not** fail for protocol-only drift (`warning`).
- Add a leaderboard test asserting the same flag **does** fail for unknown-preset drift (`error`).

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_threshold_fail_on_manifest_drift_severity tests/test_cli.py::test_zoo_subcommand_leaderboard_can_threshold_fail_on_manifest_drift_severity`

Expected: FAIL because the CLI currently only supports all-or-nothing drift failure.

### Task 2: Implement severity-threshold exits

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`
- Modify: `src/axiomrl/cli.py`

**Step 1: Write minimal implementation**
- Add `--fail-on-manifest-drift-severity {warning,error}` to the zoo CLI and top-level `axiomrl zoo`.
- Treat the new flag as enabling drift failure even when `--fail-on-manifest-drift` is omitted.
- Compare the summary severity rank against the requested threshold to decide the exit code.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_threshold_fail_on_manifest_drift_severity tests/test_cli.py::test_zoo_subcommand_leaderboard_can_threshold_fail_on_manifest_drift_severity`

Expected: PASS.

### Task 3: Document threshold behavior

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Document that `--fail-on-manifest-drift` means any drift, while `--fail-on-manifest-drift-severity error` only fails for `error` summaries.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
