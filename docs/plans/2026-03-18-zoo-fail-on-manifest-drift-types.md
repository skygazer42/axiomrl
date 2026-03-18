# Zoo Fail-On-Manifest-Drift Type Filters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let zoo report and leaderboard commands fail only for selected manifest drift categories such as unknown presets or protocol mismatches.

**Architecture:** Extend the existing drift exit-code helper with explicit type filters. Reuse the top-level `manifest_alignment_summary` counters (`unknown_preset_runs`, `protocol_mismatch_runs`) and add a repeatable CLI flag that maps directly to those counters, so CI can opt into failing on specific drift categories while leaving other drift types as warnings.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing drift-type tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report test asserting `--fail-on-manifest-drift-type unknown-preset` does not fail for protocol-only drift.
- Add a leaderboard test asserting `--fail-on-manifest-drift-type protocol-mismatch` does fail for protocol drift while preserving output.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_filter_fail_on_manifest_drift_type tests/test_cli.py::test_zoo_subcommand_leaderboard_can_filter_fail_on_manifest_drift_type`

Expected: FAIL because the CLI does not yet accept or evaluate drift-type filters.

### Task 2: Implement type-filtered exits

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add repeatable `--fail-on-manifest-drift-type {unknown-preset,protocol-mismatch}` flags to the zoo CLI and top-level `axiomrl zoo`.
- Treat explicit type filters as an additional exit gate alongside the existing all-drift and severity-threshold gates.
- Map each selected type directly to the corresponding `manifest_alignment_summary` counter.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_can_filter_fail_on_manifest_drift_type tests/test_cli.py::test_zoo_subcommand_leaderboard_can_filter_fail_on_manifest_drift_type`

Expected: PASS.

### Task 3: Document drift-type filters

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Document that `--fail-on-manifest-drift-type unknown-preset` and `--fail-on-manifest-drift-type protocol-mismatch` target specific drift categories and can be repeated.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
