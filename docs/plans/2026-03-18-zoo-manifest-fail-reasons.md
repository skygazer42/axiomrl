# Zoo Manifest Fail Reasons Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export explicit `manifest_alignment_fail_reasons` in zoo report and leaderboard outputs so CI and downstream tooling can see why the current drift gate would fail without parsing multiple summary counters.

**Architecture:** Keep the change inside `src/rl_training/zoo_cli.py`. Reuse the existing manifest alignment summary plus the active CLI drift-gate options (`--fail-on-manifest-drift`, `--fail-on-manifest-drift-severity`, `--fail-on-manifest-drift-type`) to derive a normalized list of fail reasons, attach it to the rendered payload, and flatten it into text and CSV metadata.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing fail-reason tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test asserting protocol-only drift with `--fail-on-manifest-drift-type protocol-mismatch` exports `manifest_alignment_fail_reasons == ["protocol-mismatch"]`.
- Add a leaderboard JSON test asserting an `error` threshold exports `manifest_alignment_fail_reasons == ["unknown-preset"]` when unknown presets are what actually trip the failure.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_fail_reasons tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_fail_reasons`

Expected: FAIL because current payloads do not export a fail-reasons field.

### Task 2: Implement fail-reason export

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add a helper that derives fail reasons from the manifest summary plus the active drift-gate options.
- Use that helper both for exit-code decisions and for the new top-level `manifest_alignment_fail_reasons` payload field.
- Flatten the new field into text and CSV output metadata.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_fail_reasons tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_fail_reasons`

Expected: PASS.

### Task 3: Document fail reasons

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Document that machine-readable zoo outputs now include `manifest_alignment_fail_reasons`, reflecting the current CLI fail-gate configuration.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
