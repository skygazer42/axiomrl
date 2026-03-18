# Zoo Manifest Drift Severity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export manifest drift severity and named drifted preset summaries so downstream benchmark tooling can quickly distinguish clean runs, protocol-only drift, and manifest-inventory errors.

**Architecture:** Keep the change inside `src/rl_training/zoo_cli.py`. Extend existing manifest alignment metadata with a severity classifier, attach it to run and aggregate records, and enrich the top-level `manifest_alignment_summary` with severity plus a sorted list of drifted preset names derived from the full filtered run set before any `--top-k` truncation.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing severity tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Extend the report JSON drift test to assert aggregate- and summary-level severity plus a sorted `drifted_presets` list.
- Extend the leaderboard JSON drift test to assert the summary still reports hidden drifted presets after `--top-k` truncation.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_summary tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_summary`

Expected: FAIL because the current payloads do not export severity or named drifted preset summaries.

### Task 2: Implement severity export

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add a shared severity helper that maps clean alignment to `clean`, protocol-only drift to `warning`, and any unknown-preset drift to `error`.
- Attach `manifest_alignment_severity` to run and aggregate records.
- Extend `manifest_alignment_summary` with `severity` and `drifted_presets`.
- Flatten the new metadata into CSV output and compact text output without printing hidden preset inventories in text mode.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_summary tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_summary`

Expected: PASS.

### Task 3: Document severity fields

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Document that machine-readable zoo exports now include manifest drift severity and a named list of drifted presets in the top-level summary.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
