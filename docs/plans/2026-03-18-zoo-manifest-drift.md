# Zoo Manifest Drift Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export manifest alignment / drift markers in zoo report and leaderboard outputs so benchmark consumers can see whether run metadata still matches the current manifest protocol and preset inventory.

**Architecture:** Keep the change export-only inside `src/axiomrl/zoo_cli.py`. Enrich run rows with manifest-alignment flags based on `preset_name` and `protocol_name`, aggregate those flags into per-group counters, and add a compact top-level `manifest_alignment_summary` computed from the full filtered run set before any `--top-k` truncation.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing drift tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test asserting run rows, aggregate rows, and a top-level summary expose preset/protocol drift markers.
- Add a leaderboard JSON test asserting the same summary survives `--top-k` truncation and still reflects the full filtered run set.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_summary tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_summary`

Expected: FAIL because the payloads do not export manifest drift markers yet.

### Task 2: Implement manifest drift export

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add helpers to classify each run as aligned, preset-unknown, protocol-mismatch, or both.
- Aggregate the run-level flags into per-group counts and a top-level `manifest_alignment_summary`.
- Flatten the new fields into text and CSV outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_alignment_summary tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_alignment_summary`

Expected: PASS.

### Task 3: Document drift export

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Describe that machine-readable benchmark exports now include manifest alignment / drift counts for preset and protocol mismatches.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
