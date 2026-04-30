# Zoo Manifest Fingerprint Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export a compact manifest identity block in zoo report and leaderboard outputs so benchmark results can be compared against the exact manifest snapshot that produced them.

**Architecture:** Keep the change export-only inside `src/axiomrl/zoo_cli.py`. Derive a deterministic manifest fingerprint from the loaded manifest payload, expose a small `manifest_metadata` object at the top level, and flatten the same fields into CSV outputs without changing benchmark behavior or duplicating the full manifest body in every export.

**Tech Stack:** Python 3.10+, argparse, JSON/YAML serialization, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing manifest metadata tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test asserting the payload includes `manifest_metadata` with deterministic fingerprint and preset inventory details.
- Add a leaderboard JSON test asserting the same metadata survives leaderboard export when entries are truncated with `--top-k`.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_identity_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_identity_metadata`

Expected: FAIL because the payloads do not expose manifest identity metadata yet.

### Task 2: Implement manifest identity export

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add a helper that builds deterministic manifest metadata from the loaded manifest.
- Include `manifest_metadata` in report and leaderboard JSON payloads.
- Flatten the compact manifest fields into text and CSV outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_identity_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_identity_metadata`

Expected: PASS.

### Task 3: Document manifest identity export

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Describe that machine-readable benchmark exports now include a compact manifest fingerprint block for downstream reproducibility checks.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
