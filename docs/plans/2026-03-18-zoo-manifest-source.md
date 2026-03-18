# Zoo Manifest Source Metadata Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export manifest source metadata in zoo report and leaderboard outputs so downstream consumers can see both the requested manifest path and the actual file location that was loaded.

**Architecture:** Keep the change export-only inside `src/rl_training/zoo_cli.py`. Refactor manifest loading to expose the resolved file path and source kind (`filesystem` vs `packaged_asset`), then thread a small `manifest_source` object into report and leaderboard payloads and flatten the same fields into CSV/text output.

**Tech Stack:** Python 3.10+, pathlib, argparse, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing source metadata tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test asserting `manifest_source` includes `requested_path`, `resolved_path`, and `source_kind`.
- Add a leaderboard JSON test asserting packaged-manifest resolution reports `source_kind=packaged_asset` when run outside the repo root.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_source_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_packaged_manifest_source_metadata`

Expected: FAIL because the payloads do not include `manifest_source` yet.

### Task 2: Implement manifest source export

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add a helper that resolves the manifest path and reports whether it came from the filesystem or packaged assets.
- Include `manifest_source` in report and leaderboard JSON payloads.
- Flatten the source fields into text and CSV outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_source_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_packaged_manifest_source_metadata`

Expected: PASS.

### Task 3: Document manifest source export

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Describe that machine-readable benchmark exports now include manifest source metadata showing the requested path and resolved asset/file path.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
