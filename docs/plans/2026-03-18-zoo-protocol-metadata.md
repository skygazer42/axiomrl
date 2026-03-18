# Zoo Protocol Metadata Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export manifest-derived benchmark protocol, score-normalization, and preset metadata alongside zoo report and leaderboard outputs.

**Architecture:** Keep the change export-only inside `src/rl_training/zoo_cli.py`. Build lightweight manifest lookup helpers keyed by preset name/config, enrich per-run and aggregate payloads with protocol / score-normalization / preset metadata, and thread the same structured data through text, JSON, and CSV serializers without changing training or environment behavior.

**Tech Stack:** Python 3.10+, argparse, YAML manifest loader, existing zoo CLI serializers, pytest.

---

### Task 1: Add failing metadata export tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test verifying aggregate preset rows include manifest-derived `protocol_metadata`, `score_normalization_metadata`, and `preset_metadata`.
- Add a leaderboard JSON test verifying the same metadata survives leaderboard export even when `--top-k` truncates visible entries.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_protocol_and_preset_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_protocol_and_preset_metadata`

Expected: FAIL because the export payloads do not include the structured metadata yet.

### Task 2: Implement manifest metadata export

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add manifest lookup helpers that resolve preset entries from the manifest by preset name or config path.
- Enrich per-run and aggregate report rows with manifest-derived protocol / score-normalization / preset metadata.
- Include the same metadata in report and leaderboard JSON / text / CSV outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_manifest_protocol_and_preset_metadata tests/test_cli.py::test_zoo_subcommand_leaderboard_json_includes_manifest_protocol_and_preset_metadata`

Expected: PASS.

### Task 3: Document metadata export

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Describe that machine-readable zoo benchmark outputs now expose full protocol metadata, score-normalization metadata, and preset metadata from the manifest.
- Keep the docs scoped to reporting/export behavior only.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
