# Zoo Report Grouping and Top-K Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `axiomrl zoo --format report` more useful for benchmark ranking by supporting preset-aware grouping and top-k truncation.

**Architecture:** Extend run report extraction to carry `benchmark` metadata such as `suite`, `preset_name`, and `protocol_name` from each run's `metadata.json`. Add a `--group-by` switch so aggregate summaries can group by either `(algo, env_id)` or `preset_name`, and add `--top-k` so sorted report outputs can be truncated deterministically. Keep defaults backward compatible by preserving `algo-env` grouping and unlimited result counts.

**Tech Stack:** Python 3.10+, existing zoo report payload/renderers, JSON run metadata, pytest.

---

### Task 1: Add failing grouping and top-k tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a JSON report test that includes benchmark `preset_name` metadata in run folders, requests `--group-by preset --top-k 1`, and verifies only the top preset aggregate remains.
- Add a CSV report test that verifies run rows expose `preset_name` / `protocol_name` columns and that `--top-k 1` truncates run rows after sorting.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because grouping is fixed to `(algo, env_id)`, benchmark metadata is not surfaced in reports, and `--top-k` does not exist.

### Task 2: Implement preset-aware grouping and top-k truncation

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Parse benchmark metadata from run `metadata.json`.
- Add `--group-by` with `algo-env` and `preset`.
- Add `--top-k`.
- Apply sorting before truncation.
- Surface `preset_name` and `protocol_name` in JSON/CSV/text report outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document grouping and top-k usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--group-by preset`.
- Show `--top-k`.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
