# Zoo Report Exports and Filters Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `axiomrl zoo --format report` so benchmark reports can be exported as JSON/CSV and filtered or sorted without a separate analysis script.

**Architecture:** Keep the current text report behavior as the default. Add a report-output layer in `src/rl_training/zoo_cli.py` that transforms the existing per-run and aggregate summaries into either text, JSON, or CSV. Add lightweight filters for `algo` / `env_id` and sorting by common benchmark metrics before rendering. Keep the CLI backward compatible by making all new options optional.

**Tech Stack:** Python 3.10+, stdlib `json` / `csv`, existing zoo CLI, pytest.

---

### Task 1: Add failing report export tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a JSON report test that verifies manifest metadata, filtered runs, and aggregate blocks are emitted in machine-readable form.
- Add a CSV report test that verifies rows include both `run` and `aggregate` records with stable headers.
- Add a sorting/filtering test that verifies `--algo`, `--env-id`, and `--sort-by` affect report ordering.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because report export/filter/sort options do not exist yet.

### Task 2: Implement report export/filter/sort pipeline

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add optional CLI flags:
  - `--report-output text|json|csv`
  - `--algo`
  - `--env-id`
  - `--sort-by`
  - `--descending`
- Filter run reports before aggregation.
- Sort both per-run and aggregate summaries with a stable comparator.
- Render report output as text, JSON, or CSV.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document machine-readable report usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show JSON and CSV export examples.
- Document the filter and sort flags used for multi-seed benchmark inspection.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
