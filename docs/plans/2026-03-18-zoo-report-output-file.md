# Zoo Report Output File Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let `axiomrl zoo --format report` write its rendered text, JSON, or CSV output directly to a file path for reproducible benchmark artifacts.

**Architecture:** Keep stdout behavior intact by default. Add an optional `--output` flag to both `zoo_cli` and the main CLI subcommand, then route rendered report content through a small output helper that writes the content to disk when requested. Use the already structured report payload and rendering code so file output stays identical to what the CLI prints.

**Tech Stack:** Python 3.10+, stdlib file I/O, existing zoo report renderer, pytest.

---

### Task 1: Add failing output-file tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a `zoo_main(...)` test that requests `--report-output json --output <file>` and verifies the file contains the JSON payload.
- Add a main CLI test that requests `--report-output csv --output <file>` and verifies the CSV file contains run and aggregate rows.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because `--output` does not exist yet.

### Task 2: Implement output-file support

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`
- Modify: `src/axiomrl/cli.py`

**Step 1: Write minimal implementation**
- Add `--output`.
- Render report content to a string before writing.
- Create parent directories when needed.
- Preserve stdout output when no output path is provided.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document output-file usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--output benchmark_report.json` / `benchmark_report.csv`.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
