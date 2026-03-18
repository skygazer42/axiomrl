# Zoo Report Deltas and Leaderboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit latest-vs-best comparison columns to zoo benchmark reports and provide a dedicated leaderboard view for ranked benchmark summaries.

**Architecture:** Extend run and aggregate records with derived delta metrics computed from existing latest/best values. Add a new `leaderboard` output mode to `axiomrl zoo` that reuses the aggregate pipeline, defaults to ranking by best return, and supports the existing JSON/CSV/text renderers with a leaderboard-specific payload. Keep `report` backward compatible and make leaderboard opt-in via `--format leaderboard`.

**Tech Stack:** Python 3.10+, existing zoo CLI renderers, JSON/CSV serializers, pytest.

---

### Task 1: Add failing delta and leaderboard tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a JSON report test that verifies run rows expose `best_minus_latest_*` delta fields and aggregates expose latest-vs-best gap fields.
- Add a leaderboard CLI test that verifies `--format leaderboard` ranks grouped entries, includes rank numbers, and defaults to best-return ordering.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because delta fields and leaderboard format do not exist yet.

### Task 2: Implement delta metrics and leaderboard view

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Compute per-run delta fields between best and latest metrics.
- Compute per-aggregate gap fields between best maxima and latest means.
- Add `--format leaderboard`.
- Reuse grouping, sorting, filtering, top-k, and output-path support for leaderboard.
- Default leaderboard ordering to best return descending when no explicit `--sort-by` is provided.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document leaderboard usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show `axiomrl zoo --format leaderboard`.
- Explain the new latest-vs-best delta fields.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
