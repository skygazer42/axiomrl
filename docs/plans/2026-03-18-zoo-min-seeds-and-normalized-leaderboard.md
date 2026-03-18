# Zoo Min-Seeds and Normalized Leaderboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve benchmark comparability by filtering out under-seeded aggregate rows and making leaderboard defaults prefer normalized scores when the benchmark manifest provides score normalization.

**Architecture:** Extend `src/rl_training/zoo_cli.py` so report/leaderboard payload construction can drop aggregate groups whose `seed_count` falls below a user-provided `--min-seeds` threshold while leaving per-run rows intact. Update leaderboard default sort resolution to prefer normalized-score benchmark axes when the manifest declares score normalization, while preserving explicit `--sort-by` behavior and existing JSON/CSV/text output contracts.

**Tech Stack:** Python 3.10+, existing zoo CLI/report renderers, argparse, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test that verifies `--min-seeds` filters aggregate groups but keeps raw run rows available.
- Add a leaderboard JSON test that verifies the default sort switches to human-normalized score when the manifest exposes score normalization and that `--min-seeds` removes low-seed entries.
- Add a CLI parser/forwarding test for the new `--min-seeds` option.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because `--min-seeds` is unsupported and leaderboard still defaults to best return ordering.

### Task 2: Implement filtering and default leaderboard ranking

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add `--min-seeds` argument plumbing through report and leaderboard code paths.
- Filter aggregate rows on `seed_count >= min_seeds`.
- Preserve run-row output for report mode so raw metadata remains inspectable even when an aggregate group is excluded.
- Resolve leaderboard default sort to normalized-score fields when `score_normalization.type != none`, otherwise keep best-return ordering.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document new benchmark filters and defaults

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Document `--min-seeds`.
- Document the leaderboard default preference for normalized-score ordering on normalized benchmarks.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
