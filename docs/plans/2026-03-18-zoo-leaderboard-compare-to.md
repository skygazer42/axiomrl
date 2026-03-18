# Zoo Leaderboard Compare-To Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a higher-level leaderboard switch that lets users compare peak performance versus final performance without needing to understand the full leaderboard metric alias set.

**Architecture:** Extend `src/rl_training/zoo_cli.py` with a leaderboard-only `--compare-to latest|best` option that resolves into the existing leaderboard metric alias layer. When no explicit `--leaderboard-metric` or raw `--sort-by` is provided, `compare_to=latest` should select `latest-normalized` on normalized benchmarks and `latest-return` on unnormalized ones; `compare_to=best` should analogously resolve to `best-*`. Surface the resolved `compare_to` and `leaderboard_metric` in leaderboard payload metadata and renderers.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a leaderboard JSON test verifying `--compare-to latest` on a normalized benchmark resolves to `latest-normalized` and sorts by latest normalized score.
- Add a leaderboard JSON test verifying `--compare-to latest` on an unnormalized manifest resolves to `latest-return` and sorts by latest return.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_cli.py`

Expected: FAIL because `--compare-to` is not supported yet.

### Task 2: Implement compare-to resolution

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add `--compare-to` parser support in both CLI entry points.
- Resolve `compare_to` through the existing leaderboard metric alias layer only when no explicit `--leaderboard-metric` or `--sort-by` is provided.
- Carry `compare_to` into leaderboard payload metadata and text/CSV outputs when it is the active resolution path.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_cli.py`

Expected: PASS.

### Task 3: Document compare-to usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--compare-to latest`.
- Explain that it is the higher-level “best vs latest” switch while `--leaderboard-metric` remains the lower-level metric selector.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
