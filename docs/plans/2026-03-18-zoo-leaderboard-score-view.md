# Zoo Leaderboard Score View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a higher-level leaderboard score-axis switch so users can choose return-based or normalized-score-based ranking without spelling raw metric aliases.

**Architecture:** Extend `src/rl_training/zoo_cli.py` with a leaderboard-only `--score-view return|normalized` option that resolves through the existing compare-to/metric alias layer. When no explicit `--leaderboard-metric` or raw `--sort-by` is provided, `score_view` should choose the return or normalized axis while `compare_to` still selects best vs latest. Carry the resolved `score_view` into leaderboard payload metadata and reject `score_view=normalized` when the manifest has no score normalization configured.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a leaderboard JSON test verifying `--compare-to latest --score-view return` on a normalized benchmark resolves to `latest-return` and sorts by return instead of normalized score.
- Add a leaderboard test verifying `--score-view normalized` on an unnormalized manifest raises a clear error.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_cli.py`

Expected: FAIL because `--score-view` is not supported yet.

### Task 2: Implement score-view resolution

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add `--score-view` parser support.
- Resolve `score_view` through the existing leaderboard metric alias layer only when no explicit `--leaderboard-metric` or raw `--sort-by` is provided.
- Surface `score_view` in leaderboard payload metadata and text/CSV outputs.
- Reject `score_view=normalized` when the manifest lacks score normalization.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_cli.py`

Expected: PASS.

### Task 3: Document score-view usage

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--score-view return`.
- Explain that `--score-view` controls the return vs normalized axis while `--compare-to` controls best vs latest.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
