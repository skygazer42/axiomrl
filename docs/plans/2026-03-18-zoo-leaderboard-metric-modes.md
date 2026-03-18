# Zoo Leaderboard Metric Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make leaderboard ranking easier to use by exposing a small set of explicit metric modes for latest/best/gap comparisons instead of requiring users to know raw `--sort-by` field names.

**Architecture:** Extend `src/rl_training/zoo_cli.py` with a leaderboard-metric resolver that maps stable user-facing mode names onto existing aggregate fields such as best/latest return, best/latest normalized score, and latest-vs-best gap metrics. Keep the generic `--sort-by` path for advanced use, but let leaderboard mode resolve a canonical `leaderboard_metric` into the appropriate sort field and reflect that choice in JSON/text/CSV payload metadata.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a leaderboard JSON test that verifies `--leaderboard-metric latest-normalized` sorts entries by latest normalized score and reports the resolved metric metadata.
- Add a leaderboard JSON test that verifies `--leaderboard-metric gap-return` sorts entries by latest-vs-best return gap.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_cli.py`

Expected: FAIL because `--leaderboard-metric` is not supported yet.

### Task 2: Implement metric alias resolution

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add `--leaderboard-metric` parser support with a constrained choice set.
- Resolve metric aliases to existing aggregate sort fields.
- Let explicit leaderboard metric override the default leaderboard sort resolution while leaving report mode untouched.
- Surface the resolved metric in leaderboard payload metadata and text/CSV outputs.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_cli.py`

Expected: PASS.

### Task 3: Document leaderboard metric modes

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show example commands for `latest-normalized` and `gap-return`.
- Explain that `--leaderboard-metric` is the ergonomic alias layer over raw `--sort-by`.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
