# Zoo Leaderboard Stability Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make benchmark comparisons more robust by exposing cross-seed stability statistics and adding leaderboard modes that rank by seed-to-seed consistency.

**Architecture:** Extend aggregate benchmark rows in `src/axiomrl/zoo_cli.py` with latest-metric stability summaries across seeds, specifically `min`, `max`, and `std` for return and normalized score. Reuse those fields in the existing leaderboard metric alias layer by adding `stability-return` and `stability-normalized` modes that sort ascending on standard deviation so lower variance ranks higher while single-seed groups remain unrated (`None`) and fall to the end.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON aggregate test verifying `latest_*_min`, `latest_*_max`, and `latest_*_std` fields are emitted.
- Add a leaderboard JSON test verifying `--leaderboard-metric stability-normalized` sorts by normalized-score standard deviation ascending and reports the resolved metric metadata.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because the new aggregate fields and stability aliases do not exist yet.

### Task 2: Implement stability aggregation and ranking

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`
- Modify: `src/axiomrl/cli.py`

**Step 1: Write minimal implementation**
- Add helpers for aggregate `min` and `std` on numeric fields.
- Emit latest return / latest normalized `min`, `max`, and `std` aggregate fields.
- Extend leaderboard metric aliases with `stability-return` and `stability-normalized`.
- Make stability aliases sort ascending because lower standard deviation is better.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document stability modes

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--leaderboard-metric stability-normalized`.
- Explain that stability modes rank lower cross-seed standard deviation higher.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
