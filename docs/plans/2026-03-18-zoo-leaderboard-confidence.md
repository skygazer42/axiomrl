# Zoo Leaderboard Confidence Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose uncertainty-aware benchmark summaries by adding standard error and 95% confidence-interval fields, plus leaderboard modes that rank tighter confidence bounds higher.

**Architecture:** Extend aggregate benchmark rows in `src/axiomrl/zoo_cli.py` with latest-metric `stderr` and `ci95` half-width fields derived from the existing sample standard deviation and seed count. Reuse those fields in the leaderboard metric alias layer by adding `confidence-return` and `confidence-normalized` modes that sort ascending on CI width so tighter cross-seed estimates rank higher while single-seed groups remain unrated (`None`) and fall to the end.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON aggregate test verifying latest return / normalized `stderr` and `ci95` fields are emitted.
- Add a leaderboard JSON test verifying `--leaderboard-metric confidence-normalized` sorts by normalized-score CI width ascending and reports the resolved metric metadata.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because the new aggregate fields and confidence aliases do not exist yet.

### Task 2: Implement confidence aggregation and ranking

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`
- Modify: `src/axiomrl/cli.py`

**Step 1: Write minimal implementation**
- Add helpers for aggregate `stderr` and `ci95` from the existing sample standard deviation.
- Emit latest return / latest normalized `stderr` and `ci95` aggregate fields.
- Extend leaderboard metric aliases with `confidence-return` and `confidence-normalized`.
- Make confidence aliases sort ascending because narrower confidence intervals are better.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document confidence modes

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--leaderboard-metric confidence-normalized`.
- Explain that confidence modes rank lower 95% CI half-width higher.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
