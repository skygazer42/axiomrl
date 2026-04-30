# Zoo Leaderboard Robustness Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose robust cross-seed benchmark summaries by adding median and interquartile-range aggregate fields, plus leaderboard modes that rank by robust central tendency or dispersion.

**Architecture:** Extend aggregate benchmark rows in `src/axiomrl/zoo_cli.py` with latest-metric `median` and `iqr` fields computed from ordered seed values using inclusive quartiles so small benchmark groups stay well-defined. Reuse those aggregate fields in the leaderboard alias layer by adding `median-*` and `iqr-*` modes where median sorts descending and IQR sorts ascending to surface both robust score level and cross-seed spread.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON aggregate test verifying latest return / normalized `median` and `iqr` fields are emitted.
- Add leaderboard JSON tests verifying `--leaderboard-metric median-normalized` sorts by normalized-score median descending and `--leaderboard-metric iqr-normalized` sorts by normalized-score IQR ascending.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_aggregates_include_latest_robustness_statistics tests/test_cli.py::test_zoo_subcommand_leaderboard_supports_median_normalized_metric_mode tests/test_cli.py::test_zoo_subcommand_leaderboard_supports_iqr_normalized_metric_mode`

Expected: FAIL because the new aggregate fields and leaderboard aliases do not exist yet.

### Task 2: Implement robustness aggregation and ranking

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`
- Modify: `src/axiomrl/cli.py`

**Step 1: Write minimal implementation**
- Add helpers for median, inclusive quartiles, and IQR.
- Emit latest return / latest normalized `median` and `iqr` aggregate fields.
- Extend leaderboard metric aliases with `median-return`, `median-normalized`, `iqr-return`, and `iqr-normalized`.
- Make median aliases sort descending and IQR aliases sort ascending.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document robustness modes

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--leaderboard-metric median-normalized` and `--leaderboard-metric iqr-normalized`.
- Explain that median modes rank higher robust central tendency higher and IQR modes rank lower spread higher.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
