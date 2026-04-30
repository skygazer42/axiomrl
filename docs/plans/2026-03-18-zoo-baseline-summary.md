# Zoo Baseline Summary Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export a concise baseline-aware benchmark summary that highlights the biggest movers and regressions relative to a named preset.

**Architecture:** Reuse the existing preset-level baseline comparison fields in `src/axiomrl/zoo_cli.py` to build a `baseline_summary` section for report and leaderboard payloads. Summaries rank aggregate preset rows by baseline-relative deltas, exclude the baseline preset itself, and expose the top movers/regressions for return and normalized score in text, JSON, and CSV outputs.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test verifying `baseline_summary` includes top movers and regressions by return and normalized delta.
- Add a leaderboard JSON test verifying `baseline_summary` survives leaderboard rendering and still reflects the full filtered aggregate set when `top_k` truncates visible entries.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_includes_baseline_summary_top_movers_and_regressions tests/test_cli.py::test_zoo_subcommand_leaderboard_includes_baseline_summary_top_movers_and_regressions`

Expected: FAIL because the summary section does not exist yet.

### Task 2: Implement baseline summary export

**Files:**
- Modify: `src/axiomrl/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add a helper that builds top-mover and top-regression lists from aggregate preset rows with baseline fields attached.
- Include `baseline_summary` in report and leaderboard payloads.
- Render summary data in text output and CSV rows.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document summary export

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/axiomrl/assets/zoo/README.md`

**Step 1: Add docs**
- Describe the automatic top-movers/regressions summary when `--baseline-preset` is active.
- Note that the summary is exported in machine-readable outputs alongside aggregate rows.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
