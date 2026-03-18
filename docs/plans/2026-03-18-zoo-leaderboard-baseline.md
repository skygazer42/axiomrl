# Zoo Leaderboard Baseline Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add baseline-aware benchmark comparisons so zoo reports and leaderboards can rank presets by improvement over a named baseline preset.

**Architecture:** Extend aggregate preset rows in `src/rl_training/zoo_cli.py` with baseline-relative delta and ratio fields computed against a named preset's latest mean return and normalized score. Thread a new `--baseline-preset` CLI option through report and leaderboard flows, require `--group-by preset` for baseline comparisons, and expose explicit leaderboard aliases for delta-vs-baseline and ratio-vs-baseline ranking.

**Tech Stack:** Python 3.10+, argparse, existing zoo CLI/report serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON aggregate test verifying baseline-relative return / normalized delta and ratio fields are emitted when `--baseline-preset` is provided.
- Add leaderboard JSON tests verifying `delta-vs-baseline-normalized` and `ratio-vs-baseline-return` resolve to the expected fields and ranking order.
- Add a validation test verifying baseline leaderboard metrics reject missing `--baseline-preset`.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_zoo_presets.py::test_zoo_cli_report_json_aggregates_include_baseline_comparison_statistics tests/test_cli.py::test_zoo_subcommand_leaderboard_supports_delta_vs_baseline_normalized_metric_mode tests/test_cli.py::test_zoo_subcommand_leaderboard_supports_ratio_vs_baseline_return_metric_mode tests/test_cli.py::test_zoo_subcommand_leaderboard_rejects_baseline_metric_without_baseline_preset`

Expected: FAIL because baseline fields, CLI option plumbing, and leaderboard aliases do not exist yet.

### Task 2: Implement baseline aggregation and ranking

**Files:**
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add a helper that attaches baseline-relative fields to aggregate rows.
- Add `--baseline-preset` to report / leaderboard CLI plumbing and require `--group-by preset`.
- Emit baseline delta / ratio fields in text, JSON, and CSV outputs.
- Extend leaderboard metric aliases with `delta-vs-baseline-return`, `delta-vs-baseline-normalized`, `ratio-vs-baseline-return`, and `ratio-vs-baseline-normalized`.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document baseline modes

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Show `--baseline-preset dqn_breakout`.
- Document baseline delta/ratio outputs and alias names.
- State that `--baseline-preset` requires `--group-by preset`.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
