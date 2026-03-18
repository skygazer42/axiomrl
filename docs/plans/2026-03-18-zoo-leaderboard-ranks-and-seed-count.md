# Zoo Leaderboard Ranks and Seed Count Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make zoo benchmark summaries clearer by exposing explicit `seed_count`, latest-vs-best ratio metrics, and multi-axis rank columns in report and leaderboard outputs.

**Architecture:** Extend aggregate benchmark rows in `src/rl_training/zoo_cli.py` with explicit seed-count and ratio-derived comparison fields computed from the existing latest/best metrics. Reuse those aggregate rows as the source of truth for leaderboard payloads, then stamp deterministic rank columns for the main benchmark axes so text, JSON, and CSV renderers can expose richer leaderboard comparisons without changing trainer metadata or checkpoint semantics.

**Tech Stack:** Python 3.10+, existing zoo CLI/report renderers, JSON/CSV serializers, pytest.

---

### Task 1: Add failing regression tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a report JSON test that verifies aggregate rows include `seed_count`, ratio fields, and per-metric rank columns.
- Add a leaderboard CLI/JSON test that verifies leaderboard entries expose `seed_count` and multiple rank axes with stable ordering.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: FAIL because the new aggregate and leaderboard fields are not emitted yet.

### Task 2: Implement aggregate metrics and rank derivation

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add `seed_count` alongside existing `runs`.
- Derive best-over-latest ratios for return and normalized score when both sides are present and the denominator is non-zero.
- Compute deterministic rank columns across aggregate entries for best/latest return and normalized score, then attach them to leaderboard entries.
- Keep current report behavior backward compatible by preserving existing fields.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

Expected: PASS.

### Task 3: Document the richer benchmark summary fields

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Document `seed_count`, ratio fields, and multi-axis leaderboard ranks.
- Keep examples aligned with existing `axiomrl zoo --format report` and `--format leaderboard` usage.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
