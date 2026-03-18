# Benchmark Normalization and Best Checkpoint Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make training runs benchmark-ready by adding optional score normalization metadata and automatic best-checkpoint tracking.

**Architecture:** Extend `TrainConfig` with an optional top-level `benchmark` mapping. Add a small benchmarking utility module that resolves score-normalization settings and computes human-normalized scores from evaluation returns. Update `save_training_checkpoint(...)` so every trainer gets `checkpoints/best.pt`, run metadata for the best checkpoint, and benchmark-augmented metrics without changing each trainer implementation. Update `evaluate_checkpoint(...)` to return normalized metrics when the saved config includes benchmark settings.

**Tech Stack:** Python 3.10+, existing checkpointing/run metadata flow, pytest.

---

### Task 1: Add failing benchmark tests

**Files:**
- Create: `tests/test_benchmarking.py`
- Modify: `tests/test_checkpoint_workflows.py`

**Step 1: Write the failing tests**
- Add a unit test that saves multiple checkpoints and verifies:
  - `checkpoints/best.pt` is created
  - the best checkpoint tracks the best `eval_return_mean`
  - normalized score metrics are added when benchmark score references are configured
- Add a workflow test proving `evaluate_checkpoint(...)` returns the normalized metric from the saved config.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_benchmarking.py tests/test_checkpoint_workflows.py`
- Expected: failures because benchmark config, normalization, and best-checkpoint tracking do not exist yet.

### Task 2: Implement benchmark config and normalization utilities

**Files:**
- Modify: `src/rl_training/experiment/config.py`
- Create: `src/rl_training/experiment/benchmarking.py`
- Modify: `src/rl_training/cli.py`
- Modify: `src/rl_training/runtime/workflows.py`

**Step 1: Write minimal implementation**
- Add top-level `benchmark` config support to `TrainConfig`, config loading, serialization, and checkpoint restore.
- Implement score normalization helpers for human-random scaling.
- Augment `evaluate_checkpoint(...)` with normalized metrics when benchmark config exists.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_benchmarking.py tests/test_checkpoint_workflows.py`

### Task 3: Implement best checkpoint tracking in run utilities

**Files:**
- Modify: `src/rl_training/experiment/runs.py`
- Modify: `src/rl_training/runtime/run_utils.py`

**Step 1: Write minimal implementation**
- Track best checkpoint according to `benchmark.best_metric` / `benchmark.best_metric_mode`, defaulting to `eval_return_mean` / `max`.
- Save/update `checkpoints/best.pt`.
- Persist best-checkpoint metadata in `metadata.json`.
- Add best-checkpoint fields to the metrics dict returned by trainers.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_benchmarking.py tests/test_checkpoint_workflows.py`

### Task 4: Document benchmark config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Show benchmark config with `random_score`, `human_score`, and `best_metric`.
- Document `checkpoints/best.pt` and normalized eval metrics.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_benchmarking.py tests/test_checkpoint_workflows.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
