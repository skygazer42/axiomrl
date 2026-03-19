# Multi-Seed Benchmark Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class multi-seed benchmark runner that reuses the new runtime foundation, executes repeatable seed sweeps from one config, and emits aggregate benchmark summaries.

**Architecture:** Keep single-run training behavior unchanged, but let the experiment layer detect a benchmark seed sweep and fan it out into multiple ordinary training runs. Reuse existing per-algorithm `train_fn` entrypoints, `TrainingSession`, and `FunctionRunner`, then aggregate numeric metrics into one benchmark summary artifact so the CLI and experiment manager can orchestrate benchmark-quality runs without custom shell scripting.

**Tech Stack:** Python 3.10+, existing `TrainConfig`/`ExperimentManager`/`Runner` abstractions, JSON artifacts, pytest.

---

### Task 1: Add failing seed-sweep tests

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- In `tests/test_runner.py`, add a runner-level test for a new `BenchmarkRunner` that executes child runners for seeds `[3, 5]`, returns aggregate metrics like `benchmark_run_count`, `eval_return_mean_mean`, and `eval_return_mean_std`, and preserves per-run metadata in the summary payload.
- In `tests/test_experiment_manager.py`, add a manager test asserting `DefaultExperimentManager.setup_runner(...)` returns a benchmark-aware runner when `config.benchmark["seeds"]` is set, but still returns the normal function runner for ordinary configs.
- In `tests/test_cli.py`, add a CLI test that runs `axiomrl train --config ... --seeds 11,13`, verifies two run directories are created, and verifies a benchmark summary JSON artifact is written under the configured output root.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_runner.py tests/test_experiment_manager.py tests/test_cli.py`

Expected: FAIL because seed-sweep orchestration and benchmark summary artifacts do not exist yet.

### Task 2: Add seed-sweep planning and aggregate metric helpers

**Files:**
- Create: `src/rl_training/experiment/sweeps.py`
- Modify: `src/rl_training/experiment/benchmarking.py`
- Modify: `src/rl_training/experiment/config.py`

**Step 1: Write minimal implementation**
- Create `src/rl_training/experiment/sweeps.py` with small dataclasses/helpers such as `SeedSweepPlan`, `BenchmarkRunRecord`, and `resolve_benchmark_seeds(config: TrainConfig) -> tuple[int, ...]`.
- Normalize `benchmark["seeds"]` into a validated tuple of distinct integers and reject empty or malformed seed lists with explicit errors.
- Add aggregate helpers in `src/rl_training/experiment/benchmarking.py` that compute `*_mean`, `*_std`, `*_min`, and `*_max` for numeric metrics shared across run results.
- Add a tiny helper in `src/rl_training/experiment/config.py` if needed to expose validated benchmark seed settings without expanding the dataclass surface more than necessary.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_runner.py tests/test_experiment_manager.py`

Expected: still FAIL, but now only because the runner/manager/CLI wiring is not complete.

### Task 3: Implement `BenchmarkRunner` on top of the runtime foundation

**Files:**
- Modify: `src/rl_training/runtime/runner.py`
- Modify: `src/rl_training/experiment/default_manager.py`
- Modify: `src/rl_training/experiment/manager.py`

**Step 1: Write minimal implementation**
- Extend `src/rl_training/runtime/runner.py` with a `BenchmarkRunner` that accepts a `SeedSweepPlan`, a child runner factory, and an output path for aggregate artifacts.
- Have `BenchmarkRunner.run()` execute one ordinary child runner per seed, collect each `TrainResult`, aggregate numeric metrics, and return a synthetic `TrainResult` whose `metrics` contains both aggregate values and a `benchmark_run_count`.
- Persist a machine-readable benchmark summary JSON file that includes per-seed run directories, checkpoint paths, and aggregate metrics.
- Update `DefaultExperimentManager.setup_runner(...)` so it chooses `BenchmarkRunner` when benchmark seeds are configured; otherwise it keeps returning `FunctionRunner`.
- Update the `ExperimentManager` protocol signatures only as needed to keep callback support and type hints aligned with the new runner path.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_runner.py tests/test_experiment_manager.py`

Expected: PASS.

### Task 4: Expose seed sweeps through the CLI

**Files:**
- Modify: `src/rl_training/cli.py`
- Modify: `src/rl_training/runtime/workflows.py`

**Step 1: Write minimal implementation**
- Add a `--seeds` option to `axiomrl train` that accepts a comma-separated list like `--seeds 1,2,3`.
- Parse the option into `config.benchmark["seeds"]` inside `_apply_overrides(...)` so YAML-driven and CLI-driven seed sweeps share the same plumbing.
- Keep `resume` as single-run only for now; reject `--seeds` there if you expose it accidentally.
- Make `_print_result(...)` continue to work for benchmark runs by printing the aggregate metrics and the benchmark summary path from the returned synthetic result.

**Step 2: Run focused tests**

Run: `pytest -q tests/test_cli.py tests/test_runner.py tests/test_experiment_manager.py`

Expected: PASS.

### Task 5: Document the benchmark-runner workflow

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`

**Step 1: Add docs**
- Add one example showing a single config executed with `axiomrl train --seeds 1,2,3`.
- Explain that this creates ordinary per-seed run directories plus one aggregate benchmark summary artifact.
- Clarify that this is the recommended path for benchmark-quality comparisons before feeding runs into `axiomrl zoo`.

### Task 6: Verification

**Run:**
- Focused: `pytest -q tests/test_runner.py tests/test_experiment_manager.py tests/test_cli.py`
- Broader: `pytest -q tests/test_benchmarking.py tests/test_checkpoint_workflows.py`
- Full regression: `pytest -q`

**Notes:**
- Keep the first version serial and deterministic; do not add nested parallel execution inside the benchmark runner yet.
- Reuse the new runtime foundation instead of teaching algorithms any sweep-specific behavior.
- Return aggregate metrics only for numeric keys shared across all child runs; keep raw per-run details in the summary artifact instead of bloating `TrainResult.metrics`.
