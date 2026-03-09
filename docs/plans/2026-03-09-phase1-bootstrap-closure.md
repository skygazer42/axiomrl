# Phase 1 Bootstrap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the current Phase 1 bootstrap gaps so the package imports under pytest, exposes a real environment factory, and ships a concrete rollout buffer implementation.

**Architecture:** Keep the existing contract-oriented package layout, but replace missing or placeholder pieces with minimal concrete implementations that match the tests and the longer-term module boundaries. The work stays narrowly scoped to bootstrap infrastructure and PPO-adjacent runtime pieces.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, PyTest

---

### Task 1: Fix local test imports for the `src` layout

**Files:**
- Create: `tests/conftest.py`

**Step 1: Use the existing failing smoke test**

Run: `pytest -q tests/test_package_smoke.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_training'`

**Step 2: Write minimal test harness support**

- Insert `<repo>/src` into `sys.path` from `tests/conftest.py`

**Step 3: Run test to verify it passes**

Run: `pytest -q tests/test_package_smoke.py`
Expected: PASS

### Task 2: Add the environment factory package

**Files:**
- Create: `src/rl_training/envs/__init__.py`
- Create: `src/rl_training/envs/factory.py`
- Modify: `pyproject.toml`

**Step 1: Use the existing failing env test**

Run: `pytest -q tests/test_envs.py`
Expected: FAIL because `rl_training.envs.factory` does not exist

**Step 2: Write minimal implementation**

- Add an `envs` package
- Implement `make_env(config, env_index)` and `make_vector_env(config)`
- Use `gymnasium.make(...)` plus `RecordEpisodeStatistics`
- Read `config.num_envs` and `config.env_kwargs`
- Add `gymnasium` to project dependencies

**Step 3: Run test to verify it passes**

Run: `pytest -q tests/test_envs.py`
Expected: PASS

### Task 3: Replace the rollout buffer protocol with a concrete PPO buffer

**Files:**
- Modify: `src/rl_training/data/rollout_buffer.py`

**Step 1: Use the existing failing rollout buffer tests**

Run: `PYTHONPATH=src pytest -q tests/test_rollout_buffer.py`
Expected: FAIL because `RolloutBuffer` is a `Protocol` and cannot be instantiated

**Step 2: Write minimal implementation**

- Implement a concrete tensor-backed `RolloutBuffer`
- Allocate `obs`, `actions`, `rewards`, `dones`, `values`, `logprobs`,
  `advantages`, and `returns`
- Implement `reset()`
- Implement `compute_returns_and_advantages(...)` with GAE
- Implement `iter_minibatches(...)` by flattening step and env dimensions

**Step 3: Run test to verify it passes**

Run: `pytest -q tests/test_rollout_buffer.py`
Expected: PASS

### Task 4: Verify the bootstrap slice together

**Files:**
- No new files

**Step 1: Run the focused Phase 1 bootstrap tests**

Run: `pytest -q tests/test_package_smoke.py tests/test_module_contracts.py tests/test_experiment_contracts.py tests/test_envs.py tests/test_rollout_buffer.py`
Expected: PASS
