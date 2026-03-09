# Experiment Manager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a real experiment registry and manager layer so the package has a stable high-level orchestration surface above raw trainer functions.

**Architecture:** Keep algorithm-specific training logic in the trainer modules, but introduce a small registry that maps algorithm names to train functions and a concrete experiment manager that returns trainer objects conforming to the existing `Trainer` protocol. Resume should reconstruct from checkpoints without hardcoding algorithm branches into every caller.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, PyTest

---

### Task 1: Add an algorithm registry

**Files:**
- Create: `src/rl_training/experiment/registry.py`
- Create: `tests/test_experiment_manager.py`

**Step 1: Write the failing test**

- Verify built-in algorithms `ppo`, `dqn`, and `sac` are registered

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_experiment_manager.py`
Expected: FAIL because registry does not exist

**Step 3: Write minimal implementation**

- `AlgorithmSpec` dataclass
- registry lookup for train functions

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_experiment_manager.py`
Expected: PASS

### Task 2: Add a concrete experiment manager

**Files:**
- Create: `src/rl_training/experiment/default_manager.py`
- Modify: `src/rl_training/experiment/__init__.py`
- Modify: `tests/test_experiment_manager.py`

**Step 1: Extend the failing test**

- Verify `setup(config).train()` runs the right algorithm
- Verify `resume(checkpoint).train()` advances training

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_experiment_manager.py`
Expected: FAIL because no concrete manager exists

**Step 3: Write minimal implementation**

- small trainer wrapper around a callable
- `DefaultExperimentManager.setup`
- `DefaultExperimentManager.resume`

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_experiment_manager.py`
Expected: PASS

### Task 3: Verify the package after the manager layer

**Files:**
- No new files

**Step 1: Run focused tests**

Run: `pytest -q tests/test_experiment_manager.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
