# SAC Training Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the first continuous-control training path to the package by implementing a smoke-tested SAC trainer, CLI support, checkpoint workflows, and a reference script.

**Architecture:** Reuse the replay-buffer-based off-policy runtime style introduced by DQN, but keep the continuous-action specifics inside the SAC trainer and checkpoint reconstruction helpers. Action normalization should remain in model space while environment stepping uses scaled actions derived from the environment action bounds.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Add a smoke-tested SAC training path

**Files:**
- Create: `src/axiomrl/runtime/sac_trainer.py`
- Create: `tests/test_sac_trainer_smoke.py`

**Step 1: Write the failing test**

- Verify `train_sac(...)` writes a checkpoint and returns metrics

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_sac_trainer_smoke.py`
Expected: FAIL because `train_sac` does not exist

**Step 3: Write minimal implementation**

- continuous-action scaling for environment stepping
- replay buffer driven training loop
- checkpoint, run metadata, and evaluation

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_sac_trainer_smoke.py`
Expected: PASS

### Task 2: Wire SAC into workflows and CLI

**Files:**
- Modify: `src/axiomrl/runtime/workflows.py`
- Modify: `src/axiomrl/cli.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_cli.py`
- Create: `configs/sac/pendulum.yaml`

**Step 1: Extend failing tests**

- Verify `evaluate_checkpoint(...)` supports SAC
- Verify CLI `train` works with `algo: sac`

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_checkpoint_workflows.py tests/test_cli.py`
Expected: FAIL because SAC is not wired into workflows or CLI

**Step 3: Write minimal implementation**

- reconstruct SAC from checkpoint config
- add `algo: sac` train dispatch
- add default SAC config preset

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_checkpoint_workflows.py tests/test_cli.py`
Expected: PASS

### Task 3: Add a reference script for continuous control

**Files:**
- Create: `examples/sac_pendulum_reference.py`
- Create: `tests/test_sac_reference_script.py`

**Step 1: Write the failing test**

- Verify the SAC reference script smoke runs

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_sac_reference_script.py`
Expected: FAIL because the script does not exist

**Step 3: Write minimal implementation**

- short Pendulum SAC entry script using package internals

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_sac_reference_script.py`
Expected: PASS

### Task 4: Verify the package with the continuous-control path

**Files:**
- No new files

**Step 1: Run focused SAC tests**

Run: `pytest -q tests/test_sac_update.py tests/test_sac_trainer_smoke.py tests/test_sac_reference_script.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
