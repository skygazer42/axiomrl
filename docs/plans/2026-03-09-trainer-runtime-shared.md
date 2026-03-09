# Trainer Runtime Shared Utilities Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract shared trainer runtime scaffolding so PPO, DQN, and SAC trainers stop duplicating device resolution, config serialization, run setup, and checkpoint finalization logic.

**Architecture:** Keep algorithm-specific data collection and update loops in each trainer, but move package-level run management into a shared runtime utility module. The new shared layer should only own generic concerns: run directories, config metadata, logging setup, device resolution, and checkpoint writing.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Add a shared runtime utilities module

**Files:**
- Create: `src/rl_training/runtime/run_utils.py`
- Create: `tests/test_run_utils.py`

**Step 1: Write the failing test**

- Verify config serialization normalizes `Path` and `tags`
- Verify run setup writes config and metadata files
- Verify checkpoint finalization writes a loadable checkpoint

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_run_utils.py`
Expected: FAIL because the shared runtime helpers do not exist

**Step 3: Write minimal implementation**

- `resolve_device(...)`
- `serialize_train_config(...)`
- `create_training_run(...)`
- `save_training_checkpoint(...)`

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_run_utils.py`
Expected: PASS

### Task 2: Move PPO, DQN, and SAC trainers onto the shared utilities

**Files:**
- Modify: `src/rl_training/runtime/ppo_trainer.py`
- Modify: `src/rl_training/runtime/dqn_trainer.py`
- Modify: `src/rl_training/runtime/sac_trainer.py`

**Step 1: Use the existing smoke tests**

Run: `pytest -q tests/test_trainer_smoke.py tests/test_dqn_trainer_smoke.py tests/test_sac_trainer_smoke.py`
Expected: PASS before refactor

**Step 2: Refactor trainers**

- Remove duplicated helper functions now owned by `run_utils`
- Keep algorithm-specific environment and update logic intact

**Step 3: Run the smoke tests to verify behavior stays green**

Run: `pytest -q tests/test_trainer_smoke.py tests/test_dqn_trainer_smoke.py tests/test_sac_trainer_smoke.py`
Expected: PASS

### Task 3: Verify the package after the refactor

**Files:**
- No new files

**Step 1: Run focused shared-runtime tests**

Run: `pytest -q tests/test_run_utils.py tests/test_trainer_smoke.py tests/test_dqn_trainer_smoke.py tests/test_sac_trainer_smoke.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
