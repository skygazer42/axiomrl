# Training Callbacks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the callback protocol into a real runtime extension point so training code can emit lifecycle events across PPO, DQN, and SAC.

**Architecture:** Keep callback handling lightweight and shared. A runtime callback helper should fan out lifecycle events, while each trainer remains responsible for deciding what counts as collection, update, evaluation, and train end. Public API and experiment-manager entrypoints should accept optional callbacks and forward them down.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, PyTest

---

### Task 1: Add shared callback dispatch helpers

**Files:**
- Modify: `src/axiomrl/runtime/callbacks.py`
- Modify: `src/axiomrl/runtime/trainer.py`
- Create: `tests/test_callbacks.py`

**Step 1: Write the failing test**

- Verify callback lifecycle events can be recorded during training

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_callbacks.py`
Expected: FAIL because trainers do not accept or emit callbacks

**Step 3: Write minimal implementation**

- callback fanout helper
- small runtime trainer state object to hand to callbacks

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_callbacks.py`
Expected: PASS

### Task 2: Wire callbacks through trainers and public entrypoints

**Files:**
- Modify: `src/axiomrl/runtime/ppo_trainer.py`
- Modify: `src/axiomrl/runtime/dqn_trainer.py`
- Modify: `src/axiomrl/runtime/sac_trainer.py`
- Modify: `src/axiomrl/experiment/default_manager.py`
- Modify: `src/axiomrl/api/algorithms.py`

**Step 1: Use the failing callback test**

Run: `pytest -q tests/test_callbacks.py`
Expected: FAIL before wiring is complete

**Step 2: Write minimal implementation**

- optional `callbacks` parameter in train functions
- optional `callbacks` forwarding in manager and public API
- lifecycle events:
  - `on_train_start`
  - `on_collect_end`
  - `on_update_end`
  - `on_eval_end`
  - `on_train_end`

**Step 3: Run the callback test to verify it passes**

Run: `pytest -q tests/test_callbacks.py`
Expected: PASS

### Task 3: Verify the package after callback integration

**Files:**
- No new files

**Step 1: Run focused tests**

Run: `pytest -q tests/test_callbacks.py tests/test_public_api.py tests/test_experiment_manager.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
