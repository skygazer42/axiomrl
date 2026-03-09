# A2C and TD3 Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two more real RL algorithms to the package by implementing A2C and TD3 end-to-end enough to train, evaluate, checkpoint, and use through the existing public API and registry.

**Architecture:** Reuse the current family boundaries instead of inventing new ones. A2C should reuse the actor-critic model and rollout buffer path from PPO with a simpler unclipped on-policy update. TD3 should reuse the replay-buffer-based continuous-control path from SAC while introducing deterministic actor updates, target policy smoothing, and delayed actor updates.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Add the A2C algorithm layer and smoke-tested trainer

**Files:**
- Create: `src/rl_training/algorithms/a2c.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Create: `src/rl_training/runtime/a2c_trainer.py`
- Create: `tests/test_a2c_update.py`
- Create: `tests/test_a2c_trainer_smoke.py`

**Step 1: Write the failing tests**

- verify `a2c_loss(...)` returns named metrics
- verify `A2C.update(...)` returns `UpdateResult`
- verify `train_a2c(...)` writes a checkpoint and returns metrics

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_a2c_update.py tests/test_a2c_trainer_smoke.py`
Expected: FAIL because A2C modules do not exist

**Step 3: Write minimal implementation**

- unclipped actor-critic update
- rollout-buffer-driven trainer
- evaluation, checkpointing, and callback support

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_a2c_update.py tests/test_a2c_trainer_smoke.py`
Expected: PASS

### Task 2: Add the TD3 algorithm layer and smoke-tested trainer

**Files:**
- Create: `src/rl_training/models/mlp_td3.py`
- Modify: `src/rl_training/models/__init__.py`
- Create: `src/rl_training/algorithms/td3.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Create: `src/rl_training/runtime/td3_trainer.py`
- Create: `tests/test_td3_update.py`
- Create: `tests/test_td3_trainer_smoke.py`

**Step 1: Write the failing tests**

- verify deterministic actor output is bounded
- verify `td3_loss(...)` returns named metrics
- verify `TD3.update(...)` returns `UpdateResult`
- verify `train_td3(...)` writes a checkpoint and returns metrics

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_td3_update.py tests/test_td3_trainer_smoke.py`
Expected: FAIL because TD3 modules do not exist

**Step 3: Write minimal implementation**

- deterministic actor + twin critics
- target policy smoothing
- delayed actor updates
- replay-buffer-driven continuous trainer

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_td3_update.py tests/test_td3_trainer_smoke.py`
Expected: PASS

### Task 3: Wire A2C and TD3 into registry, CLI, public API, configs, and reference scripts

**Files:**
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/cli.py`
- Modify: `src/rl_training/api/algorithms.py`
- Create: `configs/a2c/cartpole.yaml`
- Create: `configs/td3/pendulum.yaml`
- Create: `examples/a2c_cartpole_reference.py`
- Create: `examples/td3_pendulum_reference.py`
- Create: `tests/test_a2c_reference_script.py`
- Create: `tests/test_td3_reference_script.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`

**Step 1: Extend failing tests**

- verify registry exposes `a2c` and `td3`
- verify public API supports `A2C` and `TD3`
- verify reference scripts smoke run

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_public_api.py tests/test_experiment_manager.py tests/test_a2c_reference_script.py tests/test_td3_reference_script.py`
Expected: FAIL because new algorithms are not wired in

**Step 3: Write minimal implementation**

- add registry specs
- add public API wrappers
- add CLI train support
- add basic configs and reference scripts

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_public_api.py tests/test_experiment_manager.py tests/test_a2c_reference_script.py tests/test_td3_reference_script.py`
Expected: PASS

### Task 4: Verify the package after algorithm expansion

**Files:**
- No new files

**Step 1: Run focused expansion tests**

Run: `pytest -q tests/test_a2c_update.py tests/test_a2c_trainer_smoke.py tests/test_td3_update.py tests/test_td3_trainer_smoke.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
