# DQN Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the first off-policy training path to the package by implementing a real replay buffer, a minimal DQN algorithm layer, and a smoke-tested DQN trainer.

**Architecture:** Reuse the existing package boundaries instead of creating a separate ad hoc DQN stack. The replay buffer lives under `data`, the DQN update math under `algorithms`, the Q-network under `models`, and the training orchestration under `runtime`, mirroring the PPO path while keeping off-policy concerns isolated.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Replace the replay buffer protocol with a concrete implementation

**Files:**
- Modify: `src/rl_training/data/replay_buffer.py`
- Create: `tests/test_replay_buffer.py`

**Step 1: Write the failing test**

- Verify add / sample behavior
- Verify state_dict / load_state_dict roundtrip

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_replay_buffer.py`
Expected: FAIL because `ReplayBuffer` is still a `Protocol`

**Step 3: Write minimal implementation**

- Tensor-backed ring buffer
- `add(...)`
- `sample(batch_size)`
- `__len__()`
- `state_dict()` / `load_state_dict()`

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_replay_buffer.py`
Expected: PASS

### Task 2: Add the DQN algorithm layer

**Files:**
- Create: `src/rl_training/models/mlp_q_network.py`
- Modify: `src/rl_training/models/__init__.py`
- Create: `src/rl_training/algorithms/dqn.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Create: `tests/test_dqn_update.py`

**Step 1: Write the failing test**

- Verify Q-network forward shape
- Verify `dqn_loss(...)` returns named metrics
- Verify `DQN.update(...)` returns `UpdateResult`

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_dqn_update.py`
Expected: FAIL because DQN modules do not exist

**Step 3: Write minimal implementation**

- Discrete-action MLP Q-network
- Pure `dqn_loss(...)`
- `DQN` class with optimizer step and target network sync helper

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_dqn_update.py`
Expected: PASS

### Task 3: Add a smoke-tested DQN training path

**Files:**
- Create: `src/rl_training/runtime/dqn_trainer.py`
- Create: `tests/test_dqn_trainer_smoke.py`
- Modify: `src/rl_training/cli.py`
- Create: `configs/dqn/cartpole.yaml`

**Step 1: Write the failing test**

- Verify `train_dqn(...)` writes a checkpoint and returns metrics

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_dqn_trainer_smoke.py`
Expected: FAIL because `train_dqn` does not exist

**Step 3: Write minimal implementation**

- Vector-env DQN training loop
- Epsilon-greedy action selection
- Warmup + replay sampling
- Periodic target-network sync
- Checkpoint, run metadata, and evaluation
- CLI train path support for `algo: dqn`

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_dqn_trainer_smoke.py`
Expected: PASS

### Task 4: Verify the new off-policy slice with the existing suite

**Files:**
- No new files

**Step 1: Run the focused tests**

Run: `pytest -q tests/test_replay_buffer.py tests/test_dqn_update.py tests/test_dqn_trainer_smoke.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
