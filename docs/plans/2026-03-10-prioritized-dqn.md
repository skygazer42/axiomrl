# Prioritized DQN (PER) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `prioritized_dqn` as a first-class DQN-family algorithm using Prioritized Experience Replay (PER), with configs, examples, registry wiring, public API exports, and tests.

**Architecture:** Introduce `PrioritizedReplayBuffer` in `rl_training.data` and extend the existing DQN update to support optional importance-sampling weights and expose per-batch TD errors for priority updates. Keep the existing `train_dqn` trainer entrypoint and registry evaluation/prediction flow, but select the prioritized buffer when `config.algo == "prioritized_dqn"`.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, pytest

---

### Task 1: Add failing PER coverage

**Files:**
- Modify: `tests/test_dqn_trainer_smoke.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_package_api_exports.py`
- Create: `tests/test_prioritized_replay_buffer.py`
- Create: `tests/test_prioritized_dqn_reference_script.py`

**Step 1: Write the failing test**

Add tests that expect:
- `PrioritizedReplayBuffer` exists, can add/sample, and supports `update_priorities()`
- registry can train `algo: prioritized_dqn` and produces checkpoints
- CLI `train --config` supports `algo: prioritized_dqn`
- managed API exports `PrioritizedDQN`
- reference script runs successfully

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prioritized_replay_buffer.py tests/test_dqn_trainer_smoke.py tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_prioritized_dqn_reference_script.py -q`

Expected: FAIL because the buffer, wiring, exports, and scripts do not exist yet.

### Task 2: Implement PrioritizedReplayBuffer + weighted DQN loss

**Files:**
- Create: `src/rl_training/data/prioritized_replay_buffer.py`
- Modify: `src/rl_training/data/__init__.py`
- Modify: `src/rl_training/algorithms/dqn.py`
- Modify: `src/rl_training/runtime/dqn_trainer.py`

**Step 1: Write minimal implementation**

Implement:
- `PrioritizedReplayBuffer.sample(batch_size, beta=...) -> batch + indices + weights`
- `PrioritizedReplayBuffer.update_priorities(indices, priorities)`
- DQN update supports optional `weights` and stores `last_td_errors` for priority updates
- `train_dqn` uses PER buffer only for `prioritized_dqn`

**Step 2: Run focused tests**

Run: `pytest tests/test_prioritized_replay_buffer.py tests/test_dqn_trainer_smoke.py::test_train_prioritized_dqn_writes_checkpoint_and_metrics -q`

Expected: PASS

### Task 3: Wire registry/API/configs/examples + verify

**Files:**
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Create: `configs/prioritized_dqn/cartpole.yaml`
- Create: `examples/prioritized_dqn_cartpole_reference.py`
- Modify: `README.md`

**Step 1: Write minimal implementation**

Expose:
- algorithm spec `prioritized_dqn` and checkpoint load support
- managed API class `PrioritizedDQN`
- config + reference script for CartPole
- README mention in algorithm list

**Step 2: Verify**

Run targeted: `pytest tests/test_prioritized_replay_buffer.py tests/test_dqn_trainer_smoke.py tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_prioritized_dqn_reference_script.py -q`

Run full: `pytest -q`

Expected: PASS
