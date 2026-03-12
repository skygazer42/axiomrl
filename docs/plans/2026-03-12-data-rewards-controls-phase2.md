# Data, Rewards, And Training Controls Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `rl_training` materially more usable for real training by adding external offline dataset loading, generic reward transforms, and early-stopping/evaluation controls before the next major algorithm wave lands.

**Architecture:** Keep backward compatibility with the current config layout by extending `algo_kwargs` and `env_kwargs.wrappers` instead of inventing a second config system. Build shared modules first, then reuse them from existing offline trainers and env factories so later algorithms such as `BC`, `AWAC`, and `HER` can plug into the same infrastructure.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, optional Minari integration, pytest, setuptools

---

### Task 1: Add external transition dataset loading and reward processing

**Files:**
- Modify: `src/rl_training/data/offline_dataset.py`
- Create: `src/rl_training/data/dataset_loaders.py`
- Modify: `src/rl_training/data/__init__.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/runtime/cql_trainer.py`
- Modify: `src/rl_training/runtime/td3_bc_trainer.py`
- Create: `tests/test_dataset_loaders.py`
- Modify: `tests/test_offline_dataset.py`

**Step 1: Write the failing tests**

Add coverage for:

- `TransitionDataset.from_dict(...)`
- loading transitions from `.npz`
- loading transitions from `.pt`
- reward scaling / shifting / clipping for offline datasets
- offline trainers accepting `dataset_kind: npz` and `dataset_kind: pt`

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `TransitionDataset.from_dict(...)`
- `TransitionDataset.with_reward_transform(...)`
- `load_transition_dataset(...)`
- support for `dataset_kind in {\"random\", \"npz\", \"pt\", \"minari\"}`
- config keys such as `dataset_path`, `dataset_id`, `dataset_download`, `reward_scale`, `reward_shift`, `reward_clip_min`, `reward_clip_max`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 2: Add generic reward transform wrappers to env creation

**Files:**
- Create: `src/rl_training/envs/rewards.py`
- Modify: `src/rl_training/envs/factory.py`
- Modify: `src/rl_training/envs/__init__.py`
- Create: `tests/test_reward_wrappers.py`
- Modify: `tests/test_envs.py`

**Step 1: Write the failing tests**

Add coverage for:

- reward scale wrappers
- reward shift wrappers
- reward clip wrappers
- wrapper composition via `env_kwargs.wrappers.reward`
- non-reward-wrapper env flows remaining unchanged

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `RewardTransformConfig`
- `RewardScaleWrapper`
- `RewardShiftWrapper`
- `RewardClipWrapper`
- `apply_reward_wrappers(...)`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 3: Add evaluation cadence and early stopping controls for offline trainers

**Files:**
- Modify: `src/rl_training/runtime/trainer.py`
- Modify: `src/rl_training/runtime/callbacks.py`
- Create: `src/rl_training/runtime/controls.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/runtime/cql_trainer.py`
- Modify: `src/rl_training/runtime/td3_bc_trainer.py`
- Create: `tests/test_training_controls.py`
- Modify: `tests/test_callbacks.py`

**Step 1: Write the failing tests**

Add coverage for:

- `EarlyStoppingCallback` stopping on no improvement
- stopping on reward threshold
- offline trainers evaluating every `eval_interval`
- trainer state carrying stop flags / stop reason

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `TrainerState.should_stop`
- `TrainerState.stop_reason`
- `EarlyStoppingConfig`
- `EarlyStoppingCallback`
- offline trainer `eval_interval`
- break conditions after callback-triggered stop requests

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 4: Package and docs polish for the new runtime surface

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_package_smoke.py`

**Step 1: Write the failing tests**

Add or extend coverage so it asserts:

- optional Minari/offline installation guidance exists
- README documents `dataset_kind`, `reward` wrappers, and early stopping controls
- package metadata remains coherent

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- optional dependency metadata for offline dataset loading
- concise README examples for offline dataset paths and reward wrappers

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 5: Next algorithm wave after the infrastructure lands

**Files:**
- Create in later follow-up plan

**Step 1: Prepare the next plan**

Immediately after Tasks 1-4 land, write a follow-up plan for:

- `BC`
- `AWAC`
- `HER`

These three should be the first algorithm wave on top of the new infrastructure.
