# AWAC And Online Controls Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a mainstream `AWAC` offline actor-critic baseline and unify evaluation cadence / early-stopping controls across the online off-policy trainers so `rl_training` behaves more like a coherent training package.

**Architecture:** Reuse the existing continuous-control stack instead of inventing a second offline runtime. `AWAC` should sit beside `IQL`, `CQL`, `TD3+BC`, reuse the external dataset pipeline, and evaluate through the same continuous-action policy helpers. Online trainers should adopt the existing callback-based control surface so `eval_interval` and `early_stopping` work consistently across `DQN`, `DDPG`, `SAC`, `TD3`, `REDQ`, and `TQC`.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, optional Minari integration, pytest, setuptools

---

### Task 1: Add the AWAC algorithm and trainer

**Files:**
- Modify: `src/rl_training/models/mlp_sac.py`
- Create: `src/rl_training/algorithms/awac.py`
- Create: `src/rl_training/runtime/awac_trainer.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/models/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/awac/pendulum.yaml`
- Create: `src/rl_training/assets/configs/awac/pendulum.yaml`
- Create: `tests/test_awac_update.py`
- Create: `tests/test_awac_trainer_smoke.py`

**Step 1: Write the failing tests**

Add coverage for:

- `awac_loss(...)` exposing stable metric names
- invalid AWAC hyperparameters failing fast
- offline AWAC trainer producing checkpoints and evaluation metrics
- registry / public API surfacing `awac`

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `MLPSACModel.action_logprobs(...)`
- `AWAC` algorithm with weighted behavior-cloning actor loss and twin-critic TD learning
- offline `train_awac(...)` trainer reusing `_build_offline_dataset(...)`
- registry / API / packaged config integration for `awac`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 2: Unify eval cadence and early stopping for online off-policy trainers

**Files:**
- Modify: `src/rl_training/runtime/dqn_trainer.py`
- Modify: `src/rl_training/runtime/ddpg_trainer.py`
- Modify: `src/rl_training/runtime/sac_trainer.py`
- Modify: `src/rl_training/runtime/td3_trainer.py`
- Modify: `src/rl_training/runtime/redq_trainer.py`
- Modify: `src/rl_training/runtime/tqc_trainer.py`
- Modify: `tests/test_training_controls.py`
- Modify: `tests/test_callbacks.py`

**Step 1: Write the failing tests**

Add coverage for:

- `eval_interval` on online trainers
- callback-triggered early stopping breaking training loops
- checkpoint trainer state preserving `should_stop` and `stop_reason`

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- `build_control_callbacks(...)` usage to online off-policy trainers
- periodic evaluation inside the training loop
- stop-aware checkpoint state for online trainers

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 3: Product surface and roadmap polish

**Files:**
- Modify: `README.md`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`

**Step 1: Write the failing tests**

Add or extend coverage so it asserts:

- `AWAC` is exported through package APIs
- packaged configs include `awac`
- README documents `awac` and online `eval_interval` / `early_stopping`
- roadmap doc reflects the current implementation order

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- README examples for `awac`
- roadmap note that `BC` and `AWAC` are now entering the package surface
- package metadata/docs references where needed

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.
