# BCQ, BEAR, And Offline Controls Phase 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `BCQ` and `BEAR` as the next mainstream offline RL baselines while strengthening offline schedule, budget, and dataset-mixing controls so the package can support real batch-RL training workflows instead of one-off trainers.

**Architecture:** Reuse the existing offline dataset path and checkpoint/runtime surfaces rather than inventing a second offline stack. Build the shared offline control layer first, then land `BCQ` and `BEAR` on top of it so both algorithms share dataset mixing, evaluation cadence, early stopping, and budget rules. Keep the first slice narrow: MLP continuous-control support first, no sequence models, no distributed collection, no visual BCQ/BEAR in this phase.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, optional Minari integration, pytest, setuptools

---

### Task 1: Add shared offline schedule and budget utilities

**Files:**
- Create: `src/rl_training/runtime/schedules.py`
- Modify: `src/rl_training/runtime/controls.py`
- Modify: `src/rl_training/runtime/bc_trainer.py`
- Modify: `src/rl_training/runtime/awac_trainer.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/runtime/cql_trainer.py`
- Modify: `src/rl_training/runtime/td3_bc_trainer.py`
- Create: `tests/test_training_schedules.py`
- Modify: `tests/test_training_controls.py`

**Step 1: Write the failing tests**

Add coverage for:

- schedule specs such as `constant`, `linear`, and `piecewise`
- offline trainers respecting `max_updates`
- offline trainers respecting `max_dataset_passes`
- schedule-driven scalar resolution from `algo_kwargs`
- early stopping and eval cadence remaining compatible with the new budget layer

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- schedule parsing helpers such as `resolve_schedule_value(...)`
- offline budget helpers such as `resolve_max_updates(...)`
- config keys such as `max_updates`, `max_dataset_passes`, `warmup_updates`
- trainer loop guards so offline algorithms can stop on budget exhaustion instead of raw `total_timesteps` only

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 2: Add shared offline dataset mixing and reward-preset support

**Files:**
- Modify: `src/rl_training/data/offline_dataset.py`
- Modify: `src/rl_training/data/dataset_loaders.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/runtime/cql_trainer.py`
- Modify: `src/rl_training/runtime/td3_bc_trainer.py`
- Modify: `src/rl_training/runtime/awac_trainer.py`
- Modify: `src/rl_training/runtime/bc_trainer.py`
- Create: `src/rl_training/assets/configs/rewards/continuous_control.yaml`
- Create: `tests/test_offline_dataset_mixing.py`
- Modify: `tests/test_dataset_loaders.py`

**Step 1: Write the failing tests**

Add coverage for:

- mixing two offline datasets with explicit weights
- reward preset loading for common continuous-control scaling/clipping
- preserving action / observation shape validation across mixed datasets
- trainers consuming mixed datasets through the same `_build_offline_dataset(...)` path

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `TransitionDataset.concat(...)` or equivalent dataset-composition helper
- `dataset_kind: mixed` or `dataset_sources` config support
- reward preset lookup that resolves scale / shift / clip defaults from packaged YAML
- trainer-side loading so offline algorithms can reuse mixed data and reward presets without custom code

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 3: Add the BCQ algorithm, model, and trainer

**Files:**
- Create: `src/rl_training/models/mlp_bcq.py`
- Modify: `src/rl_training/models/__init__.py`
- Create: `src/rl_training/algorithms/bcq.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Create: `src/rl_training/runtime/bcq_trainer.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/bcq/pendulum.yaml`
- Create: `src/rl_training/assets/configs/bcq/pendulum.yaml`
- Create: `tests/test_bcq_update.py`
- Create: `tests/test_bcq_trainer_smoke.py`

**Step 1: Write the failing tests**

Add coverage for:

- `bcq_loss(...)` exposing stable metric names
- invalid BCQ hyperparameters failing fast
- BCQ action-sampling shape and range invariants
- BCQ trainer writing checkpoints and evaluation metrics
- registry / public API surfacing `bcq`

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- BCQ model components: actor perturbation network, twin critics, action VAE
- `BCQ` update rule with candidate-action sampling and constrained perturbation
- offline `train_bcq(...)` trainer reusing shared offline budget / dataset controls
- packaged config and public API integration for `bcq`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 4: Add the BEAR algorithm and trainer on the same offline support layer

**Files:**
- Create: `src/rl_training/algorithms/bear.py`
- Create: `src/rl_training/runtime/bear_trainer.py`
- Modify: `src/rl_training/models/mlp_bcq.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/bear/pendulum.yaml`
- Create: `src/rl_training/assets/configs/bear/pendulum.yaml`
- Create: `tests/test_bear_update.py`
- Create: `tests/test_bear_trainer_smoke.py`

**Step 1: Write the failing tests**

Add coverage for:

- `bear_loss(...)` exposing stable metric names
- BEAR MMD or support-matching constraint metrics being logged
- invalid BEAR hyperparameters failing fast
- trainer checkpoint / evaluation behavior
- registry / public API surfacing `bear`

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- BEAR algorithm reusing the generative behavior-policy support from the BCQ model family where practical
- MMD or equivalent support-matching penalty with configurable kernel settings
- offline `train_bear(...)` trainer reusing the same dataset, control, and evaluation layer as BCQ
- packaged config and public API integration for `bear`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 5: Product surface, docs, and checkpoint workflow polish

**Files:**
- Modify: `README.md`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`

**Step 1: Write the failing tests**

Add or extend coverage so it asserts:

- `BCQ` and `BEAR` are exported through package APIs
- packaged configs include `bcq` and `bear`
- checkpoint evaluation / resume workflows cover the new offline algorithms
- README documents offline schedule controls, dataset mixing, reward presets, and the new batch-RL baselines

**Step 2: Run tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- README examples for `bcq`, `bear`, mixed datasets, and offline budget rules
- roadmap note that the package is entering the canonical batch-RL wave
- CLI / package-surface polish where needed

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.
