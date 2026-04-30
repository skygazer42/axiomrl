# BCQ, BEAR, And Offline Runtime Phase 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the next serious offline RL wave to `axiomrl` by landing `BCQ`, `BEAR`, richer offline data processing, reward preset loading, and schedule / budget controls that make the package usable for longer training runs.

**Architecture:** Reuse the existing `offline_dataset + trainer + registry + managed API` surface instead of inventing a second runtime. Land the missing shared offline infrastructure first, then add `BCQ`, then `BEAR`, and only after that widen the public package surface. Keep the first release narrow: continuous-control offline RL first, flat vector observations first, and no distributed learners in this phase.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, optional Minari integration, pytest, setuptools

## Status Snapshot

As of **March 12, 2026**, this phase has now been materially executed in the
repository:

- shared offline data mixing is present
- reward presets and training schedule / budget controls are present
- `BCQ` is wired through model, algorithm, trainer, registry, managed API,
  root exports, configs, and package-surface tests
- `BEAR` is wired through the same package surfaces

Testing remains intentionally deferred in this phase until the user explicitly
allows test execution.

---

### Task 1: Add offline dataset mixing, trajectory slicing, and reward preset loading

**Files:**
- Create: `src/axiomrl/data/offline_mixers.py`
- Modify: `src/axiomrl/data/offline_dataset.py`
- Modify: `src/axiomrl/data/dataset_loaders.py`
- Modify: `src/axiomrl/data/__init__.py`
- Modify: `src/axiomrl/envs/rewards.py`
- Modify: `src/axiomrl/envs/__init__.py`
- Create: `tests/test_offline_mixers.py`
- Modify: `tests/test_dataset_loaders.py`
- Modify: `tests/test_reward_wrappers.py`

**Step 1: Write the failing tests**

Add coverage for:

- mixing two transition datasets by explicit ratio
- deterministic sampling with a mixer seed
- optional trajectory-window slicing for sequence-style offline batches
- named reward presets such as `sign_clip`, `clip_1`, and `sparse_goal_zero_one`
- backward compatibility with the current explicit `scale/shift/clip` wrapper config

**Step 2: Run focused tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `mix_transition_datasets(...)`
- `sample_trajectory_windows(...)`
- loader support for `algo_kwargs.dataset_mix`
- reward config support for `env_kwargs.wrappers.reward.preset`
- preset resolution that still composes with explicit scale / shift / clip values

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

**Step 5: Commit**

Use a focused commit after the shared offline data and reward surface lands.

### Task 2: Add schedule and budget controls for offline and off-policy trainers

**Files:**
- Create: `src/axiomrl/runtime/schedules.py`
- Modify: `src/axiomrl/runtime/controls.py`
- Modify: `src/axiomrl/runtime/bc_trainer.py`
- Modify: `src/axiomrl/runtime/awac_trainer.py`
- Modify: `src/axiomrl/runtime/iql_trainer.py`
- Modify: `src/axiomrl/runtime/cql_trainer.py`
- Modify: `src/axiomrl/runtime/td3_bc_trainer.py`
- Modify: `src/axiomrl/runtime/ddpg_trainer.py`
- Modify: `src/axiomrl/runtime/sac_trainer.py`
- Modify: `src/axiomrl/runtime/td3_trainer.py`
- Modify: `src/axiomrl/runtime/redq_trainer.py`
- Modify: `src/axiomrl/runtime/tqc_trainer.py`
- Modify: `src/axiomrl/runtime/her_trainer.py`
- Create: `tests/test_schedules.py`
- Modify: `tests/test_training_controls.py`

**Step 1: Write the failing tests**

Add coverage for:

- linear warmup and cosine decay schedule resolution
- constant schedule compatibility with current config behavior
- `max_updates`, `max_epochs`, and `min_buffer_size` budget guards
- online trainers respecting `warmup_steps` without updating early
- offline trainers stopping when `max_epochs` or `max_updates` is reached

**Step 2: Run focused tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `ScheduleSpec` and `resolve_schedule_value(...)`
- config keys such as `learning_rate_schedule`, `warmup_steps`, `max_updates`, and `max_epochs`
- trainer helpers that centralize update budgets and warmup checks
- metrics that expose `epoch`, `update_count`, and the resolved learning-rate multiplier

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

**Step 5: Commit**

Commit the schedule and trainer-control layer separately from the algorithm wave.

### Task 3: Add `BCQ` as the first constrained offline actor baseline

**Files:**
- Create: `src/axiomrl/models/mlp_bcq.py`
- Modify: `src/axiomrl/models/__init__.py`
- Create: `src/axiomrl/algorithms/bcq.py`
- Create: `src/axiomrl/runtime/bcq_trainer.py`
- Modify: `src/axiomrl/algorithms/__init__.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/bcq/pendulum.yaml`
- Create: `src/axiomrl/assets/configs/bcq/pendulum.yaml`
- Create: `tests/test_bcq_update.py`
- Create: `tests/test_bcq_trainer_smoke.py`

**Step 1: Write the failing tests**

Add coverage for:

- `bcq_loss(...)` exposing stable metric names
- invalid BCQ hyperparameters failing fast
- BCQ trainer writing checkpoints and offline evaluation metrics
- registry / public API / packaged config wiring for `bcq`

**Step 2: Run focused tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- a BCQ model containing behavior VAE, perturbation actor, and twin critics
- batch-constrained action candidate generation
- offline `train_bcq(...)` using the shared dataset path and schedule controls
- checkpoint load / evaluate / predict support through the registry

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

**Step 5: Commit**

Commit the first offline constrained baseline as its own unit.

### Task 4: Add `BEAR` as the support-matching offline baseline

**Files:**
- Create: `src/axiomrl/models/mlp_bear.py`
- Modify: `src/axiomrl/models/__init__.py`
- Create: `src/axiomrl/algorithms/bear.py`
- Create: `src/axiomrl/runtime/bear_trainer.py`
- Modify: `src/axiomrl/algorithms/__init__.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/bear/pendulum.yaml`
- Create: `src/axiomrl/assets/configs/bear/pendulum.yaml`
- Create: `tests/test_bear_update.py`
- Create: `tests/test_bear_trainer_smoke.py`

**Step 1: Write the failing tests**

Add coverage for:

- `bear_loss(...)` metric stability
- MMD-support constraint hyperparameter validation
- BEAR trainer producing checkpoints and evaluation metrics
- registry and managed API support for `bear`

**Step 2: Run focused tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- a BEAR model with behavior policy learning plus support-constrained actor updates
- MMD penalty computation against behavior-policy samples
- offline `train_bear(...)` reusing the same dataset and schedule controls as BCQ
- checkpoint load / evaluate / predict support through the registry

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

**Step 5: Commit**

Commit `BEAR` separately so offline constrained baselines remain easy to review.

### Task 5: Product surface, configs, docs, and roadmap polish

**Files:**
- Modify: `README.md`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_training_controls.py`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Step 1: Write the failing tests**

Add or extend coverage so it asserts:

- `BCQ` and `BEAR` are exported through the root package and managed APIs
- packaged configs include `bcq` and `bear`
- workflow helpers support checkpoint evaluate / resume for the new algorithms
- README documents dataset mixing, reward presets, schedules, `BCQ`, and `BEAR`

**Step 2: Run focused tests to verify they fail**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- concise README examples for offline dataset mixing and reward presets
- a short section documenting schedule keys and budget guards
- roadmap status updates showing that the package is moving from `BC/AWAC/HER` to `BCQ/BEAR`

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

**Step 5: Commit**

Commit docs and product-surface polish after the code path is stable.

## Next Follow-On After Phase 5

Once tests are allowed and this phase is verified, the next practical intake
wave should shift from classical offline RL completion to the next mainstream
gaps:

1. `TRPO` for on-policy completeness
2. `Discrete SAC` for a modern discrete actor-critic baseline
3. `CrossQ` or `DrQ-v2` as the next low-friction continuous-control / data-efficient
   addition
4. only after that, larger runtime shifts such as `IMPALA`, `APPO`, or
   world-model families
