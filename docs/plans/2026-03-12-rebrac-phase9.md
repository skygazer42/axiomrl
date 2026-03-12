# ReBRAC Phase 9 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a narrow but package-usable `ReBRAC` implementation for offline continuous-control training, plus the minimum offline dataset support needed to carry behavior actions for next states.

**Architecture:** Reuse the current `TD3+BC` offline runtime family instead of creating another trainer stack. Implement `ReBRAC` as a behavior-regularized TD3-style learner on top of `MLPTD3Model`, packaged configs, managed API wiring, and checkpoint/eval/predict paths, while extending `TransitionDataset` with optional `next_actions` so the learner can regularize bootstrapped targets without redesigning the whole data layer.

**Tech Stack:** Python, PyTorch, Gymnasium, existing `rl_training` offline dataset and experiment infrastructure

---

### Task 1: Freeze The Narrow `ReBRAC` Scope

**Files:**
- Create: `docs/plans/2026-03-12-rebrac-phase9.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Step 1: Freeze v1 boundaries**

Document the first packaged `ReBRAC` release as:

- continuous `Box` actions only
- flat vector observations only
- offline dataset training only
- single-process trainer only
- `TD3`-style target smoothing with behavior regularization
- no recurrent path
- no image observations

**Step 2: Record the package rationale**

Explain that `ReBRAC` is the next low-friction 2023 offline wave because:

- current public offline libraries still surface it
- it complements the existing `TD3+BC` baseline
- it reuses the current offline config / checkpoint / trainer stack

**Step 3: Keep test execution deferred**

Record that tests are added but intentionally not executed until the user explicitly requests it.

### Task 2: Extend Offline Dataset Payloads For Next-State Behavior Actions

**Files:**
- Modify: `src/rl_training/data/offline_dataset.py`
- Modify: `src/rl_training/data/dataset_loaders.py`
- Modify: `src/rl_training/data/offline_mixers.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/data/__init__.py`
- Modify: `tests/test_offline_dataset.py`
- Modify: `tests/test_dataset_loaders.py`

**Step 1: Add optional `next_actions` storage**

Allow `TransitionDataset` to carry optional `next_actions` tensors without breaking existing callers that only expect the standard five transition fields.

**Step 2: Preserve the field through loaders and mixing**

When file payloads or Minari episodes provide next-state behavior actions, keep them through:

- `TransitionDataset.from_dict(...)`
- file-backed dataset loading
- mixed dataset assembly
- sampling to trainer batches

**Step 3: Fill the field in random dataset generation**

Generate sequential random offline datasets with behavior-consistent `next_actions` so package-provided smoke datasets already exercise the richer offline payload.

### Task 3: Add The `ReBRAC` Learner

**Files:**
- Create: `src/rl_training/algorithms/rebrac.py`
- Modify: `src/rl_training/algorithms/__init__.py`

**Step 1: Reuse the existing TD3 model family**

Build `ReBRAC` on `MLPTD3Model` with target critics, target actor smoothing, and deterministic continuous actions.

**Step 2: Implement behavior-regularized critic and actor objectives**

Keep the package v1 knobs small and explicit:

- `actor_bc_weight`
- `critic_bc_weight`
- `actor_q_weight`
- `policy_noise`
- `noise_clip`
- `policy_delay`

Use sampled `next_actions` when present, and document the fallback behavior when a dataset batch does not carry them.

**Step 3: Expose public loss helpers**

Add a readable `rebrac_loss(...)` function and export `ReBRAC` / `ReBRACAlgorithm` through the shared algorithms package.

### Task 4: Add The Offline `ReBRAC` Trainer

**Files:**
- Create: `src/rl_training/runtime/rebrac_trainer.py`
- Modify: `src/rl_training/experiment/registry.py`

**Step 1: Reuse the offline dataset stack**

Build the trainer on `_infer_env_spaces(...)` and `_build_offline_dataset(...)` from the current offline path.

**Step 2: Preserve shared controls**

Keep support for:

- `eval_interval`
- early stopping callbacks
- offline epoch / update budgets
- learning-rate schedules
- checkpoint save / resume

**Step 3: Reuse standard evaluation and prediction**

Evaluate with the current deterministic continuous-action TD3 helper and expose package prediction through checkpoint workflows.

### Task 5: Wire `ReBRAC` Into The Package Surface

**Files:**
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/rebrac/pendulum.yaml`
- Create: `src/rl_training/assets/configs/rebrac/pendulum.yaml`
- Modify: `README.md`

**Step 1: Add the managed API entrypoint**

Expose `ReBRAC` through the root package and API namespaces.

**Step 2: Ship a starter config**

Add a packaged offline `Pendulum-v1` config that uses the random dataset path and the narrow `ReBRAC` defaults.

**Step 3: Update package docs**

Document `ReBRAC` as a 2023 offline follow-on to `TD3+BC` and explain the optional `next_actions` dataset field.

### Task 6: Add Unexecuted Coverage

**Files:**
- Create: `tests/test_rebrac_update.py`
- Create: `tests/test_rebrac_trainer_smoke.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `tests/test_cli.py`

**Step 1: Add learner-level coverage**

Add a unit test for `rebrac_loss(...)` metric keys and one update call.

**Step 2: Add trainer smoke coverage**

Add a small offline smoke test that checks checkpoint creation and eval wiring.

**Step 3: Extend package-surface expectations**

Update public exports, managed API, checkpoint workflow, and packaged-config tests so `ReBRAC` is treated as a shipped algorithm.

**Step 4: Keep execution deferred**

Do not run the tests until the user explicitly asks for test execution.
