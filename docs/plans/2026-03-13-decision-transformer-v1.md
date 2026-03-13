# Decision Transformer V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a narrow but honest `Decision Transformer` baseline for offline vector-observation, continuous-action control by introducing trajectory-window sequence modeling on top of the existing offline dataset stack.

**Architecture:** Keep this release deliberately small and aligned with the current package shape. Reuse the existing offline dataset builder, reward / returns-to-go processing, checkpointing, managed API, and continuous-control evaluation path, but add a dedicated trajectory-window data utility, a compact causal transformer policy model, a simple supervised action-prediction learner, and an offline trainer that treats `total_timesteps` as gradient updates. Implement return-conditioned autoregressive action prediction over fixed-length windows with masks and timestep embeddings. Explicitly do not implement discrete-action Decision Transformer, image observations, online fine-tuning, or full token-triplet Atari-scale training in this batch.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, pytest, existing `rl_training` offline dataset and experiment infrastructure.

---

### Task 1: Add failing Decision Transformer coverage

**Files:**
- Create: `tests/test_trajectory_window_dataset.py`
- Create: `tests/test_decision_transformer_update.py`
- Create: `tests/test_decision_transformer_trainer_smoke.py`
- Create: `tests/test_decision_transformer_reference_script.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_smoke.py`

**Step 1: Write the failing test**
- trajectory-window utility slices offline transitions into padded fixed-length windows with masks, timesteps, and returns-to-go.
- `DecisionTransformerModel.predict_actions()` returns continuous actions for masked windows.
- `decision_transformer_loss()` returns named metrics and `DecisionTransformer.update()` returns action-prediction metrics.
- `train_decision_transformer()` writes a checkpoint and evaluation metrics on `Pendulum-v1` with random offline data.
- root/api/algorithms package exports include `DecisionTransformer`.
- checkpoint workflows can evaluate and resume a saved `decision_transformer` checkpoint.
- packaged config resolves outside repo root and reference script runs as a smoke command.

**Step 2: Run test to verify it fails**
Run: `pytest -q tests/test_trajectory_window_dataset.py tests/test_decision_transformer_update.py tests/test_decision_transformer_trainer_smoke.py tests/test_decision_transformer_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py tests/test_cli.py tests/test_package_smoke.py`
Expected: FAIL with missing `decision_transformer` modules / exports.

### Task 2: Implement trajectory-window data utility and transformer model

**Files:**
- Create: `src/rl_training/data/trajectory_windows.py`
- Create: `src/rl_training/models/decision_transformer.py`
- Modify: `src/rl_training/data/__init__.py`
- Modify: `src/rl_training/models/__init__.py`

**Step 1: Write minimal implementation**
- add a utility that converts `TransitionDataset` plus discounted returns-to-go into fixed-length windows with masks and per-step timesteps.
- treat `dones` as episode boundaries and pad shorter prefixes safely.
- implement a compact causal transformer model for vector observations, continuous actions, and returns-to-go.
- expose a `predict_actions()` method that returns one action prediction per step and a helper for last-step action prediction.

**Step 2: Run tests to verify it passes**
Run: `pytest -q tests/test_trajectory_window_dataset.py tests/test_decision_transformer_update.py`
Expected: PASS.

### Task 3: Implement learner, trainer, and integration

**Files:**
- Create: `src/rl_training/algorithms/decision_transformer.py`
- Create: `src/rl_training/runtime/decision_transformer_trainer.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Modify: `src/rl_training/algorithms/__init__.py`

**Step 1: Write minimal implementation**
- implement supervised masked action-prediction updates with MSE on valid timesteps only.
- trainer reuses `_build_offline_dataset(...)` and derives returns-to-go if not present.
- treat `total_timesteps` as offline gradient updates and support `train / eval / resume / predict`.
- evaluation uses a small autoregressive rollout buffer with configurable `target_return`, `context_length`, and timestep cap.
- add managed API class `DecisionTransformer` and export wiring across root/api/algorithms surfaces.

**Step 2: Run tests to verify it passes**
Run: `pytest -q tests/test_decision_transformer_trainer_smoke.py tests/test_package_api_exports.py tests/test_public_api.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py`
Expected: PASS.

### Task 4: Add config assets, example, and docs

**Files:**
- Create: `configs/decision_transformer/pendulum.yaml`
- Create: `src/rl_training/assets/configs/decision_transformer/pendulum.yaml`
- Create: `examples/decision_transformer_pendulum_reference.py`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Step 1: Write minimal implementation**
- add a runnable offline `decision_transformer` Pendulum preset using random dataset generation.
- add a reference script for a tiny offline run.
- update README and yearly sourcebook to mark `Decision Transformer` as implemented narrow v1 and keep scope explicit.

**Step 2: Run tests to verify it passes**
Run: `pytest -q tests/test_decision_transformer_reference_script.py tests/test_cli.py tests/test_package_smoke.py`
Expected: PASS.

### Task 5: Regression verification

**Files:**
- Modify only if verification reveals regressions.

**Step 1: Run focused regression coverage**
Run: `pytest -q tests/test_trajectory_window_dataset.py tests/test_decision_transformer_update.py tests/test_decision_transformer_trainer_smoke.py tests/test_decision_transformer_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py tests/test_cli.py tests/test_package_smoke.py tests/test_bc_trainer_smoke.py tests/test_bc_update.py tests/test_iql_trainer_smoke.py`
Expected: PASS.

**Step 2: Run full suite**
Run: `pytest -q`
Expected: PASS.
