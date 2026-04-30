# MARWIL Phase 15 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a narrow but package-usable `MARWIL` implementation for offline continuous-control training by reusing the package's current offline dataset, returns-to-go processing, and actor/value model lane.

**Architecture:** Reuse `MLPIQLModel` for tanh-Gaussian policy plus value regression, and keep the packaged `MARWIL` path intentionally narrow: regress value to discounted returns-to-go, compute advantages as `returns_to_go - value(obs)`, normalize those advantages with a running squared-advantage scale, then perform exponentiated advantage-weighted behavior cloning. Reuse the current offline dataset builder, reward processing, checkpoint, eval, predict, schedule, and early-stopping surfaces instead of creating a new trainer family.

**Tech Stack:** Python, PyTorch, Gymnasium, existing `axiomrl` offline dataset stack, returns-to-go processing, and experiment infrastructure

---

### Task 1: Freeze The Narrow `MARWIL` Scope

**Files:**
- Create: `docs/plans/2026-03-12-marwil-phase15.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-12-mainstream-rl-package-design.md`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Step 1: Freeze v1 boundaries**

Document the first packaged `MARWIL` release as:

- continuous `Box` actions only
- flat vector observations only
- offline dataset training only
- single-process trainer only
- discounted returns-to-go computed from the processed reward stream in v1
- actor/value updates only, with no Q-critic path in this phase
- no recurrent path
- no image observations
- no distributed runtime

**Step 2: Record the package rationale**

Explain that `MARWIL` is the next low-friction offline / imitation intake because:

- RLlib still treats it as a public offline / imitation algorithm surface
- it productizes another recognizable bridge between `BC` and weighted offline RL
- it reuses the current actor/value mental model instead of requiring a new sequence or distributed stack

**Step 3: Keep test execution deferred**

Record that tests are added but intentionally not executed until the user explicitly requests it.

### Task 2: Add The `MARWIL` Learner

**Files:**
- Create: `src/axiomrl/algorithms/marwil.py`
- Modify: `src/axiomrl/algorithms/__init__.py`

**Step 1: Reuse the actor/value model family**

Build `MARWIL` on `MLPIQLModel` and keep the learner narrow by using the policy head plus value head only in v1.

**Step 2: Implement package-narrow MARWIL losses**

Keep the package v1 behavior small and explicit:

- regress the value head to discounted returns-to-go
- estimate per-sample advantages as `returns_to_go - value(obs)`
- maintain a moving-average squared-advantage norm
- train the actor with `exp(beta * normalized_advantage)` weighted behavior log-probabilities
- preserve the package-recognizable `beta == 0` to `BC` style behavior

**Step 3: Expose public loss helpers**

Add a readable `marwil_loss(...)` helper and export `MARWIL` / `MARWILAlgorithm` through the shared algorithms package.

### Task 3: Add The Offline `MARWIL` Trainer

**Files:**
- Create: `src/axiomrl/runtime/marwil_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`

**Step 1: Reuse offline dataset and returns processing**

Build the trainer on `_infer_env_spaces(...)` and `_build_offline_dataset(...)` from the current offline path, then derive discounted returns-to-go from the processed reward stream.

**Step 2: Preserve shared controls**

Keep support for:

- `eval_interval`
- early stopping callbacks
- offline epoch / update budgets
- learning-rate schedules
- checkpoint save / resume

**Step 3: Reuse standard evaluation and prediction**

Evaluate and predict through the current deterministic actor helper path already used by `IQL` / `AWR` family algorithms.

### Task 4: Wire `MARWIL` Into The Package Surface

**Files:**
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/marwil/pendulum.yaml`
- Create: `src/axiomrl/assets/configs/marwil/pendulum.yaml`
- Modify: `README.md`

**Step 1: Add the managed API entrypoint**

Expose `MARWIL` through the root package and API namespaces.

**Step 2: Ship a starter config**

Add a packaged offline `Pendulum-v1` config with narrow `MARWIL` defaults.

**Step 3: Update package docs**

Document `MARWIL` as a weighted offline imitation / RL bridge on the same dataset path.

### Task 5: Add Unexecuted Coverage

**Files:**
- Create: `tests/test_marwil_update.py`
- Create: `tests/test_marwil_trainer_smoke.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `tests/test_cli.py`

**Step 1: Add learner-level coverage**

Add a unit test for `marwil_loss(...)`, invalid hyperparameters, running advantage-norm state, and one update call.

**Step 2: Add trainer smoke coverage**

Add a small offline smoke test that checks checkpoint creation, running advantage-norm metrics, and eval wiring.

**Step 3: Extend package-surface expectations**

Update public exports, managed API, checkpoint workflow, and packaged-config tests so `MARWIL` is treated as a shipped algorithm.

**Step 4: Keep execution deferred**

Do not run the tests until the user explicitly asks for test execution.
