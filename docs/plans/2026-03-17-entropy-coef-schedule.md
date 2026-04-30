# Entropy Coefficient Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable entropy-coefficient schedules to trainer runtimes so exploration pressure can be ramped or annealed without changing algorithms.

**Architecture:** Reuse `src/axiomrl/runtime/schedules.py` and the existing controls pattern. Add one shared resolver in `src/axiomrl/runtime/controls.py`, then wire it into PPO-style and Dreamer-style trainers by updating `algorithm.ent_coef` / `algorithm.entropy_coef` immediately before updates and surfacing the resolved value in logged metrics.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, existing runtime trainer loops, pytest.

---

### Task 1: Add failing control tests

**Files:**
- Modify: `tests/test_training_controls.py`
- Modify: `tests/test_trainer_smoke.py`
- Modify: `tests/test_dreamer_trainer_smoke.py`

**Step 1: Write the failing tests**
- Add a control-level test for scheduled entropy resolution, including warmup and both `ent_coef_schedule` / `entropy_coef_schedule`.
- Add a PPO smoke test that configures `ent_coef_schedule` and asserts the final metrics expose the scheduled coefficient.
- Add a Dreamer smoke test that configures `entropy_coef_schedule` and asserts the final metrics expose the scheduled coefficient.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_training_controls.py tests/test_trainer_smoke.py tests/test_dreamer_trainer_smoke.py`
- Expected: failures because the resolver and trainer metric plumbing do not exist yet.

### Task 2: Implement shared entropy schedule resolver

**Files:**
- Modify: `src/axiomrl/runtime/controls.py`

**Step 1: Write minimal implementation**
- Add `resolve_entropy_coefficient(...)`.
- Support both naming styles:
  - PPO-like: `ent_coef`, `ent_coef_schedule`, `ent_coef_warmup_steps`
  - Dreamer-like: `entropy_coef`, `entropy_coef_schedule`, `entropy_coef_warmup_steps`
- Keep backward compatibility by falling back to the fixed coefficient when no schedule is configured.
- Validate the resolved coefficient is non-negative.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_training_controls.py`
- Expected: control tests pass.

### Task 3: Wire schedules into trainer loops

**Files:**
- Modify: `src/axiomrl/runtime/ppo_trainer.py`
- Modify: `src/axiomrl/runtime/a2c_trainer.py`
- Modify: `src/axiomrl/runtime/impala_trainer.py`
- Modify: `src/axiomrl/runtime/ppg_trainer.py`
- Modify: `src/axiomrl/runtime/appo_trainer.py`
- Modify: `src/axiomrl/runtime/recurrent_ppo_trainer.py`
- Modify: `src/axiomrl/runtime/gail_trainer.py`
- Modify: `src/axiomrl/runtime/trpo_trainer.py`
- Modify: `src/axiomrl/runtime/dreamer_trainer.py`

**Step 1: Minimal wiring**
- Resolve the scheduled coefficient from `global_step`.
- Update the algorithm instance before each policy / actor-critic optimization step.
- Include the active coefficient in final metrics as `ent_coef` or `entropy_coef`.

**Step 2: Run targeted smoke tests**
- Run: `pytest -q tests/test_trainer_smoke.py tests/test_a2c_trainer_smoke.py tests/test_impala_trainer_smoke.py tests/test_ppg_trainer_smoke.py tests/test_appo_trainer_smoke.py tests/test_recurrent_ppo_trainer_smoke.py tests/test_gail_trainer_smoke.py tests/test_dreamer_trainer_smoke.py`

### Task 4: Document config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `ent_coef_schedule` / `entropy_coef_schedule`.
- Show warmup usage and clarify fixed `ent_coef` / `entropy_coef` still works.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_training_controls.py tests/test_trainer_smoke.py tests/test_a2c_trainer_smoke.py tests/test_impala_trainer_smoke.py tests/test_ppg_trainer_smoke.py tests/test_appo_trainer_smoke.py tests/test_recurrent_ppo_trainer_smoke.py tests/test_gail_trainer_smoke.py tests/test_dreamer_trainer_smoke.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
