# Search Simulation Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable `num_simulations` schedules to MuZero-family trainers so search budget can be annealed over training from YAML config without changing algorithm code.

**Architecture:** Reuse the scheduler pattern already established in `src/axiomrl/runtime/controls.py`. Add a shared resolver for `num_simulations`, then thread the active integer value into MuZero-style planning on every environment step and log it in metrics. Planning should support per-call overrides instead of mutating persistent algorithm config state.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, existing MuZero-family runtime/trainers, pytest.

---

### Task 1: Add failing control and smoke tests

**Files:**
- Modify: `tests/test_training_controls.py`
- Modify: `tests/test_muzero_trainer_smoke.py`
- Modify: `tests/test_gumbel_muzero_trainer_smoke.py`
- Modify: `tests/test_efficientzero_trainer_smoke.py`
- Modify: `tests/test_scalezero_trainer_smoke.py`

**Step 1: Write the failing tests**
- Add a control-level test for `resolve_num_simulations(...)` covering:
  - fixed `num_simulations`
  - `num_simulations_schedule`
  - `num_simulations_warmup_steps`
- Add MuZero-family smoke tests that configure a schedule and assert final metrics expose the scheduled simulation count.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Expected: failures because the resolver and runtime metric plumbing do not exist yet.

### Task 2: Implement shared simulation resolver

**Files:**
- Modify: `src/axiomrl/runtime/controls.py`

**Step 1: Write minimal implementation**
- Add `resolve_num_simulations(...)`.
- Support:
  - `num_simulations`
  - `num_simulations_schedule`
  - `num_simulations_warmup_steps`
- Convert the scheduled scalar to an integer search budget and clamp to at least `1`.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_training_controls.py`

### Task 3: Wire schedules into MuZero-family planning

**Files:**
- Modify: `src/axiomrl/runtime/muzero_trainer.py`
- Modify: `src/axiomrl/runtime/efficientzero_trainer.py`
- Modify: `src/axiomrl/algorithms/muzero.py`
- Modify: `src/axiomrl/algorithms/gumbel_muzero.py`

**Step 1: Minimal wiring**
- Resolve the scheduled `num_simulations` before each planning step.
- Feed the active count into planning for `muzero`, `gumbel_muzero`, `efficientzero`, and `scalezero`.
- Include `num_simulations` in logged metrics.

**Step 2: Run targeted smoke tests**
- Run: `pytest -q tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`

### Task 4: Document config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `num_simulations_schedule` for MuZero-family trainers.
- Clarify fixed `num_simulations` still works.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
