# Root Exploration Fraction Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable root Dirichlet-noise fraction schedules to MuZero-family trainers so search-time exploration can be annealed over training from YAML config alone.

**Architecture:** Reuse the existing scheduler pattern in `src/rl_training/runtime/controls.py`. Add a shared resolver for `root_exploration_fraction`, then thread the active value into MuZero-style planning calls on every environment step and log it in training metrics. Update `GumbelMuZero` to actually honor root-noise settings during planning so the trainer control has effect there too.

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
- Add a control-level test for `resolve_root_exploration_fraction(...)` covering:
  - fixed `root_exploration_fraction`
  - `root_exploration_fraction_schedule`
  - `root_exploration_fraction_warmup_steps`
- Add MuZero-family smoke tests that configure a schedule and assert final metrics expose the scheduled fraction.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Expected: failures because the resolver and runtime metric plumbing do not exist yet.

### Task 2: Implement shared root-noise resolver

**Files:**
- Modify: `src/rl_training/runtime/controls.py`

**Step 1: Write minimal implementation**
- Add `resolve_root_exploration_fraction(...)`.
- Support:
  - `root_exploration_fraction`
  - `root_exploration_fraction_schedule`
  - `root_exploration_fraction_warmup_steps`
- Fall back to the fixed scalar when no schedule is configured.
- Validate the resolved value is in `[0, 1]`.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_training_controls.py`

### Task 3: Wire schedules into MuZero-family planning

**Files:**
- Modify: `src/rl_training/runtime/muzero_trainer.py`
- Modify: `src/rl_training/runtime/efficientzero_trainer.py`
- Modify: `src/rl_training/algorithms/muzero.py`
- Modify: `src/rl_training/algorithms/gumbel_muzero.py`

**Step 1: Minimal wiring**
- Resolve the scheduled root exploration fraction before each planning step.
- Feed the active fraction into planning for `muzero`, `gumbel_muzero`, `efficientzero`, and `scalezero`.
- Ensure `GumbelMuZero` honors `add_root_noise` instead of always discarding it.
- Include `root_exploration_fraction` in logged metrics.

**Step 2: Run targeted smoke tests**
- Run: `pytest -q tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`

### Task 4: Document config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `root_exploration_fraction_schedule` for MuZero-family trainers.
- Clarify fixed `root_exploration_fraction` still works.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
