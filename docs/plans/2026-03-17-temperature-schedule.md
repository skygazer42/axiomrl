# Search Temperature Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable search-temperature schedules to MuZero-family trainers so action-selection stochasticity can be annealed over training without editing algorithm code.

**Architecture:** Reuse `src/rl_training/runtime/schedules.py` and the existing controls pattern in `src/rl_training/runtime/controls.py`. Add one shared resolver for `temperature`, then wire it into MuZero-style trainers by resolving the current temperature from `global_step`, feeding it into planning, and logging the active value in metrics.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, existing MuZero-family runtimes, pytest.

---

### Task 1: Add failing control and smoke tests

**Files:**
- Modify: `tests/test_training_controls.py`
- Modify: `tests/test_muzero_trainer_smoke.py`
- Modify: `tests/test_gumbel_muzero_trainer_smoke.py`
- Modify: `tests/test_efficientzero_trainer_smoke.py`
- Modify: `tests/test_scalezero_trainer_smoke.py`

**Step 1: Write the failing tests**
- Add a control-level test for `resolve_temperature(...)` covering:
  - fixed `temperature`
  - `temperature_schedule`
  - `temperature_warmup_steps`
- Add smoke tests that configure `temperature_schedule` and assert final metrics expose the scheduled temperature for MuZero-family trainers.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Expected: failures because the resolver and trainer plumbing do not exist yet.

### Task 2: Implement shared temperature resolver

**Files:**
- Modify: `src/rl_training/runtime/controls.py`

**Step 1: Write minimal implementation**
- Add `resolve_temperature(...)`.
- Support:
  - `temperature`
  - `temperature_schedule`
  - `temperature_warmup_steps`
- Fall back to the fixed scalar when no schedule is provided.
- Validate the resolved value is non-negative.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_training_controls.py`

### Task 3: Wire schedules into MuZero-family runtimes

**Files:**
- Modify: `src/rl_training/runtime/muzero_trainer.py`
- Modify: `src/rl_training/runtime/efficientzero_trainer.py`
- Modify: `src/rl_training/algorithms/gumbel_muzero.py`

**Step 1: Minimal wiring**
- Resolve the scheduled temperature before each planning step.
- Pass the resolved temperature into planning for `muzero`, `gumbel_muzero`, `scalezero`, and `efficientzero`.
- Update Gumbel MuZero so temperature actually modulates stochastic root-action sampling instead of being ignored.
- Include `temperature` in logged metrics.

**Step 2: Run targeted smoke tests**
- Run: `pytest -q tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`

### Task 4: Document config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `temperature_schedule` for MuZero-family trainers.
- Clarify that fixed `temperature` still works.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_training_controls.py tests/test_muzero_trainer_smoke.py tests/test_gumbel_muzero_trainer_smoke.py tests/test_efficientzero_trainer_smoke.py tests/test_scalezero_trainer_smoke.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
