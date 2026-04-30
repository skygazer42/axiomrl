# Clip Coefficient Scheduling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable clipping-coefficient schedules to PPO-family trainers so the trust region can be widened early and annealed later without changing algorithm code.

**Architecture:** Reuse the existing schedule machinery in `src/axiomrl/runtime/schedules.py` and mirror the entropy-scheduling pattern already present in `src/axiomrl/runtime/controls.py`. Add one shared resolver for `clip_coef` / `clip_range` aliases, then wire it into clipped-policy trainers by updating `algorithm.clip_coef` before each optimization phase and logging the active value.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, existing runtime trainer loops, pytest.

---

### Task 1: Add failing control and smoke tests

**Files:**
- Modify: `tests/test_training_controls.py`
- Modify: `tests/test_trainer_smoke.py`
- Modify: `tests/test_appo_trainer_smoke.py`

**Step 1: Write the failing tests**
- Add a control-level test for `resolve_clip_coefficient(...)` covering:
  - fixed `clip_coef`
  - alias `clip_range`
  - scheduled `clip_coef_schedule`
  - warmup via `clip_coef_warmup_steps`
- Add a PPO smoke test that configures `clip_coef_schedule` and asserts the final metrics expose the scheduled value.
- Add an APPO smoke test that configures `clip_coef_schedule` and asserts the final metrics expose the scheduled value.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_training_controls.py tests/test_trainer_smoke.py tests/test_appo_trainer_smoke.py`
- Expected: failures because the new resolver and trainer metric plumbing do not exist yet.

### Task 2: Implement shared clip resolver

**Files:**
- Modify: `src/axiomrl/runtime/controls.py`

**Step 1: Write minimal implementation**
- Add `resolve_clip_coefficient(...)`.
- Support:
  - `clip_coef`
  - `clip_range` alias
  - `clip_coef_schedule`
  - `clip_range_schedule` alias
  - `clip_coef_warmup_steps`
  - `clip_range_warmup_steps` alias
- Fall back to the fixed coefficient when no schedule is provided.
- Validate the resolved coefficient is non-negative.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_training_controls.py`

### Task 3: Wire clip schedules into trainer loops

**Files:**
- Modify: `src/axiomrl/runtime/ppo_trainer.py`
- Modify: `src/axiomrl/runtime/appo_trainer.py`
- Modify: `src/axiomrl/runtime/ppg_trainer.py`
- Modify: `src/axiomrl/runtime/recurrent_ppo_trainer.py`
- Modify: `src/axiomrl/runtime/gail_trainer.py`

**Step 1: Minimal wiring**
- Resolve the scheduled `clip_coef` from `global_step`.
- Update the algorithm instance immediately before policy updates.
- Include `clip_coef` in logged metrics.

**Step 2: Run targeted smoke tests**
- Run: `pytest -q tests/test_trainer_smoke.py tests/test_appo_trainer_smoke.py tests/test_ppg_trainer_smoke.py tests/test_recurrent_ppo_trainer_smoke.py tests/test_gail_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py`

### Task 4: Document config usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `clip_coef_schedule`.
- Mention `clip_range` as a compatibility alias.
- Clarify fixed `clip_coef` / `clip_range` still work.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_training_controls.py tests/test_trainer_smoke.py tests/test_appo_trainer_smoke.py tests/test_ppg_trainer_smoke.py tests/test_recurrent_ppo_trainer_smoke.py tests/test_gail_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
