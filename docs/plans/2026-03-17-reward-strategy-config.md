# Reward Strategy Configuration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add declarative environment-level reward strategy configuration so training runs can shape scenario rewards from YAML without changing environment code.

**Architecture:** Extend the existing reward wrapper pipeline in `src/axiomrl/envs/rewards.py` instead of introducing a second config path. Keep current scalar transforms (`sign`, `scale`, `shift`, `clip`, named presets) backward compatible, add a `strategy` alias for presets, and add step-aware shaping for common RL workflows such as per-step penalties and success/failure bonuses driven by `info`.

**Tech Stack:** Python 3.10+, Gymnasium wrappers, existing env factory/config system, pytest.

---

### Task 1: Add failing reward strategy tests

**Files:**
- Modify: `tests/test_reward_wrappers.py`

**Step 1: Write the failing tests**
- Add a resolver test for `strategy` as an alias of `preset`.
- Add resolver coverage for new shaping fields such as `step_penalty`, `success_bonus`, and `failure_penalty`.
- Add wrapper behavior tests showing success-aware shaping reads `info["is_success"]`.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_reward_wrappers.py`
- Expected: failures because the alias and step-aware shaping do not exist yet.

### Task 2: Implement reward strategy parsing and wrappers

**Files:**
- Modify: `src/axiomrl/envs/rewards.py`
- Modify: `src/axiomrl/envs/__init__.py`

**Step 1: Write minimal implementation**
- Extend `RewardTransformConfig` with step-aware shaping fields.
- Add `strategy` alias handling while preserving `preset`.
- Implement a step-aware wrapper that can apply `step_penalty`, `terminal_bonus`, `success_bonus`, and `failure_penalty`.
- Keep transform ordering deterministic and backward compatible.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_reward_wrappers.py`

### Task 3: Verify factory integration

**Files:**
- Modify: `tests/test_reward_wrappers.py`

**Step 1: Add/update integration coverage**
- Exercise `build_env(...)` with reward strategy config so the env factory path stays covered.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_reward_wrappers.py`

### Task 4: Document YAML usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document `strategy` and `preset`.
- Show a sparse-goal / success-bonus example and a per-step penalty example.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_reward_wrappers.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
